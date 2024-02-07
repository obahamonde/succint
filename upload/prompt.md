My source code:

{
  "path": "src",
  "name": "src",
  "is_dir": true,
  "content": [
    {
      "path": "src/utils",
      "name": "utils",
      "is_dir": true,
      "content": [
        {
          "path": "src/utils/templates.py",
          "name": "templates.py",
          "is_dir": false,
          "content": "def html_string(base_url: str):\n    \"\"\"\n    Returns a string with the html code to render the swagger ui\n    \"\"\"\n    return f\"\"\"<!DOCTYPE html>\n            <html lang=\"en\">\n            <head>\n                <meta charset=\"UTF-8\">\n                <title>OpenAPI</title>\n                <link rel=\"stylesheet\" type=\"text/css\" href=\"https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css\" >\n                <link rel=\"icon\" type=\"image/png\" href=\"https://www.aiofauna.com/logo.png\" sizes=\"32x32\" />\n            </head>\n            <body>\n                <div id=\"swagger-ui\"></div>\n                <script src=\"https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js\"> </script>\n                <script src=\"https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-standalone-preset.js\"> </script>\n                <script>\n                window.onload = function() {{\n                const ui = SwaggerUIBundle({{\n                    url: \"{base_url}\",\n                    dom_id: '#swagger-ui',\n                    deepLinking: true,\n                    presets: [\n                    SwaggerUIBundle.presets.apis,\n                    SwaggerUIStandalonePreset\n                    ],\n                    plugins: [\n                    SwaggerUIBundle.plugins.DownloadUrl\n                    ],\n                    layout: \"StandaloneLayout\"\n                }})\n                window.ui = ui\n                }}\n            </script>\n            </body>\n            </html>\n            \"\"\"\n"
        },
        {
          "path": "src/utils/__init__.py",
          "name": "__init__.py",
          "is_dir": false,
          "content": "from .templates import html_string\n\n__all__ = [\"html_string\"]\n"
        }
      ]
    },
    {
      "path": "src/agent",
      "name": "agent",
      "is_dir": true,
      "content": [
        {
          "path": "src/agent/agent.py",
          "name": "agent.py",
          "is_dir": false,
          "content": "\"\"\"\nAgent module.\nA Large Language Model Agent (OLLAMA) is an agent based on mistral-7B-instruct-2.0 quantized to 4 bits that can be run locally and interacted with via a chat interface.\nThe agent can be used to run tools based on user input.\nThe agent is trained to provide useful responses and guidance to the user.\n\"\"\"\n\nimport asyncio\nimport json as json_module\nfrom functools import cached_property, lru_cache\nfrom typing import AsyncIterator, List, Literal, Optional, Type\n\nfrom agent_proto import BaseAgent, robust\nfrom agent_proto.agent import Message\nfrom agent_proto.tool import Tool, ToolDefinition, ToolOutput\nfrom agent_proto.utils import setup_logging\nfrom jinja2 import Template\nfrom openai import AsyncOpenAI\nfrom openai._streaming import AsyncStream\nfrom openai.types.chat.chat_completion import ChatCompletion\nfrom openai.types.chat.chat_completion_chunk import ChatCompletionChunk\nfrom pydantic import BaseModel, Field\nfrom typing_extensions import Iterable\n\nfrom ..services import PGRetrievalTool\nfrom ..tools import BrowserTool\nfrom ._prompts import CHAT_TEMPLATE, RUN_TEMPLATE\n\nlogger = setup_logging(__name__)\n\n\nclass AgentSchema(BaseModel):\n    model: str\n    tools: List[str]\n    messages: List[Message]\n\n\nclass Agent(BaseAgent[AsyncOpenAI]):\n    \"\"\"An agent that interacts with users and performs tool calls based on user input.\n\n    Attributes:\n        messages (list[Message]): List of messages exchanged between the user and the agent.\n        model (str): The model used by the agent.\n        tools (list[Type[Tool[Any]]]): List of available tool classes.\n\n    Methods:\n        chat: Send a message to the agent and return the response.\n        run: Run a tool based on a user message.\n        __call__: Run a specific tool class based on a user message.\n    \"\"\"\n\n    model: str = Field(default=\"mistralai/Mistral-7B-Instruct-v0.2\")\n    tools: Iterable[Type[Tool]] = Field(default_factory=Tool.__subclasses__)\n    messages: List[Message] = []\n\n    @cached_property\n    def run_template(self):\n        \"\"\"The template used for generating the message sent to the agent. Crafted using prompt engineering to guide the model to infer the schema for performing tool calls based on user's message.\"\"\"\n        return Template(RUN_TEMPLATE, enable_async=True)\n\n    @cached_property\n    def chat_template(self):\n        \"\"\"The template used for enabling the agent with context and instructions for responding to user's messages.\"\"\"\n        return Template(CHAT_TEMPLATE, enable_async=True)\n\n    def __call__(self):\n        \"\"\"Load the AsyncClient.\"\"\"\n        return AsyncOpenAI(\n            api_key=\".\",\n            base_url=\"http://mistral:8000/v1\",\n        )\n\n    @robust\n    async def chat(  # type: ignore\n        self, *, message: str, stream: bool = True\n    ) -> AsyncIterator[str] | Message:\n        \"\"\"\n        Chat with the agent.\n\n        Args:\n            message (str): The message sent to the agent.\n            stream (bool): Whether the response should be streamed or not.\n\n        Returns:\n            The response from the agent.\n            (AsyncIterator[Message]): If stream is True.\n            (Message): If stream is False.\n        Raises:\n            ValueError: If the response doesn't contain any content.\n        \"\"\"\n        response = await self().chat.completions.create(  # type: ignore\n            model=self.model,\n            messages=[{\"role\": \"user\", \"content\": message}],\n            stream=stream,\n            max_tokens=420,\n        )\n\n        if stream:\n            assert isinstance(response, AsyncStream)\n\n            async def _():\n                nonlocal response\n                async for choice in response:  # type: ignore\n                    assert isinstance(choice, ChatCompletionChunk)\n                    content = choice.choices[0].delta.content\n                    if content:\n                        yield Message(\n                            role=\"assistant\", content=content\n                        ).model_dump_json()\n\n                    else:\n                        continue\n                yield {\"event\": \"done\", \"data\": \"done\"}\n\n            return _()  # type: ignore\n\n        assert isinstance(response, ChatCompletion)\n        content = response.choices[0].message.content\n        if content:\n            message_ = Message(role=\"assistant\", content=content)\n            return message_\n        raise ValueError(\"No content in response\")\n\n    @robust\n    async def run(  # type: ignore\n        self, *, message: str, definitions: Optional[List[ToolDefinition]] = None\n    ) -> ToolOutput:\n        \"\"\"Executes a tool based on natural language input.\n        It works as follows:\n        1. The user provides a message.\n        2. The agent picks a tool from the list of definitions available, otherwise it returns a chat `Message` object.\n        3. The agent sends an inferred json object based on the tool definition and user input to the Tool class, which call it's constructor with the parsed json object.\n        4. The Tool executes the logic implemented on it's run method and returns the output as a ToolOutput object with the following structure:\n\n        ```json\n        {\n            \"role\": \"tool_name\",\n            \"content\": \"tool_output\"\n        }\n        ```\n\n        5. The agent returns the ToolOutput object to the user.\n\n        Args:\n            message (str): The message sent to the agent.\n\n        Returns:\n            ToolOutput: The output of the tool.\n        \"\"\"\n        if definitions is None:\n            definitions = [klass.definition() for klass in self.tools]\n        prompt_ = await self.run_template.render_async(\n            message=message,\n            definitions=definitions,\n        )\n        response = await self.chat(\n            message=prompt_,\n            stream=False,\n        )\n        assert isinstance(response, Message)\n        data = json_module.loads(response.content)\n        try:\n            for deff in definitions:\n                if deff[\"title\"].lower() == data[\"role\"]:\n                    tool = next(\n                        klass for klass in self.tools if klass.__name__ == deff[\"title\"]\n                    )\n                    assert issubclass(tool, Tool)\n                    return await tool(**data)()\n            output = await self.chat(message=message, stream=False)\n            assert isinstance(output, Message)\n            return ToolOutput(role=\"assistant\", content=output.content)\n        except (StopIteration, KeyError):\n            output = await self.chat(message=message, stream=False)\n            assert isinstance(output, Message)\n            return ToolOutput(role=\"assistant\", content=output.content)\n\n    @robust\n    async def instruct(self, *, message: str, instruction: str) -> str:\n        \"\"\"Instruct the agent to perform a tool call based on user input.\n\n        Args:\n            message (str): The message sent to the agent.\n\n        Returns:\n            str: The response from the agent.\n        \"\"\"\n        response = await self().completions.create(\n            prompt=f\"\"\"\n            \n            [INST]\n            {instruction}\n            [/INST]\n            <s>{message}</s>\n            \"\"\",\n            model=self.model,\n        )\n        return response.choices[0].text\n\n    async def search(self, query: str):\n        \"\"\"Search the web using the agent.\"\"\"\n        results = await BrowserTool(inputs=query)()\n        summarized = await asyncio.gather(\n            *[\n                self.instruct(\n                    message=el,\n                    instruction=f\"Summarize this content, the output must be 100% exclusively in the same language as the input: {query}\",\n                )\n                for el in results.content\n            ]\n        )\n        context = \"\\n\".join(summarized)\n        logger.info(f\"Results from the web: {context}\")\n        return f\"Results from the web: {context}\"\n\n    @lru_cache\n    def get_retriever(\n        self, query: str, action: Literal[\"query\", \"upsert\"], top_k: int = 5\n    ):\n        return PGRetrievalTool(\n            namespace=self.model, inputs=query, top_k=top_k, action=action\n        )\n\n    @robust\n    async def upsert(self, query: str):\n        \"\"\"Upsert a document to the knowledge store.\"\"\"\n        return await self.get_retriever(query, \"upsert\")()\n\n    @robust\n    async def query(self, query: str):\n        \"\"\"Query the knowledge store.\"\"\"\n        data = await self.get_retriever(query, \"query\")()\n        documents = data.content\n        return \"\\n\".join([doc.metadata[\"text\"] for doc in documents])\n"
        },
        {
          "path": "src/agent/_prompts.py",
          "name": "_prompts.py",
          "is_dir": false,
          "content": "RUN_TEMPLATE: str = \"\"\"\n\t\t<s> [INST]\n\t\tBased on user input determine if a tool call is gonna be performed. These are the tools that are available for this step: {{ definitions }}\n\t\t[/INST]\n\t\tThe user input is:\n\t\t{{ message }}\n\t\t</s>\n\t\t\n\t\t[INST]\n\t\tIf a tool call can be inferred according to Json Schema directly send a valid json object without any additional content.\n\t\tDon't include any kind of introductory comment or syntax no backticks (```) just the valid json object that was inferred in the following format:\n\t\t{ \"tool\": { \"name\": \"tool_name\", \"parameters\": { \"parameter_name\": \"parameter_value\" } } }\n\t\t\n\t\t[/INST]\n\t\t\"\"\"\n\nCHAT_TEMPLATE: str = \"\"\"\n[INST]\nUse the following context to have additional context about the user input:\nContext:\n{{ context }}\n[/INST]\nUser input:\n\n{{ prompt }} \n[INST]\nYour answer must be 100% in the same language as the user input.[/INST]\n\"\"\"\n"
        },
        {
          "path": "src/agent/__init__.py",
          "name": "__init__.py",
          "is_dir": false,
          "content": "from agent_proto.tool import Tool, ToolDefinition, ToolOutput\n\nfrom .agent import Agent\n\n__all__ = [\"Agent\", \"Tool\", \"ToolDefinition\", \"ToolOutput\"]\n"
        }
      ]
    },
    {
      "path": "src/out",
      "name": "out",
      "is_dir": true,
      "content": [
        {
          "path": "src/out/_parser.py",
          "name": "_parser.py",
          "is_dir": false,
          "content": ""
        }
      ]
    },
    {
      "path": "src/tools",
      "name": "tools",
      "is_dir": true,
      "content": [
        {
          "path": "src/tools/vision.py",
          "name": "vision.py",
          "is_dir": false,
          "content": "from functools import cached_property\n\nfrom agent_proto import Tool\n\n\nfrom ._base import Inference\n\n\nclass Vision(Tool):\n    \"\"\"Vision tool processes an image into a text description.\"\"\"\n\n    inputs: bytes\n\n    @cached_property\n    def client(self):\n        return Inference(model=\"Salesforce/blip-image-captioning-large\", expects=\"text\")\n\n    async def run(self):\n        response = await self.client.__load__().post(\"\", data=self.inputs)  # type: ignore\n        return response.text\n"
        },
        {
          "path": "src/tools/monitor.py",
          "name": "monitor.py",
          "is_dir": false,
          "content": "from functools import cached_property\nfrom typing import cast\n\nimport GPUtil  # disable=E0401 # type: ignore\nimport psutil\nimport torch\nfrom agent_proto import async_io, robust\nfrom pydantic import Field, computed_field\n\nfrom ..agent import Tool\n\n\nclass ResourceManager(Tool):\n    \"\"\"\n    Tool for monitoring system resources such as CPU, memory, GPU, and network usage.\n    Doesn't require any inputs, just run the tool is the user asks for the current resource usage or any related information.\n    \"\"\"\n\n    memory_usage: float = Field(\n        default=0.0, title=\"Memory Usage\", description=\"Memory usage in GB\"\n    )\n    cpu_usage: float = Field(\n        default=0.0, title=\"CPU Usage\", description=\"CPU usage in %\"\n    )\n    gpu_usage: float = Field(\n        default=0.0, title=\"GPU Usage\", description=\"GPU usage in %\"\n    )\n    network_usage: float = Field(\n        default=0.0, title=\"Network Usage\", description=\"Network usage in MB/s\"\n    )\n\n    async def run(self):\n        await self.update()\n        return self\n\n    @robust\n    @async_io\n    def update(self):\n        self.memory_usage = psutil.virtual_memory().used / self.memory\n        self.cpu_usage = psutil.cpu_percent()\n        self.network_usage = psutil.net_io_counters().bytes_sent / 1024 / 1024\n\n    @computed_field(return_type=str)\n    @cached_property\n    @torch.no_grad()  # type: ignore\n    def device(self) -> str:\n        return str(torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\"))\n\n    @computed_field(return_type=float)\n    def cpu_count(self) -> float:\n        return psutil.cpu_count()\n\n    @computed_field(return_type=float)\n    @property\n    def memory(self) -> float:\n        return cast(float, psutil.virtual_memory().total / 1024 / 1024 / 1024)  # type: ignore\n\n    @computed_field(return_type=float)\n    @property\n    def get_gpu_usage(self) -> float:\n        try:\n            return GPUtil.getGPUs()[0].load * 100  # type: ignore\n        except Exception:\n            return 0.0  # type: ignore\n"
        },
        {
          "path": "src/tools/openapi.py",
          "name": "openapi.py",
          "is_dir": false,
          "content": "import tiktoken\nfrom agent_proto import Tool, async_io\nfrom aiohttp import ClientSession\nfrom langchain.agents.agent_toolkits.openapi import planner\nfrom langchain.agents.agent_toolkits.openapi.spec import (\n    reduce_openapi_spec,\n)  # type: ignore\nfrom langchain.chat_models.openai import ChatOpenAI  # type: ignore\nfrom langchain.requests import RequestsWrapper\n\n\n@async_io\ndef tokenize(text: str) -> list[int]:\n    encoding = tiktoken.get_encoding(encoding_name=\"cl100k_base\")\n    return encoding.encode(text)\n\n\n@async_io\ndef sanitize_openapi_spec(spec: dict[str, object]) -> dict[str, object]:\n    return reduce_openapi_spec(spec)  # type: ignore\n\n\nclass Plugin(Tool):\n    \"\"\"OpenAPI plugin for the agent.\"\"\"\n\n    openapi_url: str\n    inputs: str\n    headers: dict[str, str]\n\n    async def run(self):\n        async with ClientSession(headers=self.headers) as session:\n            try:\n                spec = await (await session.get(self.openapi_url)).json()\n            except Exception:\n                spec = await (\n                    await session.get(\n                        f\"https://api.oscarbahamonde.com/static/{self.openapi_url}\"\n                    )\n                ).json()\n            executor = planner.create_openapi_agent(  # type: ignore\n                api_spec=await sanitize_openapi_spec(spec=spec),  # type: ignore\n                requests_wrapper=RequestsWrapper(\n                    aiosession=session, headers=self.headers\n                ),\n                llm=ChatOpenAI(\n                    openai_api_base=\"https://app.oscarbahamonde.com/v1\",\n                    # api_key=\"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\",\n                    model=\"mistralai/Mistral-7B-Instruct-v0.2\",\n                    # model=\"gpt-4-0125-preview\",\n                ),\n            )\n            return await executor.arun(self.inputs)\n"
        },
        {
          "path": "src/tools/_base.py",
          "name": "_base.py",
          "is_dir": false,
          "content": "import asyncio\nfrom functools import cached_property, wraps\nfrom os import environ\nfrom typing import ContextManager, Literal, Type, TypeVar\n\nfrom agent_proto.proxy import LazyProxy\nfrom httpx import AsyncClient\nfrom pydantic import BaseModel, Field\nfrom typing_extensions import ParamSpec\n\nT_co = TypeVar(\"T_co\", covariant=True)\n\n\nT = TypeVar(\"T\")\nP = ParamSpec(\"P\")\n\n\ndef singleton(cls: Type[T]) -> Type[T]:\n    \"\"\"\n    Decorator that transforms a class into a singleton.\n\n    Args:\n            cls (Type[T]): The class to be transformed into a singleton.\n\n    Returns:\n            Type[T]: The transformed singleton class.\n\n    \"\"\"\n    instances = {}\n\n    @wraps(cls)\n    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:\n        if cls not in instances:\n            instances[cls] = cls(*args, **kwargs)\n        return instances[cls]  # type: ignore\n\n    return wrapper  # type: ignore\n\n\nclass Inference(BaseModel, LazyProxy[AsyncClient]):\n    model: str = Field(...)\n    expects: Literal[\"image\", \"audio\", \"video\", \"text\", \"json\"] = Field(...)\n\n    @cached_property\n    def base_url(self) -> str:\n        return f\"https://api-inference.huggingface.co/models/{self.model}\"\n\n    @cached_property\n    def headers(self) -> dict[str, str]:\n        return {\n            \"Authorization\": f\"Bearer {environ.get('HF_TOKEN')}\",\n            \"Content-Type\": \"application/json\",\n            \"Accept\": self._parse_accept(),\n        }\n\n    def _parse_accept(self) -> str:\n        if self.expects == \"image\":\n            return \"image/png\"\n        if self.expects == \"audio\":\n            return \"audio/wav\"\n        if self.expects == \"video\":\n            return \"video/mp4\"\n        if self.expects == \"text\":\n            return \"text/plain\"\n        if self.expects == \"json\":\n            return \"application/json\"\n        raise ValueError(f\"Invalid expects value: {self.expects}\")\n\n    def __load__(self) -> AsyncClient:\n        return AsyncClient(\n            base_url=self.base_url,\n            headers=self.headers,\n        )\n\n    async def __call__(self, inputs: str | list[str]):\n        client = self.__load__()\n        return await client.post(\"\", json={\"inputs\": inputs})\n\n\n@singleton\nclass LoopContextManager(ContextManager[asyncio.AbstractEventLoop]):\n    def _ensure_loop(self) -> asyncio.AbstractEventLoop:\n        if asyncio.get_running_loop() and asyncio.get_running_loop().is_running():\n            return asyncio.get_running_loop()\n        if asyncio.get_event_loop() and asyncio.get_event_loop().is_running():\n            return asyncio.get_event_loop()\n        loop = asyncio.new_event_loop()\n        asyncio.set_event_loop(loop)\n        loop.run_forever()\n        return loop\n\n    def __enter__(self) -> asyncio.AbstractEventLoop:\n        return self._ensure_loop()\n\n    def __exit__(self, *_) -> None:\n        pass\n"
        },
        {
          "path": "src/tools/__init__.py",
          "name": "__init__.py",
          "is_dir": false,
          "content": "from .browser import BrowserTool\nfrom .images import GenerateImage\nfrom .monitor import ResourceManager\nfrom .openapi import Plugin  # pylint: disable=E0401\nfrom .vision import Vision\n\n__all__ = [\"GenerateImage\", \"Vision\", \"ResourceManager\", \"BrowserTool\", \"Plugin\"]\n"
        },
        {
          "path": "src/tools/browser.py",
          "name": "browser.py",
          "is_dir": false,
          "content": "import asyncio\nimport re\nfrom functools import cached_property\nfrom typing import AsyncIterator\n\nfrom bs4 import BeautifulSoup, Tag\nfrom pydantic import Field\nfrom pyppeteer import browser, launch\n\nfrom ..agent import Tool\n\nchrome: browser.Browser = None  # type: ignore\n\n\nclass BrowserTool(Tool):\n    inputs: str = Field(..., description=\"The query to search for.\")\n\n    async def run(self) -> list[str]:\n        try:\n            global chrome  # pylint: disable=global-statement\n            chrome = await launch(\n                headless=True,\n                args=[\"--no-sandbox\"],\n            )\n            page = await chrome.newPage()  # type: ignore\n            await page.setUserAgent(\n                \"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3\"\n            )\n            await page.goto(\"https://www.google.com\")  # type: ignore\n            await page.type(\"input[name=q]\", self.inputs)  # type: ignore\n            await page.keyboard.press(\"Enter\")  # type: ignore\n            await page.waitForNavigation()  # type: ignore\n            content = await page.content()  # type: ignore\n            soup = BeautifulSoup(content, \"lxml\")\n            links = soup.find_all(\"a\")\n            non_empty_links = [link.get(\"href\") for link in links if link.get(\"href\")]\n            return [\n                re.search(r\"(?P<url>https?://[^\\s]+)\", link)[\"url\"]  # type: ignore\n                for link in non_empty_links\n                if re.search(r\"(?P<url>https?://[^\\s]+)\", link)\n            ]\n\n        except (RuntimeError, KeyError):\n            return []  # type: ignore\n        finally:\n            await chrome.close()\n"
        },
        {
          "path": "src/tools/images.py",
          "name": "images.py",
          "is_dir": false,
          "content": "import base64\nfrom functools import cached_property\nfrom uuid import uuid4\n\nfrom ..agent import Tool\nfrom ._base import Inference\n\n\nclass GenerateImage(Tool):\n    \"\"\"Generates an image from a given text.\"\"\"\n\n    inputs: str\n\n    @cached_property\n    def client(self) -> Inference:\n        return Inference(\n            model=\"stabilityai/stable-diffusion-xl-base-1.0\",\n            expects=\"image\",\n        )\n\n    async def run(self):\n        response = await self.client(self.inputs)\n        image = response.content\n        with open(f\"images/{self.inputs[16]}-{str(uuid4())}.png\", \"wb\") as file:\n            file.write(image)\n        return f\"data:image/png;base64,{base64.b64encode(image).decode()}\"\n"
        }
      ]
    },
    {
      "path": "src/__init__.py",
      "name": "__init__.py",
      "is_dir": false,
      "content": "from fastapi import FastAPI as Application\n\nfrom .agent import Agent\nfrom .routes import ai_controller, auth_controller\n\n\ndef create_app():\n    app = Application(\n        title=\"NotOpenAI\",\n        version=\"0.0.1\",\n        servers=[\n            {\"url\": \"https://api.oscarbahamonde.com\", \"description\": \"Local Server\"}\n        ],\n    )\n    app.include_router(auth_controller, prefix=\"/api\")\n    app.include_router(ai_controller, prefix=\"/api\")\n    return app\n\n\n__all__ = [\"create_app\", \"Agent\"]\n"
    },
    {
      "path": "src/services",
      "name": "services",
      "is_dir": true,
      "content": [
        {
          "path": "src/services/pgVector.py",
          "name": "pgVector.py",
          "is_dir": false,
          "content": "from functools import cached_property\nfrom os import environ\nfrom typing import Literal\n\nfrom agent_proto import Tool\nfrom agent_proto.utils import setup_logging\nfrom langchain.embeddings.huggingface import HuggingFaceEmbeddings  # type: ignore\nfrom langchain.vectorstores.pgvector import PGVector\nfrom pydantic import Field\n\n\nclass PGRetrievalTool(Tool):\n    \"\"\"\n    Retrieval Augmented Generation (RAG) tool for storing and querying sentence embeddings.\n    \"\"\"\n\n    namespace: str = Field(..., description=\"Namespace on the knowledge store\")\n    inputs: str = Field(..., description=\"Input text to store or query\")\n    top_k: int = Field(5, description=\"Number of documents to retrieve\")\n    action: Literal[\"query\", \"upsert\"] = Field(\"query\")\n\n    @cached_property\n    def embeddings(self):\n        return HuggingFaceEmbeddings()\n\n    @cached_property\n    def store(self):\n        return PGVector(\n            connection_string=environ[\"DATABASE_URL\"],\n            embedding_function=self.embeddings,\n            embedding_length=1536,\n            collection_name=self.namespace,\n            collection_metadata=self.model_dump(),\n            logger=setup_logging(self.__class__.__name__),\n        )\n\n    @cached_property\n    def retriever(self):\n        return self.store.as_retriever()\n\n    async def query(self):\n        return self.retriever.get_relevant_documents(query=self.inputs)\n\n    async def upsert(self):\n        response = await self.retriever.vectorstore.aadd_texts(  # type: ignore\n            texts=self.inputs, metadatas=[self.model_dump()]\n        )\n        return len(response)\n\n    async def run(self):\n        return await self.query() if self.action == \"query\" else await self.upsert()\n"
        },
        {
          "path": "src/services/__init__.py",
          "name": "__init__.py",
          "is_dir": false,
          "content": "from .pgVector import PGRetrievalTool\n\n__all__ = [\"PGRetrievalTool\"]\n"
        }
      ]
    },
    {
      "path": "src/routes",
      "name": "routes",
      "is_dir": true,
      "content": [
        {
          "path": "src/routes/auth.py",
          "name": "auth.py",
          "is_dir": false,
          "content": "from os import environ\nfrom typing import Optional\n\nfrom fastapi import APIRouter, Depends, HTTPException, Request\nfrom httpx import AsyncClient\nfrom prisma.models import IUser\nfrom pydantic import BaseModel, Field\n\n\nclass User(BaseModel):\n    \"\"\"\n    Auth0 User\n    \"\"\"\n\n    email: Optional[str] = Field(default=None)\n    email_verified: Optional[bool] = Field(default=False)\n    family_name: Optional[str] = Field(default=None)\n    given_name: Optional[str] = Field(default=None)\n    locale: Optional[str] = Field(default=None)\n    name: str = Field(...)\n    nickname: Optional[str] = Field(default=None)\n    picture: Optional[str] = Field(default=None)\n    sub: str = Field(...)\n    updated_at: Optional[str] = Field(default=None)\n\n\ndef get_token(request: Request) -> str:\n    try:\n        return request.headers[\"Authorization\"].split(\"Bearer \")[1]\n    except (KeyError, IndexError) as e:\n        raise HTTPException(status_code=401, detail=\"Invalid token\") from e\n\n\nAUTH0_URL = environ[\"AUTH0_URL\"]\n\napp = APIRouter()\n\n\n@app.post(\"/auth\")\nasync def auth_endpoint(token: str = Depends(get_token)):\n    async with AsyncClient(\n        base_url=AUTH0_URL, headers={\"Authorization\": f\"Bearer {token}\"}\n    ) as client:\n        response = await client.get(\"/userinfo\")\n        user = User(**response.json())\n        return await IUser.prisma().upsert(\n            where={\"sub\": user.sub},\n            data={\n                \"create\": user.model_dump(),  # type: ignore\n                \"update\": user.model_dump(),  # type: ignore\n            },\n        )\n"
        },
        {
          "path": "src/routes/ai.py",
          "name": "ai.py",
          "is_dir": false,
          "content": "from typing import AsyncIterable\n\nfrom fastapi import APIRouter, File, UploadFile\nfrom sse_starlette.sse import EventSourceResponse\n\nfrom src import Agent\nfrom src.tools import BrowserTool, GenerateImage, Plugin, ResourceManager, Vision\n\napp = APIRouter()\nagent = Agent()\n\n\n@app.post(\"/image\")\nasync def generate_image(tool: GenerateImage):\n    return await tool()\n\n\n@app.post(\"/vision\")\nasync def generate_vision(inputs: UploadFile = File(...)):\n    return await Vision(inputs=await inputs.read())()\n\n\n@app.get(\"/chat\")\nasync def generate_chat_stream(inputs: str):\n    _generator = await agent.chat(message=inputs, stream=True)\n    assert isinstance(_generator, AsyncIterable)\n\n    async def _stream():\n        async for chunk in _generator:\n            if chunk:\n                yield chunk\n            else:\n                yield {\"event\": \"done\", \"data\": \"done\"}\n\n    return EventSourceResponse(_stream())\n\n\n@app.post(\"/chat\")\nasync def generate_chat(inputs: str):\n    return await agent.chat(\n        message=await agent.chat_template.render_async(\n            prompt=inputs, context=await agent.search(query=inputs)\n        ),\n        stream=False,\n    )\n\n\n@app.post(\"/completion\")\nasync def generate_completion(\n    inputs: str, instruction: str = \"Autocomplete the following.\"\n):\n    return await agent.instruct(\n        message=inputs,\n        instruction=instruction,\n    )\n\n\n@app.post(\"/resources\")\nasync def generate_resource():\n    return await ResourceManager()()\n\n\n@app.post(\"/plugin\")\nasync def generate_openapi(tool: Plugin):\n    return await tool()\n\n\n@app.post(\"/browser\")\nasync def generate_browser(tool: BrowserTool):\n    return await tool()\n"
        },
        {
          "path": "src/routes/__init__.py",
          "name": "__init__.py",
          "is_dir": false,
          "content": "from .ai import app as ai_controller\nfrom .auth import app as auth_controller\n\n__all__ = [\"auth_controller\", \"ai_controller\"]\n"
        }
      ]
    }
  ]
}


my docker compose

version: "3.7"
services:
  
  db:
    container_name: db
    build: 
      context: .
      dockerfile: db.Dockerfile
    restart: always
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_USER: postgres
      POSTGRES_DB: postgres
      DATABASE_URL: postgres://postgres:postgres@db:5432/postgres?authSource=admin
    ports:
      - "5432:5432"
    volumes:
      - ~/.data:/var/lib/postgresql/data
      #- ./db_hba.conf:/var/lib/postgresql/data/db_hba.conf

    networks:
      - main

  mistral:
    container_name: mistral
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    environment:
      - HF_TOKEN=${HF_TOKEN}
    command: --host 0.0.0.0 --model TheBloke/Mistral-7B-Instruct-v0.2-AWQ   --gpu-memory-utilization 0.80  --max-model-len 2048
    env_file:
      - .env
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: "all"
              capabilities: [gpu]
    networks:
      - main
    restart: always
    depends_on:
      - db

  queue:
    image: rabbitmq:3.7.7-management
    container_name: queue
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      - RABBITMQ_DEFAULT_USER=guest
      - RABBITMQ_DEFAULT_PASS=guest
    volumes:
      - ~/.docker/rabbitmq:/var/lib/rabbitmq
    networks:
      - default
    restart: always
    depends_on:
      - db

  redis:
    image: redis/redis-stack:latest
    container_name: redis
    ports:
      - "6379:6379"
      - "8001:8001"
    volumes:
      - ~/.docker/redis:/data
    networks:
      - default
    restart: on-failure
    depends_on:
      - db

  bucket:
    image: minio/minio:latest
    container_name: bucket
    ports:
      - "9000:9000"
      - "9001:9001"
    env_file:
      - .env
    volumes:
      - ./static:/data
    networks:
      - default 
    restart: always
    command: server /data --console-address ":9001"
    depends_on:
      - db
  app:
    container_name: app
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./:/app 
    depends_on:
      - db
      - mistral
    env_file:
      - .env
    networks:
      - main
   
  proxy:
    container_name: proxy
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - app
    networks:
      - main

networks:
  main:
    driver: bridge
