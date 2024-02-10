"""
Agent module.
A Large Language Model Agent (OLLAMA) is an agent based on mistral-7B-instruct-2.0 quantized to 4 bits that can be run locally and interacted with via a chat interface.
The agent can be used to run tools based on user input.
The agent is trained to provide useful responses and guidance to the user.
"""

import json as json_module
from functools import cached_property
from typing import AsyncIterator, Hashable, List, Type

from agent_proto import BaseAgent, robust
from agent_proto.agent import Message
from agent_proto.tool import Tool, ToolDefinition, ToolOutput
from agent_proto.utils import setup_logging
from fastapi import UploadFile
from openai import AsyncOpenAI
from openai._streaming import AsyncStream
from openai.types.chat import ChatCompletionChunk, ChatCompletionToolParam
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel, Field
from typing_extensions import Iterable

from ..services import PineconeClient
from ..services.minioStorage import ObjectStorage
from ..tools import GoogleSearchTool

logger = setup_logging(__name__)


class AgentSchema(BaseModel):
    model: str
    tools: List[str]
    messages: List[Message]


class ChatGPT(BaseAgent[AsyncOpenAI], Hashable):
    """An agent that interacts with users and performs tool calls based on user input.

    Attributes:
        messages (list[Message]): List of messages exchanged between the user and the agent.
        model (str): The model used by the agent.
        tools (list[Type[Tool[Any]]]): List of available tool classes.

    Methods:
        chat: Send a message to the agent and return the response.
        run: Run a tool based on a user message.
        __call__: Run a specific tool class based on a user message.
    """

    model: str = Field(default="gpt-3.5-turbo-0125")
    tools: Iterable[Type[Tool]] = Field(default_factory=Tool.__subclasses__)
    messages: List[Message] = []
    namespace: str = Field(default="default")

    def __call__(self):
        """Load the AsyncClient."""
        return AsyncOpenAI()

    @cached_property
    def retriever(self):
        """The retriever tool used by the agent."""
        return PineconeClient(namespace=self.namespace)

    @robust
    async def chat(self, *, message: str, stream: bool = True):  # type: ignore
        """
        Chat with the agent.

        Args:
            message (str): The message sent to the agent.
            stream (bool): Whether the response should be streamed or not.

        Returns:
            The response from the agent.
            (AsyncIterator[Message]): If stream is True.
            (Message): If stream is False.
        Raises:
            ValueError: If the response doesn't contain any content.
        """
        query_vector = await self.embed(message=message)
        matches = (await self.retriever.query(vector=query_vector)).matches
        context = "\n".join(
            f"Result: {match.metadata.get('text')}\nScore: {match.score}"
            for match in matches
        )
        response = await self().chat.completions.create(  # type: ignore
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"Use the following context to provide a better response: {context}",
                },
                {"role": "user", "content": message},
            ],
            stream=stream,
            max_tokens=4096,
            tools=[
                ChatCompletionToolParam(
                    type="function",
                    function=FunctionDefinition(
                        name=t.__name__,
                        description=(
                            t.__doc__ if t.__doc__ else "[No description available]"
                        ),
                        parameters=t.definition(),  # type: ignore
                    ),
                )
                for t in self.tools  # pylint: disable=E1133
            ],
        )

        if stream:
            assert isinstance(response, AsyncStream)

            response = await self().chat.completions.create(  # type: ignore
                model=self.model,
                messages=[{"role": "user", "content": message}],
                stream=stream,
                max_tokens=3072,
            )

        if stream:
            assert isinstance(response, AsyncStream)

            async def _():
                nonlocal response
                async for choice in response:  # type: ignore
                    assert isinstance(choice, ChatCompletionChunk)
                    content = choice.choices[0].delta.content
                    if content:
                        yield content

                    else:
                        continue
                yield {"event": "done", "data": "done"}

            return _()  # type: ignore  # type: ignore

        assert isinstance(response, ChatCompletion)
        content = response.choices[0].message.content
        if content:
            return ToolOutput(content=content, role="assistant")

    @robust
    async def run(  # type: ignore
        self, *, message: str, definitions: Iterable[ToolDefinition] | None = None
    ) -> ToolOutput:
        if definitions is None:
            defs = [
                ChatCompletionToolParam(
                    type="function",
                    function=FunctionDefinition(
                        name=t.__name__,
                        description=(
                            t.__doc__ if t.__doc__ else "[No description available]"
                        ),
                        parameters=t.definition()["properties"],
                    ),
                )
                for t in self.tools  # pylint: disable=E1133
            ]
        else:
            defs = [
                ChatCompletionToolParam(
                    type="function",
                    function=FunctionDefinition(
                        name=definition["title"],
                        description=definition["description"],
                        parameters=definition["properties"],
                    ),
                )
                for definition in definitions
            ]

        response = await self().chat.completions.create(
            messages=[
                {"role": "user", "content": message},
                {
                    "role": "system",
                    "content": "You are a function orchestrator. Based on user input determine which tool is gonna be used.",
                },
            ],
            model=self.model,
            max_tokens=4096,
            tools=defs,
        )
        calls = response.choices[0].message.tool_calls
        if not calls:
            content = response.choices[0].message.content
            if content:
                return ToolOutput(content=content, role="assistant")
            return ToolOutput(content="No tool call was inferred.", role="assistant")
        for call in calls:
            tool = next(
                t
                for t in self.tools
                if t.__name__.lower() == call.function.name.lower()
            )
            parameters = call.function.arguments
            return await tool(**json_module.loads(parameters))()
        return ToolOutput(content="No tool call was inferred.", role="assistant")

    @robust
    async def instruct(self, *, message: str, instruction: str) -> str:
        """Instruct the agent to perform a tool call based on user input.

        Args:
            message (str): The message sent to the agent.

        Returns:
            str: The response from the agent.
        """
        response = await self().completions.create(
            prompt=f"""
            Instruction: {instruction}
            Message: {message}
            """,
            model=self.model,
        )
        return response.choices[0].text

    @robust
    async def search(self, query: str):
        """Search the web using the agent."""
        return await GoogleSearchTool(inputs=query)()

    async def embed(self, *, message: str):
        """Embed the message in a vector space."""
        response = await self().embeddings.create(
            input=message, model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def __hash__(self):
        return hash(self.model + self.namespace)

    @cached_property
    def storage(self):
        """The object storage used by the agent."""
        return ObjectStorage()

    @robust
    async def upload(self, *, file: UploadFile):
        """Upload a file to the object storage."""
        return await self.storage.put(
            key=f"{self.namespace}/{file.filename}", file=file
        )
