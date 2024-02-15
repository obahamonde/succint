"""
Agent module.
A Large Language Model Agent (OLLAMA) is an agent based on mistral-7B-instruct-2.0 quantized to 4 bits that can be run locally and interacted with via a chat interface.
The agent can be used to run tools based on user input.
The agent is trained to provide useful responses and guidance to the user.
"""

import asyncio
import json as json_module
from functools import cached_property
from typing import AsyncIterator, List, Optional, Type

import tiktoken
from agent_proto import BaseAgent, robust
from agent_proto.agent import Message
from agent_proto.tool import Tool, ToolDefinition, ToolOutput
from agent_proto.utils import setup_logging
from fastapi import File, UploadFile
from jinja2 import Template
from openai import AsyncOpenAI
from openai._streaming import AsyncStream
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from prisma.models import IMessage
from pydantic import BaseModel, Field
from typing_extensions import Iterable

from ._prompts import CHAT_TEMPLATE, RUN_TEMPLATE

logger = setup_logging(__name__)


class AgentSchema(BaseModel):
    model: str
    tools: List[str]
    messages: List[Message]


class MistralAI(BaseAgent[AsyncOpenAI]):
    """An agent that interacts with users and performs tool calls based on user input.

    Attributes:
        messages (list[Message]): List of messages exchanged between the user and the agent.
        model (str): The model used by the agent.
        chat: Send a message to the agent and return the response.
        run: Run a tool based on a user message.
        __call__: Run a specific tool class based on a user message.
    """
    namespace: str = Field(default="default")
    model: str = Field(default="TheBloke/Mistral-7B-Instruct-v0.2-AWQ")
    # model: str = Field(default="clibrain/lince-mistral-7b-it-es")
    tools: Iterable[Type[Tool]] = Field(default_factory=Tool.__subclasses__)
    messages: List[Message] = []
    max_sequence_length: int = Field(default=4096)
    @cached_property
    def run_template(self):
        """The template used for generating the message sent to the agent. Crafted using prompt engineering to guide the model to infer the schema for performing tool calls based on user's message."""
        return Template(RUN_TEMPLATE, enable_async=True)

    @cached_property
    def tokenizer(self):
        """The tokenizer used by the agent."""
        return tiktoken.encoding_for_model(model_name="gpt-3.5")

    @cached_property
    def chat_template(self):
        """The template used for enabling the agent with context and instructions for responding to user's messages."""
        return Template(CHAT_TEMPLATE, enable_async=True)

    def __call__(self):
        """Load the AsyncClient."""
        return AsyncOpenAI(
            api_key=".",
            base_url="http://mistral:8000/v1",
        )

    @robust
    async def chat(  # type: ignore
        self, *, message: str, stream: bool = True
    ) -> AsyncIterator[str] | Message:
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
        # messages = await IMessage.prisma().find_many(
        #     take=20, where={"user": {"sub": sub}}  # type: ignore
        # )
        # tokens = self.tokenizer.encode(
        #     " ".join([message.content for message in messages])
        # )
        # if len(tokens) > self.max_sequence_length:
        #     messages = messages[-(self.max_sequence_length // 2) :]
        response = await self().chat.completions.create(  # type: ignore
            model=self.model,
            # messages=[
            #     {
            #         "role": "user",
            #         "content": message,
            #     },
            #     *[
            #         {
            #             "role": message.role,
            #             "content": message.content,
            #         }
            #         for message in messages
            #     ],  # type: ignore
            # ],
            messages=[{"role": "user", "content": message}],
            stream=stream,
            max_tokens=2048,
        )

        if stream:
            assert isinstance(response, AsyncStream)

            async def _():
                nonlocal response
                string = ""
                async for choice in response:  # type: ignore
                    assert isinstance(choice, ChatCompletionChunk)
                    content = choice.choices[0].delta.content
                    if content:
                        string += content
                        yield Message(
                            role="assistant", content=content
                        ).model_dump_json()

                    else:
                        continue
                yield {"event": "done", "data": "done"}
                # await IMessage.prisma().create(
                #     data={"user": {"connect": {"sub": sub}}, "content": message}
                # )
                # await IMessage.prisma().create(  # type: ignore
                #     data={"user": {"connect": {"sub": sub}}, "content": string}
                # )
            return _()  # type: ignore

        assert isinstance(response, ChatCompletion)
        content = response.choices[0].message.content
        if content:
            message_ = Message(role="assistant", content=content)
            return message_
        raise ValueError("No content in response")

    @robust
    async def run(  # type: ignore
        self, *, message: str, definitions: Optional[List[ToolDefinition]] = None
    ) -> ToolOutput:
        """Executes a tool based on natural language input.
        It works as follows:
        1. The user provides a message.
        2. The agent picks a tool from the list of definitions available, otherwise it returns a chat `Message` object.
        3. The agent sends an inferred json object based on the tool definition and user input to the Tool class, which call it's constructor with the parsed json object.
        4. The Tool executes the logic implemented on it's run method and returns the output as a ToolOutput object with the following structure:

        ```json
        {
            "role": "tool_name",
            "content": "tool_output"
        }
        ```

        5. The agent returns the ToolOutput object to the user.

        Args:
            message (str): The message sent to the agent.

        Returns:
            ToolOutput: The output of the tool.
        """
        if definitions is None:
            definitions = [klass.definition() for klass in self.tools]
        prompt_ = await self.run_template.render_async(
            message=message,
            definitions=definitions,
        )
        response = await self.chat(
            message=prompt_,
            stream=False,
        )
        assert isinstance(response, Message)
        data = json_module.loads(response.content)
        try:
            for deff in definitions:
                if deff["title"].lower() == data["role"]:
                    tool = next(
                        klass for klass in self.tools if klass.__name__ == deff["title"]
                    )
                    assert issubclass(tool, Tool)
                    return await tool(**data)()
            output = await self.chat(message=message, stream=False)
            assert isinstance(output, Message)
            return ToolOutput(role="assistant", content=output.content)
        except (StopIteration, KeyError):
            output = await self.chat(message=message, stream=False)
            assert isinstance(output, Message)
            return ToolOutput(role="assistant", content=output.content)

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
            <s>
            [INST]
            {instruction}
            [/INST]
            {message}</s>
            """,
            model=self.model,
        )
        return response.choices[0].text

    async def search(self, query: str):
        """Search the web using the agent."""
        results = await GoogleSearchTool(inputs=query)()
        summarized = await asyncio.gather(
            *[
                self.instruct(
                    message=el,
                    instruction=f"Summarize this content, the output must be 100% exclusively in the same language as the input: {query}",
                )
                for el in results.content
            ]
        )
        context = "\n".join(summarized)
        logger.info(f"Results from the web: {context}")
        return f"Results from the web: {context}"

    @cached_property
    def storage(self):
        """The object storage used by the agent."""
        return ObjectStorage()

    @robust
    async def upload(self, *, file: UploadFile = File(...)):
        """Upload a file to the object storage."""
        return await self.storage.put(
            key=f"{self.namespace}/{file.filename}", file=file
        )
