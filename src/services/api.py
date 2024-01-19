import base64
import json
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, List, Literal, Type, TypeVar

from fastapi import APIRouter, Depends, FastAPI, File, UploadFile
from jinja2 import Template
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion
from sse_starlette import EventSourceResponse

from ..database.application import get_db
from ..utils.decorators import robust
from .audio import AudioTranscription
from ..tools.context.embeddings import DatabaseModel, Sentence, Surreal, VectorStore
from .functions import AIFunction, AIFunctionCall, AIFunctionStruct
from .images import Images

T = TypeVar("T")
F = TypeVar("F", bound=AIFunction)


class Definition(DatabaseModel):
    name: str
    parameters: dict[str, object]


class Message(DatabaseModel):
    role: Literal["user", "assistant"]


@dataclass
class AsyncMistralAI(AsyncOpenAI):
    model: str = field(default="mistralai/Mistral-7B-Instruct-v0.2")

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, base_url="http://localhost:8000/v1", **kwargs)

    @cached_property
    def template(self) -> Template:
        return Template(
            """
		These are the functions that are available for this step: {{definitions}}
		The user input is:
        {{prompt}}
		Think step by step if user is asking you to perform an action. If so, 
		check if a json_object can be inferred according to Json Schema and directly send the valid json output without any additional content in the following format:
		{ "function": { "name": "function_name", "parameters": {"parameter_name": "parameter_value" } } }
		"""
        )

    @cached_property
    def vector_db(self):  # type: ignore
        return VectorStore()

    @cached_property
    def whisper(self):  # type: ignore
        return AudioTranscription()

    @cached_property
    def diffusor(self):
        return Images()

    async def complete(self, prompt: str, stream: bool = False, max_tokens: int = 256):
        response = await self.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            stream=stream,
        )
        if isinstance(response, Completion):
            return response.choices[0].text

        async def generator():
            async for completion in response:
                yield json.dumps(
                    {"role": "assistant", "content": completion.choices[0]}
                )

        return EventSourceResponse(
            generator(),
            media_type="text/event-stream",
            headers={"Access-Control-Allow-Origin": "*"},
        )

    async def chat_completion(
        self, prompt: str, stream: bool = False, max_tokens: int = 256
    ):
        response = await self.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            stream=stream,
            model=self.model,
            max_tokens=max_tokens,
        )
        if isinstance(response, ChatCompletion):
            return response.choices[0].message

        async def generator():
            async for chat_completion in response:
                yield chat_completion.choices[0].delta.model_dump_json()

        return EventSourceResponse(generator(), media_type="text/event-stream")

    @robust
    async def __call__(
        self, *, prompt: str, functions: List[Type[F]]
    ) -> AIFunctionCall:
        defs = [f.definition() for f in functions]
        prompt = self.template.render(
            definitions=defs,
            prompt=prompt,
        )
        response = await self.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.2,
        )
        content = response.choices[0].message.content
        assert isinstance(content, str), "Invalid response from Mistral"
        data = AIFunctionStruct.parse_raw(content)  # type: ignore
        for f in functions:
            if f.__name__.lower() == data.function.name:
                return await f.model_validate(data.function.parameters)()
        content = response.choices[0].message.content
        assert isinstance(content, str), "Invalid response from Mistral"
        raise ValueError(f"Invalid function {content}")

    @robust
    async def upsert(self, sentence: Sentence, db: Surreal):
        return await self.vector_db.upsert(data=sentence, db=db)

    @robust
    async def query(self, sentence: Sentence, db: Surreal):
        return await self.vector_db.query(data=sentence, db=db)

    @robust
    async def generate_image(self, prompt: str):
        response = await self.diffusor(inputs=prompt)
        _b64_string = base64.b64encode(response).decode("utf-8")
        return f"data:image/png;base64,{_b64_string}"


def ai_controller(application: FastAPI):
    app = APIRouter(prefix="/ai", tags=["ai"])
    ai = AsyncMistralAI(api_key="foo")

    @app.get("/chat")
    async def _(prompt: str, stream: bool = False):
        return await ai.chat_completion(prompt=prompt, stream=stream)

    @app.get("/completion")
    async def _(prompt: str, stream: bool = False):
        return await ai.complete(prompt=prompt, stream=stream)

    @app.post("/functions")
    async def _(prompt: str):
        return await ai(prompt=prompt, functions=AIFunction.__subclasses__())

    @app.post("/embeddings")
    async def _(
        sentence: Sentence,
        operation: Literal["upsert", "query"] = "upsert",
        db: Surreal = Depends(get_db),
    ):
        if operation == "query":
            return await ai.query(sentence=sentence, db=db)
        return await ai.upsert(sentence=sentence, db=db)

    @app.post("/audio/transcribe")
    async def _(file: UploadFile = File(...)):
        async def generator():
            yield await file.read()

        return ai.whisper.run(audio=generator())

    @app.post("/images/generate")
    async def _(prompt: str):
        return await ai.generate_image(prompt=prompt)

    application.include_router(app, prefix="/api")
    return application
