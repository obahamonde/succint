from functools import cached_property
from typing import Any, List, Literal, Type, TypeVar

from fastapi import APIRouter, Depends, FastAPI, File, UploadFile
from jinja2 import Template
from openai import AsyncOpenAI
from sse_starlette import EventSourceResponse

from ..database.embeddings import EmbeddingsController, Model, Sentence, Surreal, get_db
from ..utils.decorators import robust
from .audio import Whisper
from .functions import AIFunction, AIFunctionCall, AIFunctionStruct

T = TypeVar("T")
F = TypeVar("F", bound=AIFunction)


class Definition(Model):
    name: str
    parameters: dict[str, object]


class Message(Model):
    role: Literal["user", "assistant"]


class OpenCopilot(AsyncOpenAI):
    model: str = "mistralai/Mistral-7B-Instruct-v0.2"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, base_url="https://ai.oscarbahamonde.com/v1", **kwargs)

    @cached_property
    def template(self) -> Template:
        return Template(
            """
		[INST]
		These are the functions that are available for this step: {{definitions}}
		[/INST]
		The user input is:
		<s> {{prompt}}
		</s>
		[INST]
		Think step by step if user is asking you to perform an action. If so, 
		check if a json_object can be inferred according to Json Schema and directly send the valid json output without any additional content in the following format:
		<s> { "function": { "name": "function_name", "parameters": {"parameter_name": "parameter_value" } } }
		</s>
		[/INST]
		"""
        )

    @cached_property
    def vectordb(self):  # type: ignore
        return EmbeddingsController()

    @cached_property
    def whisper(self):  # type: ignore
        return Whisper()

    @robust
    async def complete(self, prompt: str):
        response = await self.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=1024,
            stream=False,
        )
        return response.choices[0].text

    @robust
    async def complete_stream(self, prompt: str):
        response = await self.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=1024,
            stream=True,
        )

        async def generator():
            async for completion in response:
                yield completion.choices[0].text

        return EventSourceResponse(generator(), media_type="text/event-stream")

    @robust
    async def chat_completion(self, prompt: str):
        response = await self.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            model=self.model,
        )
        text = response.choices[0].message.content
        while text is None:
            response = await self.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                model=self.model,
            )
            text = response.choices[0].message.content
            if isinstance(text, str):
                break
        return text

    @robust
    async def chat_completion_stream(self, prompt: str):
        response = await self.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            model=self.model,
        )

        async def generator():
            async for chat_completion in response:
                text = chat_completion.choices[0].delta.content
                if text:
                    yield text

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
            model="mistralai/Mistral-7B-Instruct-v0.2",
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
        raise ValueError("No function call found")

    @robust
    async def upsert(self, sentence: Sentence, db: Surreal):
        return await self.vectordb.upsert(data=sentence, db=db)

    @robust
    async def query(self, sentence: Sentence, db: Surreal):
        return await self.vectordb.query(data=sentence, db=db)


def create_ai_controller(application: FastAPI):
    app = APIRouter(prefix="/ai", tags=["ai"])
    ai = OpenCopilot()

    @app.post("/complete")
    async def _(prompt: str):
        return await ai.complete(prompt)

    @app.post("/complete/stream")
    async def _(prompt: str):
        return await ai.complete_stream(prompt)

    @app.post("/chat")
    async def _(prompt: str):
        return await ai.chat_completion(prompt)

    @app.post("/chat/stream")
    async def _(prompt: str):
        return await ai.chat_completion_stream(prompt)

    @app.post("/functions")
    async def _(prompt: str):
        return await ai(prompt=prompt, functions=AIFunction.__subclasses__())

    @app.post("/embeddings/upsert")
    async def _(
        sentence: Sentence,
        db: Surreal = Depends(get_db),
    ):
        return await ai.upsert(sentence=sentence, db=db)

    @app.post("/embeddings/query")
    async def _(
        sentence: Sentence,
        db: Surreal = Depends(get_db),
    ):
        return await ai.query(sentence=sentence, db=db)

    @app.post("/audio/transcribe")
    async def _(file: UploadFile = File(...)):
        async def generator():
            yield await file.read()

        return await ai.whisper.run(audio=generator())

    application.include_router(app, prefix="/api")
    return application
