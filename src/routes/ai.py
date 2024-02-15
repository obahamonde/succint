from typing import AsyncIterable

from fastapi import APIRouter, File, UploadFile
from sse_starlette.sse import EventSourceResponse

from src.agent.mistral import MistralAI
from src.services import ObjectStorage
from src.tools import GenerateImage, Plugin, ResourceManager, Vision

app = APIRouter()
storage = ObjectStorage()


@app.post("/image/{namespace}")
async def generate_image(tool: GenerateImage):
    return await tool()


@app.post("/vision")
async def generate_vision(inputs: UploadFile = File(...)):
    return await Vision(inputs=await inputs.read())()


@app.get("/chat/{namespace}")
async def generate_chat_stream(inputs: str, namespace: str):
    agent = MistralAI(namespace=namespace)
    _generator = await agent.chat(message=inputs, sub=namespace, stream=True)
    assert isinstance(_generator, AsyncIterable)

    async def _stream():
        async for chunk in _generator:
            if chunk:
                yield chunk
            else:
                continue
        yield {"event": "done", "data": "done"}

    return EventSourceResponse(_stream())


@app.post("/chat/{namespace}")
async def generate_chat(message: str, namespace: str):
    agent = MistralAI(namespace=namespace)
    response = await agent.chat(message=message, sub=namespace, stream=False)
    assert not isinstance(response, AsyncIterable)
    return response

@app.post("/completion/{namespace}")
async def generate_completion(
    inputs: str, namespace: str, instruction: str = "Autocomplete the following."
):
    agent = MistralAI(namespace=namespace)
    return await agent.instruct(
        message=inputs,
        instruction=instruction,
    )


@app.post("/resources")
async def generate_resource():
    return await ResourceManager()()


@app.post("/plugin")
async def generate_openapi(tool: Plugin):
    return await tool()


@app.post("/load/{namespace}")
async def upload_file(file: UploadFile, namespace: str):
    agent = MistralAI(namespace=namespace)
    return await agent.upload(file=file)
