from typing import AsyncIterable

from fastapi import APIRouter, File, UploadFile
from sse_starlette.sse import EventSourceResponse

from src import Agent
from src.tools import GenerateImage, GoogleSearchTool, Plugin, ResourceManager, Vision

app = APIRouter()
agent = Agent()


@app.post("/image")
async def generate_image(tool: GenerateImage):
    return await tool()


@app.post("/vision")
async def generate_vision(inputs: UploadFile = File(...)):
    return await Vision(inputs=await inputs.read())()


@app.get("/chat")
async def generate_chat_stream(inputs: str):
    _generator = await agent.chat(message=inputs, stream=True)
    assert isinstance(_generator, AsyncIterable)

    async def _stream():
        async for chunk in _generator:
            if chunk:
                yield chunk
            else:
                yield {"event": "done", "data": "done"}

    return EventSourceResponse(_stream())


@app.post("/chat")
async def generate_chat(inputs: str):
    return await agent.chat(
        message=await agent.chat_template.render_async(
            prompt=inputs, context=await agent.search(query=inputs)
        ),
        stream=False,
    )


@app.post("/completion")
async def generate_completion(
    inputs: str, instruction: str = "Autocomplete the following."
):
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
