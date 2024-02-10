import base64
import io
import os
from functools import cached_property
from uuid import uuid4

from agent_proto.client import APIClient
from agent_proto.tool import Tool
from fastapi import UploadFile
from openai import AsyncOpenAI

from ..services import ObjectStorage

ai = AsyncOpenAI()


async def oai_generate_image(*, inputs: str) -> bytes:
    response = await ai.images.generate(
        prompt=inputs,
        model="dall-e-3",
        quality="hd",
        response_format="b64_json",
        size="1024x1024",
    )
    response = response.data[0].b64_json  # type: ignore
    return base64.b64decode(f"data:image/png;base64,{response}")


class GenerateImage(Tool):
    """Generates an image from a given text."""

    inputs: str

    @cached_property
    def client(self) -> APIClient:
        return APIClient(
            base_url="https://api.runpod.ai/v2/l6jgxv9b799f5z/runsync",
            headers={
                "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
            },
        )

    @cached_property
    def storage(self) -> ObjectStorage:
        return ObjectStorage()

    async def run(self):
        key = f"images/{self.inputs[8]}-{str(uuid4())}.png"
        try:
            response = await self.client.__load__().request(
                "", "POST", json={"inputs": self.inputs + "\n\n" + "Size:1024x1024"}
            )
            image = response.content
            url = await self.storage.put(
                key=key, file=UploadFile(file=image, filename=key)
            )
            return {"url": url}
        except (AttributeError, KeyError):
            response = await oai_generate_image(inputs=self.inputs)
            url = await self.storage.put(
                key=key, file=UploadFile(file=io.BytesIO(response), filename=key)
            )
            return {"url": url}
