import base64
import io
import os
from functools import cached_property
from time import perf_counter
from uuid import uuid4

from agent_proto.client import APIClient
from agent_proto.tool import Tool
from fastapi import UploadFile
from openai import AsyncOpenAI

from ..schemas import URLResponse
from ..schemas.responses import TikTokenizer
from ..services import ObjectStorage

ai = AsyncOpenAI()
tokenizer = TikTokenizer()


async def oai_generate_image(*, inputs: str) -> dict[str, str | int]:
    response = await ai.images.generate(
        prompt=inputs,
        model="dall-e-3",
        quality="hd",
        response_format="b64_json",
        size="1024x1024",
    )
    response = response.data[0].b64_json  # type: ignore
    tokens = tokenizer.encode(inputs)
    url = base64.b64decode(f"data:image/png;base64,{response}").decode("utf-8")
    return {"url": url, "tokens": len(tokens)}


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
        start = perf_counter()
        key = f"images/{self.inputs[8]} -{str(uuid4())}.png"
        try:
            response = await self.client.__load__().request(
                "", "POST", json={"inputs": self.inputs + "\n\n" + "Size:1024x1024"}
            )
            image = io.BytesIO(response.content)
            url = await self.storage.put(
                key=key, file=UploadFile(file=image, filename=key)
            )
            return URLResponse(
                url=url,
                processing_time=perf_counter() - start,
                tokens=len(tokenizer.encode(self.inputs)),
            )
        except Exception:
            response = await oai_generate_image(inputs=self.inputs)
            b64_image = response["url"]
            tokens = response["tokens"]
            assert isinstance(b64_image, str)
            assert isinstance(tokens, int)
            url = await self.storage.put(
                key=key,
                file=UploadFile(file=io.BytesIO(b64_image.encode()), filename=key),
            )
            return URLResponse(
                url=url,
                processing_time=perf_counter() - start,
                tokens=tokens,
            )
