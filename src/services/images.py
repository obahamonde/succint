from os import environ

from pydantic import BaseModel, Field

from src.services._client import APIClient


class Request(BaseModel):
    inputs: str | list[str] = Field(..., description="Text to summarize")


class Images(APIClient):
    base_url: str = Field(
        default="https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    )
    headers: dict[str, str] = Field(
        default={
            "Authorization": f"Bearer {environ.get('HF_TOKEN')}",
            "Content-Type": "application/json",
            "Accept": "image/png",
        }
    )

    async def __call__(self, *, inputs: str | list[str]) -> bytes:
        return await self.blob(
            "/", method="POST", json=Request(inputs=inputs).model_dump()
        )
