import base64
from functools import cached_property
from uuid import uuid4

from ..agent import Tool
from ._base import Inference


class GenerateImage(Tool):
    """Generates an image from a given text."""

    inputs: str

    @cached_property
    def client(self) -> Inference:
        return Inference(
            model="stabilityai/stable-diffusion-xl-base-1.0",
            expects="image",
        )

    async def run(self):
        response = await self.client(self.inputs)
        image = response.content
        with open(f"images/{self.inputs[16]}-{str(uuid4())}.png", "wb") as file:
            file.write(image)
        return f"data:image/png;base64,{base64.b64encode(image).decode()}"
