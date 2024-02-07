from functools import cached_property

from agent_proto import Tool


from ._base import Inference


class Vision(Tool):
    """Vision tool processes an image into a text description."""

    inputs: bytes

    @cached_property
    def client(self):
        return Inference(model="Salesforce/blip-image-captioning-large", expects="text")

    async def run(self):
        response = await self.client.__load__().post("", data=self.inputs)  # type: ignore
        return response.text
