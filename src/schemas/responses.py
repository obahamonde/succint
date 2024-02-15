from functools import cached_property

from pydantic import BaseModel, Field

from ._utils import TikTokenizer


class URLResponse(BaseModel):
    url: str = Field(..., description="The URL of the generated content.")
    processing_time: float = Field(
        ..., description="The time taken to generate the content."
    )
    tokens: int = Field(
        ..., description="The number of tokens used to generate the content."
    )

    @cached_property
    def tokenizer(self):
        return TikTokenizer()
