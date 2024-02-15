from abc import ABC, abstractmethod
from functools import cached_property
from typing import Generic, TypeVar

import tiktoken

T = TypeVar("T")


class AbstractTokenizer(ABC, Generic[T]):
    @abstractmethod
    def encode(self, text: str) -> T:
        pass


class TikTokenizer(AbstractTokenizer[list[int]]):
    @cached_property
    def encoding(self):
        return tiktoken.encoding_for_model("cl100k_base")

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text=text)
