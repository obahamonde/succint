import asyncio
from functools import cached_property, wraps
from os import environ
from typing import ContextManager, Literal, Type, TypeVar

from agent_proto.proxy import LazyProxy
from httpx import AsyncClient
from pydantic import BaseModel, Field
from typing_extensions import ParamSpec

T_co = TypeVar("T_co", covariant=True)


T = TypeVar("T")
P = ParamSpec("P")


def singleton(cls: Type[T]) -> Type[T]:
    """
    Decorator that transforms a class into a singleton.

    Args:
            cls (Type[T]): The class to be transformed into a singleton.

    Returns:
            Type[T]: The transformed singleton class.

    """
    instances = {}

    @wraps(cls)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]  # type: ignore

    return wrapper  # type: ignore


class Inference(BaseModel, LazyProxy[AsyncClient]):
    model: str = Field(...)
    expects: Literal["image", "audio", "video", "text", "json"] = Field(...)

    @cached_property
    def base_url(self) -> str:
        return f"https://api-inference.huggingface.co/models/{self.model}"

    @cached_property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {environ.get('HF_TOKEN')}",
            "Content-Type": "application/json",
            "Accept": self._parse_accept(),
        }

    def _parse_accept(self) -> str:
        if self.expects == "image":
            return "image/png"
        if self.expects == "audio":
            return "audio/wav"
        if self.expects == "video":
            return "video/mp4"
        if self.expects == "text":
            return "text/plain"
        if self.expects == "json":
            return "application/json"
        raise ValueError(f"Invalid expects value: {self.expects}")

    def __load__(self) -> AsyncClient:
        return AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
        )

    async def __call__(self, inputs: str | list[str]):
        client = self.__load__()
        return await client.post("", json={"inputs": inputs})


@singleton
class LoopContextManager(ContextManager[asyncio.AbstractEventLoop]):
    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if asyncio.get_running_loop() and asyncio.get_running_loop().is_running():
            return asyncio.get_running_loop()
        if asyncio.get_event_loop() and asyncio.get_event_loop().is_running():
            return asyncio.get_event_loop()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_forever()
        return loop

    def __enter__(self) -> asyncio.AbstractEventLoop:
        return self._ensure_loop()

    def __exit__(self, *_) -> None:
        pass
