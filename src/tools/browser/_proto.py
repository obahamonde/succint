from typing import AsyncGenerator, Awaitable, Generator, Protocol, TypeVar

T = TypeVar("T", contravariant=True)


class Crawler(Protocol[T]):
    def html(
        self, tool: T, *, url: str
    ) -> str | Awaitable[str] | Generator[str, None, None]:
        ...

    def screenshot(
        self, tool: T, *, url: str
    ) -> bytes | Awaitable[bytes] | Generator[bytes, None, None]:
        ...

    def json(
        self, tool: T, *, url: str
    ) -> str | Awaitable[str] | Generator[str, None, None]:
        ...

    def pdf(
        self, tool: T, *, url: str
    ) -> str | Awaitable[str] | Generator[str, None, None]:
        ...

    def children(
        self, tool: T, *, base_url: str, limit: int = 100
    ) -> Generator[str, None, None] | AsyncGenerator[str, None]:
        ...
