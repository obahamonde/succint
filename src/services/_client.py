from typing import Any, AsyncIterable, Dict, List, Literal, Optional, TypeVar, Union

from httpx import AsyncClient
from pydantic import BaseModel

from ..utils.decorators import robust
from ._proxy import LazyProxy

T = TypeVar("T")
Method = Literal["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS", "TRACE"]
Json = Union[Dict[str, Any], List[Any], str, int, float, bool, None]
Scalar = Union[str, int, float, bool, None]


class APIClient(BaseModel, LazyProxy[AsyncClient]):
    base_url: str
    headers: Dict[str, str]

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.__load__(*args, **kwargs)

    def __load__(self, *args: Any, **kwargs: Any):
        return AsyncClient(
            base_url=self.base_url, headers=self.headers, timeout=30, *args, **kwargs
        )

    def dict(self, *args: Any, **kwargs: Any):
        return super().model_dump(*args, **kwargs, exclude={"headers"})

    def _update_headers(self, additional_headers: Optional[Dict[str, str]] = None):
        if additional_headers:
            self.headers.update(additional_headers)
        return self.headers

    @robust
    async def fetch(
        self,
        url: str,
        *,
        method: Method,
        params: Optional[Dict[str, Scalar]] = None,
        json: Optional[Json] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        headers = self._update_headers(headers)
        return await self.__load__().request(
            method=method, url=url, headers=headers, json=json, params=params
        )

    @robust
    async def get(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Scalar]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        response = await self.fetch(
            method="GET", url=url, headers=headers, params=params
        )
        return response.json()

    @robust
    async def post(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Scalar]] = None,
        json: Optional[Json] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        response = await self.fetch(
            method="POST", url=url, json=json, headers=headers, params=params
        )
        return response.json()

    @robust
    async def put(
        self,
        url: str,
        *,
        json: Optional[Json] = None,
        params: Optional[Dict[str, Scalar]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        response = await self.fetch(
            method="PUT", url=url, json=json, headers=headers, params=params
        )
        return response.json()

    @robust
    async def delete(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Scalar]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        response = await self.fetch(
            method="DELETE", url=url, headers=headers, params=params
        )
        return response.json()

    @robust
    async def patch(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Scalar]] = None,
        json: Optional[Json] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        response = await self.fetch(
            method="PATCH", url=url, json=json, headers=headers, params=params
        )
        return response.json()

    @robust
    async def head(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Scalar]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        response = await self.fetch(
            method="HEAD", url=url, headers=headers, params=params
        )
        return response.json()

    @robust
    async def options(
        self,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
    ):
        response = await self.fetch(method="OPTIONS", url=url, headers=headers)
        return response.json()

    @robust
    async def trace(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Scalar]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        response = await self.fetch(
            method="TRACE", url=url, headers=headers, params=params
        )
        return response.json()

    @robust
    async def text(
        self,
        url: str,
        *,
        method: Method = "GET",
        params: Optional[Dict[str, Scalar]] = None,
        json: Optional[Json] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        response = await self.fetch(
            method=method, url=url, json=json, headers=headers, params=params
        )
        return response.text

    @robust
    async def blob(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Scalar]] = None,
        method: Method = "GET",
        json: Optional[Json] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        response = await self.fetch(
            method=method, url=url, json=json, params=params, headers=headers
        )
        return response.content

    async def stream(
        self,
        url: str,
        *,
        method: Method,
        params: Optional[Dict[str, Scalar]] = None,
        json: Optional[Json] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> AsyncIterable[bytes]:
        headers = self._update_headers(headers)
        response = await self.fetch(
            url, method=method, json=json, params=params, headers=headers
        )
        async for chunk in response.aiter_bytes():
            yield chunk
