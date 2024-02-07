from functools import wraps
from typing import Awaitable, Callable, TypeVar

import aioredis
from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")

db = aioredis.Redis.from_url("redis://redis:6379/1")  # type: ignore


def cache(
    func: Callable[P, Awaitable[T]], *, ttl: int = 3600
) -> Callable[P, Awaitable[T]]:
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        key = f"{func.__module__}.{func.__name__}:{args}:{kwargs}"
        if await db.exists(key):  # type: ignore
            return await db.get(key)  # type: ignore
        result = await func(*args, **kwargs)
        await db.set(key, result, ex=ttl)  # type: ignore
        return result

    return wrapper
