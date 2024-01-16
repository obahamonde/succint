from __future__ import annotations

import asyncio
import functools
import logging
from time import perf_counter
from typing import Awaitable, Callable, Generator, Sequence, Type, TypeVar, Union

from fastapi import HTTPException
from rich.console import Console
from rich.logging import RichHandler
from rich.pretty import install
from rich.traceback import install as ins
from tenacity import retry as retry_
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential
from typing_extensions import ParamSpec

T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")


def setup_logging(name: str = __name__) -> logging.Logger:
    install()
    ins()
    console = Console(record=True, force_terminal=True)
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        tracebacks_extra_lines=2,
        tracebacks_theme="monokai",
        show_level=False,
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, handlers=[console_handler])
    logger_ = logging.getLogger(name)
    logger_.setLevel(logging.INFO)
    return logger_


logger = setup_logging()


def process_time(
    func: Callable[P, Union[Awaitable[T], T]]
) -> Callable[P, Awaitable[T]]:
    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = perf_counter()
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        end = perf_counter()
        logger.info(
            "Time taken to execute %s: %s seconds", wrapper.__name__, end - start
        )
        return result  # type: ignore

    return wrapper


def handle_errors(
    func: Callable[P, Union[Awaitable[T], T]]
) -> Callable[P, Awaitable[T]]:
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            wrapper.__name__ = func.__name__
            logger.info("Calling %s", wrapper.__name__)
            if asyncio.iscoroutinefunction(func):
                response = await func(*args, **kwargs)
                logger.info(response)
                return response  # type: ignore
            response = func(*args, **kwargs)
            logger.info(response)
            return response  # type: ignore
        except Exception as exc:
            raise HTTPException(status_code=500, detail=repr(exc))

    return wrapper


def retry(
    retries: int = 3,
    wait: int = 1,
    max_wait: int = 3,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        @retry_(
            stop=stop_after_attempt(retries),
            wait=wait_exponential(multiplier=wait, max=max_wait),
            retry=retry_if_exception_type(exceptions),
            reraise=True,
        )
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def robust(
    func: Callable[P, Awaitable[T]],
    *,
    max_retries: int = 3,
    wait: int = 1,
    max_wait: int = 3,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
) -> Callable[P, Awaitable[T]]:
    return functools.reduce(
        lambda f, g: g(f),  # type: ignore
        [retry(max_retries, wait, max_wait, exceptions), process_time, handle_errors],
        func,
    )


def chunker(seq: Sequence[T], size: int) -> Generator[Sequence[T], None, None]:
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def async_io(func: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    @functools.wraps(func)
    @robust
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


def clean_object(
    d: dict[str, object] | list[object]
) -> dict[str, object] | list[object]:
    if isinstance(d, dict):
        keys_to_delete = [k for k, v in d.items() if not v]
        for k in keys_to_delete:
            del d[k]
        for v in d.values():
            if isinstance(v, (dict, list)):
                clean_object(v)
    elif isinstance(d, list):
        d[:] = [v for v in d if v]
        for v in d:
            if isinstance(v, (dict, list)):
                clean_object(v)
    return d
