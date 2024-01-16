import json
from functools import singledispatch
from typing import Any

from randomname import get_name  # type: ignore


def generate_name() -> str:
    s = get_name()
    return "".join(x.title() for x in s.split("-"))


@singledispatch
def json_loads(data: Any) -> Any:
    return data


@json_loads.register(str)
def _(data: str) -> Any:
    return json.loads(data)


@json_loads.register(bytes)
def _(data: bytes) -> Any:
    return json.loads(data.decode("utf-8"))


@json_loads.register(bytearray)
def _(data: bytearray) -> Any:
    return json.loads(data.decode("utf-8"))


@json_loads.register(memoryview)
def _(data: memoryview) -> Any:
    return json.loads(data.tobytes().decode("utf-8"))


@json_loads.register(list)
def _(data: list) -> Any:
    return [json_loads(x) for x in data]


@json_loads.register(tuple)
def _(data: tuple) -> Any:
    return tuple(json_loads(x) for x in data)


@json_loads.register(dict)
def _(data: dict) -> Any:
    return {k: json_loads(v) for k, v in data.items()}


@json_loads.register(set)
def _(data: set) -> Any:
    return {json_loads(x) for x in data}


@json_loads.register(frozenset)
def _(data: frozenset) -> Any:
    return frozenset(json_loads(x) for x in data)


@json_loads.register(type(None))
def _(data: type(None)) -> Any:
    return None


@json_loads.register(bool)
def _(data: bool) -> Any:
    return bool(data)


@json_loads.register(int)
def _(data: int) -> Any:
    return int(data)


@json_loads.register(float)
def _(data: float) -> Any:
    return float(data)
