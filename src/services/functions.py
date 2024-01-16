from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from src.utils.decorators import robust


class AIFunctionStruct(BaseModel):
    function: AIFunctionDefinition


class AIFunctionCall(BaseModel):
    data: Any
    name: str


class AIFunctionDefinition(BaseModel):
    name: str
    parameters: dict[str, object]


class AIFunction(ABC, BaseModel):
    model_config = ConfigDict(extra="allow", loc_by_alias=True)

    @classmethod
    def definition(cls) -> dict[str, object]:
        _schema = cls.schema()  # type: ignore
        _name = cls.__name__.lower()
        _description = cls.__doc__
        assert isinstance(
            _description, str
        ), "All Mistral functions must have a docstring describing their purpose"
        _parameters = cast(
            dict[str, object],
            (
                {
                    "type": "object",
                    "properties": {
                        k: v for k, v in _schema["properties"].items() if k != "self"
                    },
                    "required": _schema.get("required", []),
                }
            ),
        )
        return AIFunctionDefinition(
            name=_name, parameters=_parameters  # type: ignore
        ).model_dump()

    @abstractmethod
    async def run(self) -> Any:
        raise NotImplementedError

    @robust
    async def __call__(self) -> AIFunctionCall:
        data = await self.run()
        return AIFunctionCall(
            data=data,
            name=self.__class__.__name__.lower(),
        )
