import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any, AsyncGenerator, Generic, List, Optional, Type, TypeVar
from uuid import UUID

import numpy as np
import torch
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict
from surrealdb import Surreal

from ..utils.decorators import logger, robust


async def get_db(namespace: str, key: str):
    url = os.getenv("DATABASE_URL") or os.getenv("SURREAL_DB_URL") or "ws://db:8000/rpc"
    async with Surreal(url=url, max_size=2 ** 22) as db:
        if db.client_state.value == 2:
            await db.connect()
        await db.use(namespace, key)
        yield db


class ModelEncoder(json.JSONEncoder):
    @torch.no_grad()  # type: ignore
    def default(self, o: Any):  # type: ignore
        if isinstance(o, datetime):
            return o.astimezone().isoformat()
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, UUID):
            return str(o)
        if isinstance(o, torch.Tensor):
            return o.tolist()  # type: ignore
        return super().default(o)


class ModelDecoder(json.JSONDecoder):
    def __init__(self, *args: Any, **kwargs: Any):
        json.JSONDecoder.__init__(self, object_hook=self._object_hook, *args, **kwargs)

    def _object_hook(self, o: Any):
        """
        Custom object hook function that converts datetime strings to datetime objects.

        Args:
                                        o: The object to be decoded.

        Returns:
                                        The decoded object.

        """
        if isinstance(o, str):
            try:
                return datetime.fromisoformat(o)
            except ValueError:
                return o
        return o


class Model(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="allow", loc_by_alias=True
    )

    def dict(
        self,
        *,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = True,
        **dumps_kwargs: Any,
    ):
        return self.model_dump(
            mode="python",
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            **dumps_kwargs,
        )

    def json(
        self,
        *,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = True,
        **dumps_kwargs: Any,
    ):
        return json.dumps(
            super().dict(  # type: ignore
                by_alias=by_alias,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
                **dumps_kwargs,
            ),
            cls=ModelEncoder,
        )

    def model_dump_json(
        self,
        *,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = True,
        **dumps_kwargs: Any,
    ):
        return json.dumps(
            self.model_dump(
                mode="python",
                by_alias=by_alias,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
                **dumps_kwargs,
            ),
            cls=ModelEncoder,
        )


S = TypeVar("S", bound=BaseModel)
M = TypeVar("M", bound=Model)


class Service(Generic[S, M], ABC):
    @abstractmethod
    async def create(self, model: S) -> M:
        raise NotImplementedError

    @abstractmethod
    async def read(self, id: str) -> M:
        raise NotImplementedError

    @abstractmethod
    async def update(self, model: S) -> M:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, id: str) -> M:
        raise NotImplementedError

    @abstractmethod
    async def list(self) -> List[M]:
        raise NotImplementedError

    @abstractmethod
    async def query(self, query: str) -> List[M]:
        raise NotImplementedError

    @abstractmethod
    async def stream(self) -> AsyncGenerator[M, None]:
        raise NotImplementedError


class Repository(Generic[M, S]):
    model: Type[M]

    def __init__(self, *, model: Type[M]):
        self.model = model

    @property
    def table_name(self) -> str:
        return self.model.__name__

    @robust
    async def create_(self, data: S, db: Surreal) -> List[M]:
        response = await db.create(thing=self.table_name, data=data.model_dump())
        return [self.model(**res) for res in response]

    @robust
    async def read_(
        self,
        *,
        db: Surreal,
        id: Optional[str] = None,
        where: Optional[dict[str, Any]] = None,
    ) -> list[M]:
        if not id and not where:
            response = await db.select(thing=self.table_name)
        elif id:
            response = await db.select(thing=id)
        elif where:
            query = (
                f"SELECT * FROM {self.table_name} WHERE "
                + " AND ".join([f"{key} = '{value}'" for key, value in where.items()])
                + ";"
            )
            response = await db.query(query)
        else:
            logger.error("Invalid query parameters. id: %s, where: %s", id, where)
            return []
        return [self.model(**res) for res in response]

    @robust
    async def update_(self, id: str, data: M, db: Surreal) -> List[M]:
        response = await db.update(id, data.dict())  # type: ignore #[TODO] Migrate to pydantic v2
        if len(response) > 1:
            logger.error("Weird response from the database: %s", response)
            return []
        return [self.model(**res) for res in response]

    @robust
    async def delete_(self, id: str, db: Surreal) -> List[M]:
        response = await db.delete(id)
        if len(response):
            logger.error("Failed to delete item %s", id)
            return [self.model(**res) for res in response]
        return []


class Controller(APIRouter, Generic[M, S]):
    model: Type[M]

    def __init__(self, *, model: Type[M], repository: Repository[M, S]):
        super().__init__(
            prefix="/" + model.__name__.lower(), tags=[model.__name__.lower()]
        )
        self.model = model
        self.repository = repository

    @robust
    async def post_(self, *, data: S, db: Surreal) -> List[M]:
        return await self.repository.create_(data, db)

    @robust
    async def get_(
        self,
        *,
        db: Surreal,
        id: Optional[str] = None,
        where: Optional[dict[str, Any]] = None,
    ) -> list[M]:
        if id is None and where is None:
            response = await db.select(self.model.__name__)
        elif id is not None:
            response = await db.select(id)
        elif where is not None:
            self._validate_select_query(where)
            query = (
                f"SELECT * FROM {self.model.__name__} WHERE "
                + " AND ".join([f"{key} = '{value}'" for key, value in where.items()])
                + ";"
            )
            response = await db.query(query)
        else:
            raise ValueError("Invalid query parameters.")
        return [self.model(**res) for res in response]

    @robust
    async def put_(self, *, id: str, data: M, db: Surreal) -> List[M]:
        return await self.repository.update_(id, data, db)

    @robust
    async def delete_(self, *, id: str, db: Surreal) -> List[M]:
        return await self.repository.delete_(id, db)

    def _validate_select_query(self, query: dict[str, Any]) -> None:
        for key, value in query.items():
            if not hasattr(self.model, key):
                raise ValueError(f"Invalid query parameter: {key}.")
            if not isinstance(value, type(getattr(self.model, key))):
                raise ValueError(
                    f"Invalid query parameter type: {type(value)} for {key}."
                )
