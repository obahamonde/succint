from functools import cached_property
from typing import Any, Literal, Optional, TypeAlias, Union, cast

import torch
from fastapi import APIRouter
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer  # type: ignore

from src.database.application import Model, Surreal, get_db
from src.utils.decorators import async_io, robust

Value: TypeAlias = Union[str, int, float, bool, list[str]]
EmbeddingsModel: TypeAlias = Literal["all-MiniLM-L6-v2"]


@robust
@async_io
def tensor_dumps(obj: torch.Tensor | list[torch.Tensor]):
    return cast(list[float], obj.tolist())  # type: ignore


def tensor_loads(obj: Any):
    return torch.tensor(obj) if isinstance(obj, list) else obj


class Embeddings(Model):
    model: EmbeddingsModel = Field(
        default="all-MiniLM-L6-v2", description="Model to use for embeddings"
    )
    sentences: list[str] | str = Field(..., description="Sentences to embed")
    metadata: Optional[dict[str, Value]] = Field(
        default=None, description="Metadata to associate with the embeddings"
    )
    vector: list[float] = Field(..., description="Embeddings for the sentences")

    def __len__(self):
        return len(self.vector)

    def __call__(self):
        return (tensor_loads(self.vector)).to("cuda")


class Sentence(BaseModel):
    sentence: str | list[str] = Field(..., description="Sentence to embed")
    metadata: Optional[dict[str, Value]] = Field(
        default=None, description="Metadata to associate with the embeddings"
    )


class EmbeddingsController(APIRouter):
    @cached_property
    def ai(self):
        return SentenceTransformer("all-MiniLM-L6-v2")  # type: ignore

    async def upsert(self, *, data: Sentence, db: Surreal):
        encoded = torch.from_numpy(self.ai.encode(data.sentence))  # type: ignore
        embedding = Embeddings(
            sentences=data.sentence,
            metadata=data.metadata,
            vector=await tensor_dumps(encoded),  # type: ignore
        )
        response = await db.create(self.__class__.__name__, embedding.dict())
        return len(Embeddings(**response[0]))

    async def query(self, *, data: Sentence, db: Surreal):
        encoded = self.ai.encode(data.sentence)  # type: ignore
        assert isinstance(encoded, torch.Tensor)
        universe = [
            Embeddings(**item) for item in await db.select(self.__class__.__name__)
        ]
        return sorted(
            universe,
            key=lambda x: torch.cosine_similarity(encoded, x(), dim=0).item(),
            reverse=True,
        )
