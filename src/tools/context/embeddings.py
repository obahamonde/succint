from functools import cached_property
from typing import Any, Literal, Optional, TypeAlias, Union, cast, List

import torch
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer  # type: ignore

from src.database.application import DatabaseModel, Surreal

Value: TypeAlias = Union[str, int, float, bool, list[str]]
EmbeddingsModel: TypeAlias = Literal["all-MiniLM-L6-v2"]


def tensor_dumps(obj: torch.Tensor | list[torch.Tensor]):
    return cast(list[float], obj.tolist())  # type: ignore


def tensor_loads(obj: Any):
    if isinstance(obj, list):
        return torch.tensor(obj)
    if isinstance(obj, torch.Tensor):
        return obj
    raise TypeError(f"Invalid type {type(obj)}")


class Embeddings(DatabaseModel):
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
        return tensor_loads(self.vector)


class Sentence(BaseModel):
    sentence: str | list[str] = Field(..., description="Sentence to embed")
    metadata: Optional[dict[str, Value]] = Field(
        default=None, description="Metadata to associate with the embeddings"
    )


class VectorStore:
    @cached_property
    def embeddings(self):
        return SentenceTransformer("all-MiniLM-L6-v2")  # type: ignore

    async def upsert(self, *, data: Sentence, db: Surreal):
        encoded = torch.from_numpy(self.embeddings.encode(data.sentence))  # type: ignore
        embedding = Embeddings(
            sentences=data.sentence,
            metadata=data.metadata,
            vector=tensor_dumps(encoded),  # type: ignore
        )
        return await db.create(self.__class__.__name__, embedding.dict())

    async def query(self, *, data: Sentence, db: Surreal, top_k:int=10) -> list[tuple[str, float]]:
        # Encode the input sentence
        encoded_query = torch.from_numpy(self.embeddings.encode(data.sentence))

        # Fetch the universe of embeddings from the database
        universe = [
            Embeddings(**item) for item in await db.select(self.__class__.__name__)
        ]
        scores = torch.nn.functional.cosine_similarity(
            encoded_query, torch.tensor([item.vector for item in universe])
        )
        return ({'sentence': universe[i].sentences, 'score': scores[i].item()} for i in scores.argsort(descending=True)[:top_k])