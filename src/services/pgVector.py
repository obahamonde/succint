from abc import ABC, abstractmethod
from functools import cached_property
from os import environ
from typing import AsyncGenerator, Generic, TypeVar

from agent_proto.utils import robust, setup_logging
from langchain.embeddings.openai import OpenAIEmbeddings  # type: ignore
from langchain.vectorstores.pgvector import PGVector  # type: ignore
from langchain_core.documents import Document
from pydantic import BaseModel, Field

T = TypeVar("T")


class ScoredDocument(Document):
    score: float


class Retriever(BaseModel, ABC, Generic[T]):
    """
    Base class for a retriever tool.
    """

    @cached_property
    @abstractmethod
    def embeddings(self) -> T:
        pass

    @robust
    @abstractmethod
    async def query(self, inputs: str, top_k: int) -> list[ScoredDocument]:
        pass

    @robust
    @abstractmethod
    async def upsert(self, inputs: str) -> int:
        pass


class PGRetriever(Retriever[OpenAIEmbeddings]):
    """
    Retrieval Augmented Generation (RAG) tool for storing and querying sentence embeddings.
    """

    namespace: str = Field(..., description="Namespace on the knowledge store")

    @cached_property
    def embeddings(self):
        return OpenAIEmbeddings()

    @cached_property
    def store(self):
        return PGVector(
            connection_string=environ["DATABASE_URL"]
            .replace("postgres://", "postgresql+psycopg2://")
            .split("?")[0],
            embedding_function=self.embeddings,
            embedding_length=1536,
            collection_name=self.namespace,
            collection_metadata=self.model_dump(),
            logger=setup_logging(self.__class__.__name__),
        )

    @cached_property
    def retriever(self):
        return self.store.as_retriever()

    async def _query(
        self, inputs: str, top_k: int = 5
    ) -> AsyncGenerator[ScoredDocument, None]:
        response = await self.retriever.vectorstore.asimilarity_search_with_relevance_scores(  # type: ignore
            query=inputs, k=top_k
        )
        for res in response:
            doc, score = res
            yield ScoredDocument(**doc.dict(), score=score)

    @robust
    async def query(self, inputs: str, top_k: int = 5):
        return [doc async for doc in self._query(inputs, top_k)]

    @robust
    async def upsert(self, inputs: str):
        response = await self.retriever.vectorstore.aadd_texts(  # type: ignore
            texts=inputs, metadatas=[{"namespace": self.namespace, "text:": inputs}]
        )
        return len(response)
