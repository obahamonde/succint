from os import environ
from typing import Any, List
from uuid import uuid4

from agent_proto import APIClient
from httpx import AsyncClient
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Query request schema."""

    namespace: str = Field(
        default="default", description="Namespace on the knowledge store"
    )
    topK: int = Field(default=5, description="Number of documents to retrieve")
    filter: dict[str, object] | None = Field(
        default=None, description="Filter for the query"
    )
    includeValues: bool = Field(
        default=False, description="Include values in the response"
    )
    includeMetadata: bool = Field(
        default=True, description="Include metadata in the response"
    )
    vector: list[float] = Field(..., description="Vector to query")


class QueryMatch(BaseModel):
    """Query match schema."""

    id: str = Field(..., description="ID of the document")
    score: float = Field(..., description="Score of the document")
    metadata: dict[str, object] = Field(..., description="Metadata of the document")


class QueryResponse(BaseModel):
    """Query response schema."""

    matches: list[QueryMatch] = Field(..., description="Matches from the query")
    namespace: str = Field(..., description="Namespace on the knowledge store")


class Vector(BaseModel):
    """Vector schema."""

    id: str = Field(
        default_factory=lambda: str(uuid4()), description="ID of the document"
    )
    values: list[float] = Field(..., description="Vector to store")
    metadata: dict[str, object] = Field({}, description="Metadata of the document")


class UpsertRequest(BaseModel):
    """Upsert request schema."""

    namespace: str = Field(
        default="default", description="Namespace on the knowledge store"
    )
    vectors: list[Vector] = Field(..., description="Vectors to upsert")


class UpsertResponse(BaseModel):
    """Upsert response schema."""

    upsertedCount: int = Field(..., description="Number of vectors upserted")


class PineconeClient(APIClient):
    """
    PineconeClient
    Pinecone client for storing and querying vectors.
    """

    namespace: str = Field(default="default")
    base_url: str = Field(default=environ["PINECONE_URL"], repr=True)
    headers: dict[str, str] = Field(
        default_factory=lambda: {"api-key": environ["PINECONE_API_KEY"]},
        init=True,
        repr=False,
    )

    def __load__(self, *args: Any, **kwargs: Any):
        return AsyncClient(base_url=self.base_url, headers=self.headers)

    async def upsert(self, embeddings: List[Vector]) -> UpsertResponse:
        """
        upsert
        Upsert embeddings into the vector index.

        Args:
                                        embeddings (List[Embedding]): Embeddings to upsert.

        Returns:
                                        UpsertResponse: Upsert response.
        """
        response = await self.post(
            "/vectors/upsert",
            json=UpsertRequest(
                namespace=self.namespace, vectors=embeddings
            ).model_dump(),
        )
        return UpsertResponse(**response)

    async def query(
        self,
        vector: List[float],
        topK: int = 5,
        filter: dict[str, Any] | None = None,
        includeValues: bool = False,
        includeMetadata: bool = True,
    ) -> QueryResponse:
        """
        query
        Query the vector index.

        Args:
                                        vector (List[float]): Vector to query.
                                        namespace (str, optional): Namespace on the knowledge store. Defaults to "default".
                                        topK (int, optional): Number of documents to retrieve. Defaults to 5.
                                        filter (dict[str, Any], optional): Filter for the query. Defaults to None.
                                        includeValues (bool, optional): Include values in the response. Defaults to False.
                                        includeMetadata (bool, optional): Include metadata in the response. Defaults to True.

        Returns:
                                        QueryResponse: Query response.
        """
        response = await self.post(
            "/query",
            json=QueryRequest(
                namespace=self.namespace,
                topK=topK,
                filter=filter,
                includeValues=includeValues,
                includeMetadata=includeMetadata,
                vector=vector,
            ).model_dump(),
        )
        return QueryResponse(**response)
