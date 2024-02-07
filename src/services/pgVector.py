from functools import cached_property
from os import environ
from typing import Literal

from agent_proto import Tool
from agent_proto.utils import setup_logging
from langchain.embeddings.huggingface import HuggingFaceEmbeddings  # type: ignore
from langchain.vectorstores.pgvector import PGVector
from pydantic import Field


class PGRetrievalTool(Tool):
    """
    Retrieval Augmented Generation (RAG) tool for storing and querying sentence embeddings.
    """

    namespace: str = Field(..., description="Namespace on the knowledge store")
    inputs: str = Field(..., description="Input text to store or query")
    top_k: int = Field(5, description="Number of documents to retrieve")
    action: Literal["query", "upsert"] = Field("query")

    @cached_property
    def embeddings(self):
        return HuggingFaceEmbeddings()

    @cached_property
    def store(self):
        return PGVector(
            connection_string=environ["DATABASE_URL"],
            embedding_function=self.embeddings,
            embedding_length=1536,
            collection_name=self.namespace,
            collection_metadata=self.model_dump(),
            logger=setup_logging(self.__class__.__name__),
        )

    @cached_property
    def retriever(self):
        return self.store.as_retriever()

    async def query(self):
        return self.retriever.get_relevant_documents(query=self.inputs)

    async def upsert(self):
        response = await self.retriever.vectorstore.aadd_texts(  # type: ignore
            texts=self.inputs, metadatas=[self.model_dump()]
        )
        return len(response)

    async def run(self):
        return await self.query() if self.action == "query" else await self.upsert()
