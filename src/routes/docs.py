import asyncio

from fastapi import APIRouter, File, UploadFile

from ..agent import ChatGPT
from ..services.pineconeVector import Vector
from ..utils import process_document

app = APIRouter()


@app.post("/upload/{namespace}")
async def load(namespace: str, file: UploadFile = File(...)):
    agent = ChatGPT(namespace=namespace)
    return await agent.upload(file=file)


@app.post("/upsert/{namespace}")
async def upsert(namespace: str, file: UploadFile = File(...)):
    agent = ChatGPT(namespace=namespace)
    documents = await process_document(file)
    embeddings = await asyncio.gather(
        *[agent.embed(message=document) for document in documents]
    )
    vectors = [
        Vector(values=embedding, metadata={"text": document})
        for embedding, document in zip(embeddings, documents)
    ]
    return await agent.retriever.upsert(embeddings=vectors)
