import asyncio

from fastapi import APIRouter, File, UploadFile

from ..agent.mistral import MistralAI
from ..services.pineconeVector import Vector
from ..utils import process_document

app = APIRouter()


@app.post("/upload/{namespace}")
async def upload(namespace: str, file: UploadFile = File(...)):
    agent = MistralAI(namespace=namespace)
    url = await agent.upload(file=file)
    return {"url": url}


@app.post("/upsert/{namespace}")
async def upsert(namespace: str, file: UploadFile = File(...)):
    agent = MistralAI(namespace=namespace)
    documents = await process_document(file)
    embeddings = await asyncio.gather(
        *[agent.embed(message=document) for document in documents]
    )
    vectors = [
        Vector(values=embedding, metadata={"text": document})
        for embedding, document in zip(embeddings, documents)
    ]
    return await agent.retriever.upsert(embeddings=vectors)
