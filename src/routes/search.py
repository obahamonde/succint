from fastapi import APIRouter

from ..tools import BrowsePageTool, GoogleSearchTool, WebCrawlerTool

app = APIRouter()


@app.post("/search")
async def search(query: str):
    return await GoogleSearchTool(inputs=query)()


@app.post("/browse")
async def browse(url: str):
    return await BrowsePageTool(inputs=url)()


@app.post("/crawl")
async def crawl(url: str):
    return await WebCrawlerTool(inputs=url)()
