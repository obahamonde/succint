import base64
import json
from typing import AsyncGenerator
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup  # type: ignore
from pyppeteer import browser, launch  # type: ignore

from ..utils.decorators import robust, setup_logging
from ._json import SoupEncoder
from ._proto import Crawler

logger = setup_logging(name=__name__)


async def get_page() -> AsyncGenerator[browser.Page, None]:
    browser = await launch()
    yield await browser.newPage()


class SPACrawler(Crawler[browser.Page]):
    @robust
    async def html(self, tool: browser.Page, *, url: str) -> str:
        await tool.goto(url)  # type: ignore
        return await tool.content()

    @robust
    async def screenshot(self, tool: browser.Page, *, url: str) -> bytes:
        await tool.goto(url)  # type: ignore
        return await tool.screenshot()  # type: ignore

    @robust
    async def json(self, tool: browser.Page, *, url: str) -> str:
        await tool.goto(url)  # type: ignore
        return json.dumps(BeautifulSoup(await tool.content(), "lxml"), cls=SoupEncoder)

    @robust
    async def pdf(self, tool: browser.Page, *, url: str) -> str:
        await tool.goto(url)  # type: ignore
        pdf = await tool.pdf()  # type: ignore
        return f"data:application/pdf;base64,{base64.b64encode(pdf).decode('utf-8')}"

    async def children(
        self, tool: browser.Page, *, base_url: str, limit: int = 100
    ) -> AsyncGenerator[str, None]:
        await tool.goto(base_url)  # type: ignore

        async def get_urls(url: str) -> set[str]:
            await tool.goto(url)  # type: ignore
            logger.info("Crawling %s", url)
            content = await tool.content()  # type: ignore
            soup = BeautifulSoup(content, "lxml")
            urls: set[str] = set()
            for link in soup.find_all("a"):
                href = link.get("href")
                if href and "#" not in href:
                    if href.startswith("/"):
                        href = urljoin(base=base_url, url=href)
                    elif (
                        href.startswith("http")
                        and urlparse(url=href).netloc == urlparse(base_url).netloc  # type: ignore
                    ):
                        urls.add(href)  # type: ignore
                if len(urls) >= limit:
                    break
            return urls

        visited_urls: set[str] = set()
        urls: set[str] = set()
        urls.update(await get_urls(url=base_url))
        while urls:
            try:
                url = urls.pop()
                if url not in visited_urls:
                    yield url
                    visited_urls.add(url)
                    urls.update(await get_urls(url=url))
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(exc)
                continue
