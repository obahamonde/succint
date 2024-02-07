import re
from typing import AsyncIterator

from agent_proto import Tool
from agent_proto.utils import setup_logging
from bs4 import BeautifulSoup
from pydantic import Field
from pyppeteer import browser, launch  # type: ignore

chrome: browser.Browser = None  # type: ignore
logger = setup_logging(__name__)


class GoogleSearchTool(Tool):
    """Performs a Google search and returns the URLs of the search results."""

    inputs: str = Field(..., description="The query to search for.")

    async def run(self) -> list[str]:
        try:
            global chrome  # pylint: disable=global-statement
            chrome = await launch(
                headless=True,
                args=["--no-sandbox"],
            )
            page = await chrome.newPage()  # type: ignore
            await page.setUserAgent(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            )
            await page.goto("https://www.google.com")  # type: ignore
            await page.type("input[name=q]", self.inputs)  # type: ignore
            await page.keyboard.press("Enter")  # type: ignore
            await page.waitForNavigation()  # type: ignore
            content = await page.content()  # type: ignore
            soup = BeautifulSoup(content, "lxml")
            links = soup.find_all("a")
            non_empty_links = [link.get("href") for link in links if link.get("href")]
            return [
                re.search(r"(?P<url>https?://[^\s]+)", link)["url"]  # type: ignore
                for link in non_empty_links
                if re.search(r"(?P<url>https?://[^\s]+)", link)
            ]

        except (RuntimeError, KeyError):
            return []  # type: ignore
        finally:
            await chrome.close()


class BrowsePageTool(Tool):
    """Retrieves the content of a web page."""

    inputs: str = Field(..., description="The URL of the web page to retrieve.")

    async def run(self) -> str:
        try:
            global chrome  # pylint: disable=global-statement
            chrome = await launch(
                headless=True,
                args=["--no-sandbox"],
            )
            page = await chrome.newPage()  # type: ignore
            await page.goto(self.inputs)  # type: ignore
            content = await page.content()  # type: ignore
            return BeautifulSoup(content, "lxml").get_text()
        except (RuntimeError, KeyError):
            return ""
        finally:
            await chrome.close()


async def get_children(url: str) -> AsyncIterator[str]:
    try:
        global chrome  # pylint: disable=global-statement
        visited = set[str]()
        seek = set[str]()
        chrome = await launch(
            headless=True,
            args=["--no-sandbox"],
        )
        page = await chrome.newPage()  # type: ignore

        async def get_links(page: browser.Page):
            links = await page.querySelectorAll("a")
            for link in links:
                href = await page.evaluate("(element) => element.href", link)
                if href and href.startswith("http"):

                    yield href

        seek.add(url)
        while seek:
            url = seek.pop()
            if url in visited:
                continue
            logger.info("Visiting %s", url)
            visited.add(url)
            await page.goto(url)  # type: ignore
            async for link in get_links(page):
                if link not in visited:
                    seek.add(link)
            yield url
    except (RuntimeError, KeyError):
        return
    finally:
        await chrome.close()


class WebCrawlerTool(Tool):
    """Crawls a website and returns the URLs of its pages."""

    inputs: str = Field(..., description="The URL of the website to crawl.")

    async def run(self) -> list[str]:
        return [url async for url in get_children(self.inputs)]
