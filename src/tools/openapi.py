import tiktoken
from agent_proto import Tool, async_io
from aiohttp import ClientSession
from langchain.agents.agent_toolkits.openapi import planner
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.chat_models.openai import ChatOpenAI  # type: ignore
from langchain.requests import RequestsWrapper


@async_io
def tokenize(text: str) -> list[int]:
    encoding = tiktoken.get_encoding(encoding_name="cl100k_base")
    return encoding.encode(text)


@async_io
def sanitize_openapi_spec(spec: dict[str, object]) -> dict[str, object]:
    return reduce_openapi_spec(spec)  # type: ignore


class Plugin(Tool):
    """OpenAPI plugin for the agent."""

    openapi_url: str
    inputs: str
    headers: dict[str, str]

    async def run(self):
        async with ClientSession(headers=self.headers) as session:
            try:
                spec = await (await session.get(self.openapi_url)).json()
            except Exception:
                spec = await (
                    await session.get(
                        f"https://api.oscarbahamonde.com/static/{self.openapi_url}"
                    )
                ).json()
            executor = planner.create_openapi_agent(  # type: ignore
                api_spec=await sanitize_openapi_spec(spec=spec),  # type: ignore
                requests_wrapper=RequestsWrapper(
                    aiosession=session, headers=self.headers
                ),
                llm=ChatOpenAI(
                    openai_api_base="https://app.oscarbahamonde.com/v1",
                    model="mistralai/Mistral-7B-Instruct-v0.2",
                    # model="gpt-4-0125-preview",
                ),
            )
            return await executor.arun(self.inputs)
