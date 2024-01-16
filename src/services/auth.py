import json

from fastapi import APIRouter, Depends
from google_auth_oauthlib.flow import Flow  # type: ignore
from httpx import AsyncClient
from pydantic import BaseModel, Field  # pylint: disable=E0611

from ..database.application import Model, Surreal, get_db  # type: ignore
from ..utils.decorators import async_io


class User(Model):
    id: str = Field(..., alias="sub")
    name: str
    given_name: str
    family_name: str
    picture: str
    locale: str = Field(default="en")


class WebConfig(BaseModel):
    client_id: str
    project_id: str
    auth_uri: str
    token_uri: str
    auth_provider_x509_cert_url: str
    client_secret: str
    redirect_uris: list[str]
    javascript_origins: list[str]


class Code(BaseModel):
    code: str


class Config(BaseModel):
    web: WebConfig


@async_io
def get_flow():
    config_data = json.loads(open("config.json", encoding="utf-8").read())
    config = Config(**config_data)
    return Flow.from_client_config(  # type: ignore
        client_config={"web": config.web.model_dump()},
        scopes=[
            "https://www.googleapis.com/auth/gmail.send",
            "https://www.googleapis.com/auth/calendar.events",
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/documents",
            "https://www.googleapis.com/auth/presentations",
            "https://www.googleapis.com/auth/forms",
            "https://www.googleapis.com/auth/userinfo.profile",
            "openid",
        ],
        redirect_uri="http://localhost:3000",
    )


def add_auth():
    app = APIRouter(prefix="/auth", tags=["google"])

    @app.get("/")
    async def _():
        flow = await get_flow()
        auth_url, _ = flow.authorization_url(prompt="consent")  # type: ignore
        return {"url": auth_url}  # type: ignore

    @app.post("/")
    async def _(code: Code, db: Surreal = Depends(get_db)):
        flow = await get_flow()
        flow.fetch_token(code=code.code)  # type: ignore
        credentials = flow.credentials
        token = credentials.token  # type: ignore
        assert isinstance(token, str)
        async with AsyncClient(headers={"Authorization": f"Bearer {token}"}) as client:
            response = await client.get("https://www.googleapis.com/oauth2/v2/userinfo")
            response.raise_for_status()
            data = response.json()
            user_info = User(**data)
            existing = await db.query(
                f"SELECT * FROM users WHERE sub='{user_info.sub}'"
            )
            if len(existing) == 0:
                response = await db.create(
                    user_info.__class__.__name__, user_info.dict()
                )
            else:
                response = await db.update(
                    user_info.__class__.__name__, user_info.dict()
                )
            user = User(**response[0])
            return {"token": token, "user_info": user}  # type: ignore

    return app
