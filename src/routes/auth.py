from os import environ
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from httpx import AsyncClient
from pydantic import BaseModel, Field

from prisma.models import IUser


class User(BaseModel):
    """
    Auth0 User
    """

    email: Optional[str] = Field(default=None)
    email_verified: Optional[bool] = Field(default=False)
    family_name: Optional[str] = Field(default=None)
    given_name: Optional[str] = Field(default=None)
    locale: Optional[str] = Field(default=None)
    name: str = Field(...)
    nickname: Optional[str] = Field(default=None)
    picture: Optional[str] = Field(default=None)
    sub: str = Field(...)
    updated_at: Optional[str] = Field(default=None)


def get_token(request: Request) -> str:
    try:
        return request.headers["Authorization"].split("Bearer ")[1]
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=401, detail="Invalid token") from e


AUTH0_URL = environ["AUTH0_URL"]

app = APIRouter()


@app.post("/auth")
async def auth_endpoint(token: str = Depends(get_token)):
    async with AsyncClient(
        base_url=AUTH0_URL, headers={"Authorization": f"Bearer {token}"}
    ) as client:
        response = await client.get("/userinfo")
        user = User(**response.json())
        return await IUser.prisma().upsert(
            where={"sub": user.sub},
            data={
                "create": user.model_dump(),  # type: ignore
                "update": user.model_dump(),  # type: ignore
            },
        )
