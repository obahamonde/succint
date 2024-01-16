from pydantic import BaseModel, Field

from .database.application import Application, Model
from .services import create_ai_controller


class User(Model):
    name: str = Field(..., title="Name of the user", max_length=128)
    email: str = Field(..., title="Email of the user", max_length=128)
    password: str = Field(..., title="Password of the user", max_length=128)


class UserSchema(BaseModel):
    name: str = Field(..., title="Name of the user", max_length=128)
    email: str = Field(..., title="Email of the user", max_length=128)
    password: str = Field(..., title="Password of the user", max_length=128)


def create_app():
    app = Application(title="OpenCopilot", version="0.0.1")
    return create_ai_controller(app)


__all__ = ["create_app"]
