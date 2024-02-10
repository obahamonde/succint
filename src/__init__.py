from fastapi import FastAPI as Application

from .agent import ChatGPT
from .routes import ai_controller, auth_controller, docs_controller, search_controller


def create_app():
    app = Application(
        title="NotOpenAI",
        version="0.0.1",
        servers=[
            {"url": "https://api.oscarbahamonde.com", "description": "Local Server"}
        ],
    )
    app.include_router(auth_controller, prefix="/api")
    app.include_router(ai_controller, prefix="/api")
    app.include_router(search_controller, prefix="/api")
    app.include_router(docs_controller, prefix="/api")
    return app


__all__ = ["create_app", "ChatGPT"]
