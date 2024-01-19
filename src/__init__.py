from .database.application import Application
from .services import ai_controller, auth_controller


def create_app():
    app = Application(title="NotOpenAI", version="0.0.1")
    return auth_controller(ai_controller(app))


__all__ = ["create_app"]
