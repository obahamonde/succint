from .ai import app as ai_controller
from .auth import app as auth_controller
from .docs import app as docs_controller
from .search import app as search_controller

__all__ = ["auth_controller", "ai_controller", "search_controller", "docs_controller"]
