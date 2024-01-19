from pydantic import BaseModel, Field
from typing import Any, List, Literal, Union
from ..database import DatabaseModel, Surreal

class MessageRequest(BaseModel):
	role: str = Field(..., description="Role of the message")
	message: str = Field(..., description="Message to send")