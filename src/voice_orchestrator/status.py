from typing import Literal

from pydantic import BaseModel, Field


class ItemStatus(BaseModel):
    state: Literal["ok", "pending", "error"] = Field(description="ok: working; pending: connecting; error: see detail")
    detail: str | None = Field(default=None, description="What the item reports about itself, or why it fails")


class Status(BaseModel):
    items: dict[str, dict[str, ItemStatus]] = Field(
        description="Per settings list (devices, mcp_servers): the state of each item by its name"
    )
