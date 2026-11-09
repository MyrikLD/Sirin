import asyncio
from typing import Literal

from pydantic import BaseModel, Field


class ItemStatus(BaseModel):
    state: Literal["ok", "pending", "error"] = Field(description="ok: working; pending: connecting; error: see detail")
    detail: str | None = Field(default=None, description="What the item reports about itself, or why it fails")


class Status(BaseModel):
    items: dict[str, dict[str, ItemStatus]] = Field(
        description="Per settings list (devices, mcp_servers): the state of each item by its name"
    )


class StatusBoard:
    """Wakes status watchers whenever a device or MCP server changes state."""

    def __init__(self) -> None:
        self.changed = asyncio.Event()
        self.closed = False

    def close(self) -> None:
        """Ends the watchers' streams, so a server shutting down does not wait for open settings pages."""
        self.closed = True
        self.touch()

    def touch(self) -> None:
        # Watchers hold the event they took before reading the status, so a change between their
        # read and their wait still wakes them.
        self.changed.set()
        self.changed = asyncio.Event()
