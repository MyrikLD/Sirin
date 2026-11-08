import asyncio
from collections.abc import Callable
from typing import Self

from mcp.types import Tool
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import WrapperToolset


class IsolatedToolset(WrapperToolset[None]):
    """An MCP toolset whose connection is opened and held in a task of its own.

    A failed anyio transport leaks its cancel scope into the task that opened it and silently cancels
    that task; here it is a private holder task, never the caller. Entering is re-entrant, as the agent
    run enters toolsets it already holds: the first enter connects and the last exit disconnects.
    """

    def __init__(self, wrapped: MCPToolset[None], on_lost: Callable[[Exception], None] | None = None) -> None:
        super().__init__(wrapped)
        self._mcp = wrapped
        self._on_lost = on_lost
        self._users = 0
        self._connected: asyncio.Future[None] | None = None
        self._stop = asyncio.Event()
        self._holder: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Self:
        self._users += 1
        if self._connected is None:
            self._connected = asyncio.get_running_loop().create_future()
            self._stop = asyncio.Event()
            self._holder = asyncio.create_task(self._hold(self._connected))
        try:
            await asyncio.shield(self._connected)
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, *args: object) -> bool | None:
        self._users -= 1
        if self._users == 0 and self._holder is not None:
            self._stop.set()
            await asyncio.gather(self._holder, return_exceptions=True)
            self._holder, self._connected = None, None
        return None

    async def list_tools(self) -> list[Tool]:
        return await self._mcp.list_tools()

    async def _hold(self, connected: "asyncio.Future[None]") -> None:
        try:
            async with self._mcp:
                connected.set_result(None)
                await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001
            if not connected.done():
                connected.set_exception(exc if isinstance(exc, Exception) else RuntimeError("MCP connection cancelled"))
            elif self._on_lost is not None and isinstance(exc, Exception):
                self._on_lost(exc)
