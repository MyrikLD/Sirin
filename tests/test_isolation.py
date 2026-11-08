import asyncio

import pytest

from voice_orchestrator.agent import list_server_tools
from voice_orchestrator.config import McpServerConfig
from voice_orchestrator.oauth import OAuthManager


async def _ignore(_: str) -> None:
    pass


async def test_failed_mcp_connect_does_not_cancel_the_caller(tmp_path):
    server = McpServerConfig(name="dead", url="http://127.0.0.1:1/mcp")
    with pytest.raises(RuntimeError):
        await list_server_tools(server, OAuthManager(tmp_path, _ignore))
    # A leaked anyio cancel scope would cancel this task at its next await.
    await asyncio.sleep(0.05)
