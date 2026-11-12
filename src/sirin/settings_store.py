import logging
from typing import Any

import sqlalchemy as sa
from pydantic import TypeAdapter, ValidationError
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession

from sirin.config import McpServerConfig, RuntimeSettings
from sirin.database import Database
from sirin.tables import McpServer, McpToken, Setting

_log = logging.getLogger(__name__)

# The blocks of RuntimeSettings kept as rows of `settings`; MCP servers have a table of their own.
BLOCKS = [name for name in RuntimeSettings.model_fields if name != "mcp_servers"]


class SettingsStore:
    """RuntimeSettings in the database: one `settings` row per block and the shared rows of `mcp_servers`."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def load(self) -> RuntimeSettings:
        async with self._db.session() as session:
            return await load_settings(session)

    async def save(self, settings: RuntimeSettings) -> None:
        async with self._db.session() as session, session.begin():
            await save_settings(session, settings)


async def load_settings(session: AsyncSession) -> RuntimeSettings:
    blocks = await _stored_blocks(session)
    rows = await session.execute(
        sa.select(McpServer.name, McpServer.url, McpServer.headers, McpServer.oauth)
        .where(McpServer.owner_id.is_(None))
        .order_by(McpServer.id)
    )
    return RuntimeSettings.model_validate(_valid_blocks(blocks) | {"mcp_servers": [dict(r) for r in rows.mappings()]})


async def save_settings(session: AsyncSession, settings: RuntimeSettings) -> None:
    """Writes only the blocks that differ from the stored ones."""
    stored = await _stored_blocks(session)
    for name in BLOCKS:
        value = settings.model_dump(mode="json", include={name})[name]
        if stored.get(name) != value:
            stmt = insert(Setting).values(name=name, value=value)
            await session.execute(stmt.on_conflict_do_update(index_elements=[Setting.name], set_={"value": value}))
    await _save_shared_servers(session, settings.mcp_servers)


async def _stored_blocks(session: AsyncSession) -> dict[str, Any]:
    rows = await session.execute(sa.select(Setting.name, Setting.value).where(Setting.name.in_(BLOCKS)))
    return {r.name: r.value for r in rows}


def _valid_blocks(blocks: dict[str, Any]) -> dict[str, Any]:
    """The stored blocks that fit the current RuntimeSettings; one written by another version starts from defaults."""
    valid = {}
    for name, value in blocks.items():
        try:
            TypeAdapter(RuntimeSettings.model_fields[name].annotation).validate_python(value)
        except ValidationError as exc:
            _log.warning("the stored %r settings do not fit this version and are replaced by defaults: %s", name, exc)
            continue
        valid[name] = value
    return valid


async def _save_shared_servers(session: AsyncSession, servers: list[McpServerConfig]) -> None:
    """Matches the servers by name: a kept server keeps its row id, so its tokens stay with it."""
    rows = await session.execute(
        sa.select(McpServer.id, McpServer.name, McpServer.url, McpServer.headers, McpServer.oauth).where(
            McpServer.owner_id.is_(None)
        )
    )
    existing = {r.name: r for r in rows}
    wanted = {s.name for s in servers}
    gone = [r.id for name, r in existing.items() if name not in wanted]
    if gone:
        await session.execute(sa.delete(McpServer).where(McpServer.id.in_(gone)))
    for server in servers:
        values = server.model_dump(mode="json", include={"url", "headers", "oauth"})
        row = existing.get(server.name)
        if row is None:
            await session.execute(sa.insert(McpServer).values(owner_id=None, name=server.name, **values))
            continue
        if values == {"url": row.url, "headers": row.headers, "oauth": row.oauth}:
            continue
        await session.execute(sa.update(McpServer).where(McpServer.id == row.id).values(**values))
        if row.url != server.url:
            # Tokens of one URL mean nothing to another: the moved server starts from a clean login.
            await session.execute(sa.delete(McpToken).where(McpToken.server_id == row.id))
