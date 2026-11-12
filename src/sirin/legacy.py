"""The JSON files of versions before the database, imported once at the first start."""

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import sqlalchemy as sa
from pydantic import TypeAdapter
from sqlalchemy.ext.asyncio import AsyncSession

from sirin import tables
from sirin.config import RuntimeSettings
from sirin.database import Database
from sirin.reminders import Reminder
from sirin.settings_store import save_settings
from sirin.tables import McpServer, McpToken, Setting

IMPORTED = ".imported"

_log = logging.getLogger(__name__)

_reminder_list = TypeAdapter(list[Reminder])


async def import_files(db: Database, data_dir: Path) -> None:
    """Imports settings.json, reminders.json and data/oauth/ into an empty database and renames them.

    The files are renamed only after the import is committed; a database that already holds settings is left
    alone, so a file put back by hand is never imported twice.
    """
    settings_path, reminders_path, oauth_dir = (
        data_dir / "settings.json",
        data_dir / "reminders.json",
        data_dir / "oauth",
    )
    found = [p for p in (settings_path, reminders_path, oauth_dir) if p.exists()]
    if not found:
        return
    async with db.session() as session, session.begin():
        has_settings = await session.scalar(sa.select(sa.func.count()).select_from(Setting))
        has_servers = await session.scalar(sa.select(sa.func.count()).select_from(McpServer))
        if has_settings or has_servers:
            _log.warning("not importing %s: the database already has settings", ", ".join(map(str, found)))
            return
        if settings_path.exists():
            settings = RuntimeSettings.model_validate_json(settings_path.read_text())
            await save_settings(session, settings)
            if oauth_dir.is_dir():
                await _import_tokens(session, settings, oauth_dir)
        if reminders_path.exists():
            for reminder in _reminder_list.validate_json(reminders_path.read_text()):
                await session.execute(sa.insert(tables.Reminder).values(**reminder.model_dump()))
    for path in found:
        path.rename(path.with_name(path.name + IMPORTED))
    _log.info("imported %s into the database", ", ".join(p.name for p in found))


async def _import_tokens(session: AsyncSession, settings: RuntimeSettings, oauth_dir: Path) -> None:
    rows = await session.execute(sa.select(McpServer.id, McpServer.name).where(McpServer.owner_id.is_(None)))
    ids = {r.name: r.id for r in rows}
    for server in settings.mcp_servers:
        # The file name the old file store gave the server.
        key = hashlib.sha256(f"{server.name}\n{server.url}".encode()).hexdigest()[:16]
        path = oauth_dir / f"{key}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        expires_at = data.get("expires_at")
        await session.execute(
            sa.insert(McpToken).values(
                server_id=ids[server.name],
                tokens=data.get("tokens"),
                expires_at=datetime.fromtimestamp(expires_at, UTC) if expires_at is not None else None,
                client_info=data.get("client"),
                server_metadata=data.get("server_metadata"),
            )
        )
