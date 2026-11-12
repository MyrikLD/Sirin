"""The MCP servers of single people: rows of `mcp_servers` with an owner."""

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from sirin.agent import PersonalServer
from sirin.config import McpServerConfig
from sirin.tables import McpServer, McpToken, User


class NameTaken(ValueError):
    pass


_COLUMNS = [McpServer.id, McpServer.owner_id, McpServer.name, McpServer.url, McpServer.headers, McpServer.oauth]


def _server(row: sa.Row) -> PersonalServer:
    return PersonalServer(
        id=row.id,
        owner_id=row.owner_id,
        cfg=McpServerConfig(name=row.name, url=row.url, headers=row.headers, oauth=row.oauth),
    )


async def all_personal(session: AsyncSession) -> list[PersonalServer]:
    rows = await session.execute(sa.select(*_COLUMNS).where(McpServer.owner_id.is_not(None)).order_by(McpServer.id))
    return [_server(r) for r in rows]


async def personal_of(session: AsyncSession, owner_id: int) -> list[PersonalServer]:
    rows = await session.execute(sa.select(*_COLUMNS).where(McpServer.owner_id == owner_id).order_by(McpServer.id))
    return [_server(r) for r in rows]


async def get_personal(session: AsyncSession, owner_id: int, server_id: int) -> PersonalServer | None:
    rows = await session.execute(sa.select(*_COLUMNS).where(McpServer.id == server_id, McpServer.owner_id == owner_id))
    row = rows.one_or_none()
    return _server(row) if row is not None else None


async def owners(session: AsyncSession) -> dict[int, str]:
    """The names of the people who have personal servers, for the admin's overview."""
    rows = await session.execute(
        sa.select(User.id, User.name).where(
            User.id.in_(sa.select(McpServer.owner_id).where(McpServer.owner_id.is_not(None)))
        )
    )
    return {r.id: r.name for r in rows}


async def personal_names(session: AsyncSession) -> set[str]:
    return set(await session.scalars(sa.select(McpServer.name).where(McpServer.owner_id.is_not(None))))


async def _check_name(session: AsyncSession, owner_id: int, name: str, server_id: int | None = None) -> None:
    """A tool prefix is unique among the servers one turn can see: the shared ones and the person's own."""
    clash = sa.select(McpServer.id).where(
        McpServer.name == name,
        sa.or_(McpServer.owner_id.is_(None), McpServer.owner_id == owner_id),
    )
    if server_id is not None:
        clash = clash.where(McpServer.id != server_id)
    if await session.scalar(clash) is not None:
        raise NameTaken(f"the name {name!r} is taken by a shared server or another server of yours")


async def create_personal(session: AsyncSession, owner_id: int, cfg: McpServerConfig) -> int:
    await _check_name(session, owner_id, cfg.name)
    result = await session.execute(
        sa.insert(McpServer)
        .values(owner_id=owner_id, **cfg.model_dump(mode="json", include={"name", "url", "headers", "oauth"}))
        .returning(McpServer.id)
    )
    return result.scalar_one()


async def update_personal(session: AsyncSession, owner_id: int, server_id: int, cfg: McpServerConfig) -> bool:
    current = await get_personal(session, owner_id, server_id)
    if current is None:
        return False
    await _check_name(session, owner_id, cfg.name, server_id)
    await session.execute(
        sa.update(McpServer)
        .where(McpServer.id == server_id)
        .values(**cfg.model_dump(mode="json", include={"name", "url", "headers", "oauth"}))
    )
    if current.cfg.url != cfg.url:
        # A token of one URL means nothing to another: the moved server starts from a clean login.
        await session.execute(sa.delete(McpToken).where(McpToken.server_id == server_id))
    return True


async def delete_personal(session: AsyncSession, owner_id: int, server_id: int) -> bool:
    """Its tokens go with it."""
    deleted = await session.scalar(
        sa.delete(McpServer).where(McpServer.id == server_id, McpServer.owner_id == owner_id).returning(McpServer.id)
    )
    return deleted is not None
