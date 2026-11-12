from pathlib import Path
from typing import Self

from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, event
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

MIGRATIONS = Path(__file__).resolve().parent / "migrations"


class Database:
    """The SQLite database: WAL mode, foreign keys on, migrated to the latest schema by migrate()."""

    def __init__(self, url: str) -> None:
        memory = ":memory:" in url
        # An in-memory database lives in its one connection, so every session has to share it.
        self.engine: AsyncEngine = create_async_engine(url, poolclass=StaticPool if memory else None)
        event.listen(self.engine.sync_engine, "connect", _on_connect)
        self.session = async_sessionmaker(self.engine, expire_on_commit=False)

    @classmethod
    def in_dir(cls, data_dir: Path) -> Self:
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / "sirin.db"
        # It holds API keys and OAuth tokens; SQLite gives its -wal and -shm files the same mode.
        path.touch(mode=0o600)
        path.chmod(0o600)
        return cls(f"sqlite+aiosqlite:///{path}")

    async def migrate(self) -> None:
        async with self.engine.connect() as conn:
            # A batch migration recreates a table; with foreign keys on, dropping the old one would cascade.
            # The pragma has no effect inside a transaction, so it is set before the migration begins one.
            await conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
            try:
                await conn.run_sync(_upgrade)
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
            finally:
                await conn.exec_driver_sql("PRAGMA foreign_keys=ON")
                await conn.commit()

    async def close(self) -> None:
        await self.engine.dispose()


def _on_connect(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def _upgrade(connection: Connection) -> None:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.attributes["connection"] = connection
    command.upgrade(cfg, "head")
