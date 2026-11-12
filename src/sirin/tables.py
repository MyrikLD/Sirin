from datetime import UTC, datetime

import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeDecorator


class UtcDateTime(TypeDecorator[datetime]):
    """An aware datetime stored as naive UTC: SQLite keeps no time zone."""

    impl = sa.DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: sa.Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(f"naive datetime {value} would be stored without a time zone")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: sa.Dialect) -> datetime | None:
        return None if value is None else value.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    metadata = sa.MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


class User(Base):
    __tablename__ = "users"

    id = sa.Column(sa.Integer, primary_key=True)
    login = sa.Column(sa.String, nullable=False, unique=True)
    # What the LLM sees; two people may share it.
    name = sa.Column(sa.String, nullable=False)
    password_hash = sa.Column(sa.String, nullable=False)
    role = sa.Column(sa.String, nullable=False)
    created_at = sa.Column(UtcDateTime, nullable=False)


class LoginSession(Base):
    """A login on the web page; the browser holds the token in a cookie, the table only its hash."""

    __tablename__ = "sessions"

    id = sa.Column(sa.String, primary_key=True)
    user_id = sa.Column(sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = sa.Column(UtcDateTime, nullable=False)
    expires_at = sa.Column(UtcDateTime, nullable=False)


class Setting(Base):
    """One block of RuntimeSettings, or one single value, as JSON."""

    __tablename__ = "settings"

    name = sa.Column(sa.String, primary_key=True)
    value = sa.Column(sa.JSON, nullable=False)


class McpServer(Base):
    __tablename__ = "mcp_servers"

    id = sa.Column(sa.Integer, primary_key=True)
    # NULL: shared by everyone.
    owner_id = sa.Column(sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    # The tool prefix.
    name = sa.Column(sa.String, nullable=False)
    url = sa.Column(sa.String, nullable=False)
    headers = sa.Column(sa.JSON, nullable=False)
    oauth = sa.Column(sa.Boolean, nullable=False)

    __table_args__ = (
        sa.UniqueConstraint(owner_id, name),
        # SQLite treats NULL owners as distinct in the constraint above, so shared names need their own index.
        sa.Index("ix_mcp_servers_shared_name", name, unique=True, sqlite_where=owner_id.is_(None)),
    )


class McpToken(Base):
    """The OAuth state of one MCP server: tokens, the registered client and the authorization server metadata."""

    __tablename__ = "mcp_tokens"

    server_id = sa.Column(sa.Integer, sa.ForeignKey("mcp_servers.id", ondelete="CASCADE"), primary_key=True)
    tokens = sa.Column(sa.JSON, nullable=True)
    expires_at = sa.Column(UtcDateTime, nullable=True)
    client_info = sa.Column(sa.JSON, nullable=True)
    server_metadata = sa.Column(sa.JSON, nullable=True)


class Reminder(Base):
    __tablename__ = "reminders"

    id = sa.Column(sa.String, primary_key=True)
    device = sa.Column(sa.String, nullable=False)
    due = sa.Column(UtcDateTime, nullable=False)
    prompt = sa.Column(sa.Text, nullable=False)
    created = sa.Column(UtcDateTime, nullable=False)
    schedule = sa.Column(sa.String, nullable=True)
    until = sa.Column(UtcDateTime, nullable=True)
