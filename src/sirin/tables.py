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


class Voice(Base):
    """A voice heard in the history: created for every speaker nobody's voice matched, linked to a person by hand."""

    __tablename__ = "voices"

    id = sa.Column(sa.Integer, primary_key=True)
    # NULL: nobody has said it is theirs; such a voice lives while history turns point to it.
    user_id = sa.Column(sa.Integer, sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    # The profile it is identified by: float32, the mean of its confident turns.
    embedding = sa.Column(sa.LargeBinary, nullable=False)
    # The isolation network's tensors (np.savez), about 10 MB; NULL until a turn of 3 s of speech or more.
    voiceprint = sa.Column(sa.LargeBinary, nullable=True)
    created_at = sa.Column(UtcDateTime, nullable=False)


class Turn(Base):
    """One utterance and its answer: the history."""

    __tablename__ = "turns"

    id = sa.Column(sa.Integer, primary_key=True)
    started_at = sa.Column(UtcDateTime, nullable=False, index=True)
    device = sa.Column(sa.String, nullable=False)
    conversation_id = sa.Column(sa.String, nullable=False)
    voice_id = sa.Column(sa.Integer, sa.ForeignKey("voices.id", ondelete="SET NULL"), nullable=True, index=True)
    # The person said "this is me": the turn is an anchor of its voice's profile.
    voice_confirmed = sa.Column(sa.Boolean, nullable=False, server_default=sa.false())
    # Similarity with the closest known voice; NULL when nothing was compared.
    similarity = sa.Column(sa.Float, nullable=True)
    # The speaker embedding of the whole turn, float32; NULL without voice models or with too little speech.
    embedding = sa.Column(sa.LargeBinary, nullable=True)
    speech_ms = sa.Column(sa.Integer, nullable=True)
    # Empty when a question was met with silence.
    text = sa.Column(sa.Text, nullable=False)
    answer = sa.Column(sa.Text, nullable=True)
    tools = sa.Column(sa.JSON, nullable=False)
    # Milliseconds from the start of the run to each stage.
    timings = sa.Column(sa.JSON, nullable=False)
    error = sa.Column(sa.Text, nullable=True)


class Reminder(Base):
    __tablename__ = "reminders"

    id = sa.Column(sa.String, primary_key=True)
    device = sa.Column(sa.String, nullable=False)
    due = sa.Column(UtcDateTime, nullable=False)
    prompt = sa.Column(sa.Text, nullable=False)
    created = sa.Column(UtcDateTime, nullable=False)
    schedule = sa.Column(sa.String, nullable=True)
    until = sa.Column(UtcDateTime, nullable=True)
    # The turn that set it; history retention keeps that turn while the reminder is pending.
    turn_id = sa.Column(sa.Integer, sa.ForeignKey("turns.id", ondelete="SET NULL"), nullable=True)
