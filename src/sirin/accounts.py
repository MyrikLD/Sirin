import asyncio
import base64
import functools
import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta
from enum import StrEnum

import sqlalchemy as sa
from pydantic import AwareDatetime, BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from sirin.tables import LoginSession, User

SESSION_TTL = timedelta(days=30)
FIRST_ADMIN_LOGIN = "admin"

# scrypt cost: about 50 ms and 16 MB per hash.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**14, 8, 1


class Role(StrEnum):
    ADMIN = "admin"
    MEMBER = "member"


class Account(BaseModel):
    id: int
    login: str
    name: str = Field(description="What the LLM calls the person; need not be unique")
    role: Role
    created_at: AwareDatetime


class LastAdmin(ValueError):
    def __init__(self) -> None:
        super().__init__("at least one admin must remain")


_ACCOUNT_COLUMNS = [User.id, User.login, User.name, User.role, User.created_at]


def normalize_login(login: str) -> str:
    return login.strip().lower()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    scheme, n, r, p, salt, digest = stored.split("$")
    if scheme != "scrypt":
        raise ValueError(f"unknown password hash scheme {scheme!r}")
    expected = base64.b64decode(digest)
    actual = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p))
    return hmac.compare_digest(actual, expected)


@functools.cache
def _dummy_hash() -> str:
    """Compared against when the login is unknown, so a wrong login takes as long as a wrong password."""
    return hash_password(secrets.token_urlsafe())


async def list_accounts(session: AsyncSession) -> list[Account]:
    rows = await session.execute(sa.select(*_ACCOUNT_COLUMNS).order_by(User.id))
    return [Account.model_validate(dict(r)) for r in rows.mappings()]


async def get_account(session: AsyncSession, user_id: int) -> Account | None:
    row = (await session.execute(sa.select(*_ACCOUNT_COLUMNS).where(User.id == user_id))).mappings().one_or_none()
    return Account.model_validate(dict(row)) if row is not None else None


async def login_taken(session: AsyncSession, login: str) -> bool:
    return await session.scalar(sa.select(User.id).where(User.login == normalize_login(login))) is not None


async def create_account(session: AsyncSession, login: str, name: str, password: str, role: Role) -> Account:
    password_hash = await asyncio.to_thread(hash_password, password)
    stmt = (
        sa.insert(User)
        .values(
            login=normalize_login(login),
            name=name.strip(),
            password_hash=password_hash,
            role=role,
            created_at=datetime.now(UTC),
        )
        .returning(*_ACCOUNT_COLUMNS)
    )
    return Account.model_validate(dict((await session.execute(stmt)).mappings().one()))


async def update_account(
    session: AsyncSession, user_id: int, name: str | None = None, role: Role | None = None
) -> Account | None:
    values: dict[str, object] = {}
    if name is not None:
        values["name"] = name.strip()
    if role is not None:
        if role != Role.ADMIN:
            await _keep_an_admin(session, user_id)
        values["role"] = role
    if values:
        await session.execute(sa.update(User).where(User.id == user_id).values(**values))
    return await get_account(session, user_id)


async def delete_account(session: AsyncSession, user_id: int) -> bool:
    """Its sessions go with it."""
    await _keep_an_admin(session, user_id)
    deleted = await session.scalar(sa.delete(User).where(User.id == user_id).returning(User.id))
    return deleted is not None


async def set_password(session: AsyncSession, user_id: int, password: str, keep_session: str | None = None) -> None:
    """Ends every session of the account except keep_session (a token), so an old password logs nobody in."""
    password_hash = await asyncio.to_thread(hash_password, password)
    await session.execute(sa.update(User).where(User.id == user_id).values(password_hash=password_hash))
    stmt = sa.delete(LoginSession).where(LoginSession.user_id == user_id)
    if keep_session is not None:
        stmt = stmt.where(LoginSession.id != _token_hash(keep_session))
    await session.execute(stmt)


async def check_password(session: AsyncSession, user_id: int, password: str) -> bool:
    stored = await session.scalar(sa.select(User.password_hash).where(User.id == user_id))
    return stored is not None and await asyncio.to_thread(verify_password, password, stored)


async def log_in(session: AsyncSession, login: str, password: str) -> str | None:
    """A new session token, or None when the login or the password is wrong."""
    row = (
        await session.execute(sa.select(User.id, User.password_hash).where(User.login == normalize_login(login)))
    ).one_or_none()
    valid = await asyncio.to_thread(verify_password, password, row.password_hash if row else _dummy_hash())
    if row is None or not valid:
        return None
    now = datetime.now(UTC)
    await session.execute(sa.delete(LoginSession).where(LoginSession.expires_at <= now))
    token = secrets.token_urlsafe(32)
    await session.execute(
        sa.insert(LoginSession).values(
            id=_token_hash(token), user_id=row.id, created_at=now, expires_at=now + SESSION_TTL
        )
    )
    return token


async def log_out(session: AsyncSession, token: str) -> None:
    await session.execute(sa.delete(LoginSession).where(LoginSession.id == _token_hash(token)))


async def session_account(session: AsyncSession, token: str) -> Account | None:
    row = (
        (
            await session.execute(
                sa.select(*_ACCOUNT_COLUMNS)
                .join(LoginSession, LoginSession.user_id == User.id)
                .where(LoginSession.id == _token_hash(token), LoginSession.expires_at > datetime.now(UTC))
            )
        )
        .mappings()
        .one_or_none()
    )
    return Account.model_validate(dict(row)) if row is not None else None


async def ensure_admin(session: AsyncSession) -> str | None:
    """Creates the first admin when there are no accounts; returns its generated password."""
    if await session.scalar(sa.select(sa.func.count()).select_from(User)):
        return None
    password = secrets.token_urlsafe(12)
    await create_account(session, FIRST_ADMIN_LOGIN, "Admin", password, Role.ADMIN)
    return password


async def reset_password(session: AsyncSession, login: str) -> str | None:
    """Sets a generated password for login and ends its sessions; None when there is no such account."""
    user_id = await session.scalar(sa.select(User.id).where(User.login == normalize_login(login)))
    if user_id is None:
        return None
    password = secrets.token_urlsafe(12)
    await set_password(session, user_id, password)
    return password


async def _keep_an_admin(session: AsyncSession, user_id: int) -> None:
    """Refuses to remove the admin role from user_id when it is the last admin."""
    others = await session.scalar(
        sa.select(sa.func.count()).select_from(User).where(User.role == Role.ADMIN, User.id != user_id)
    )
    is_admin = await session.scalar(sa.select(User.role).where(User.id == user_id)) == Role.ADMIN
    if is_admin and not others:
        raise LastAdmin


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
