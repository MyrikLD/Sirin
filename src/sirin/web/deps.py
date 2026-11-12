from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from sirin.accounts import Account, Role, session_account
from sirin.runtime import Runtime

SESSION_COOKIE = "sirin_session"


def get_runtime(request: Request) -> Runtime:
    return request.app.state.runtime


RuntimeDep = Annotated[Runtime, Depends(get_runtime)]


async def session_dep(runtime: RuntimeDep) -> AsyncIterator[AsyncSession]:
    """One transaction per request, committed before the response is sent."""
    async with runtime.db.session() as session, session.begin():
        yield session


SessionDep = Annotated[AsyncSession, Depends(session_dep, scope="function")]


async def current_account(request: Request, session: SessionDep) -> Account:
    token = request.cookies.get(SESSION_COOKIE)
    account = await session_account(session, token) if token else None
    if account is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Log in first")
    return account


AccountDep = Annotated[Account, Depends(current_account)]


async def require_admin(account: AccountDep) -> Account:
    if account.role != Role.ADMIN:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only an admin can do this")
    return account


AdminDep = Annotated[Account, Depends(require_admin)]
