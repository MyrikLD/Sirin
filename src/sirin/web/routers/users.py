from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from sirin.accounts import (
    Account,
    LastAdmin,
    Role,
    create_account,
    delete_account,
    list_accounts,
    login_taken,
    normalize_login,
    set_password,
    update_account,
)
from sirin.web.deps import SessionDep, require_admin

router = APIRouter(prefix="/api/users", dependencies=[Depends(require_admin)])


class NewAccount(BaseModel):
    login: str = Field(min_length=1, pattern=r"^\s*[A-Za-z0-9._-]+\s*$", description="Letters, digits, . _ -")
    name: str = Field(min_length=1, description="What the LLM calls the person")
    password: str
    role: Role = Role.MEMBER


class AccountChange(BaseModel):
    name: str | None = Field(default=None, min_length=1)
    role: Role | None = None
    password: str | None = Field(default=None, description="A new password; the person's sessions end")


@router.get("")
async def get_users(session: SessionDep) -> list[Account]:
    return await list_accounts(session)


@router.post("", status_code=201)
async def post_user(new: NewAccount, session: SessionDep) -> Account:
    if await login_taken(session, new.login):
        raise HTTPException(status.HTTP_409_CONFLICT, f"Login {normalize_login(new.login)!r} is taken")
    return await create_account(session, new.login, new.name, new.password, new.role)


@router.patch("/{user_id}")
async def patch_user(user_id: int, change: AccountChange, session: SessionDep) -> Account:
    try:
        account = await update_account(session, user_id, change.name, change.role)
    except LastAdmin as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    if account is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    if change.password is not None:
        await set_password(session, user_id, change.password)
    return account


@router.delete("/{user_id}", status_code=204)
async def delete_user(user_id: int, session: SessionDep) -> None:
    try:
        deleted = await delete_account(session, user_id)
    except LastAdmin as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
