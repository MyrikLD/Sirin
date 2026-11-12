from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from sirin.accounts import SESSION_TTL, Account, check_password, log_in, log_out, set_password
from sirin.web.deps import SESSION_COOKIE, AccountDep, SessionDep

router = APIRouter(prefix="/api/auth")


class Credentials(BaseModel):
    login: str
    password: str


class PasswordChange(BaseModel):
    current: str = Field(description="The password now")
    new: str = Field(description="The new password")


@router.post("/login", status_code=204)
async def post_login(credentials: Credentials, session: SessionDep, request: Request, response: Response) -> None:
    token = await log_in(session, credentials.login, credentials.password)
    if token is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong login or password")
    # Lax keeps the cookie off cross-site POST and PUT requests, so another site cannot act with it.
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=int(SESSION_TTL.total_seconds()),
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )


@router.post("/logout", status_code=204)
async def post_logout(session: SessionDep, request: Request, response: Response) -> None:
    if token := request.cookies.get(SESSION_COOKIE):
        await log_out(session, token)
    response.delete_cookie(SESSION_COOKIE)


@router.get("/me")
async def get_me(account: AccountDep) -> Account:
    return account


@router.post("/password", status_code=204)
async def post_password(change: PasswordChange, account: AccountDep, session: SessionDep, request: Request) -> None:
    """Changes the own password; other sessions of the account end."""
    if not await check_password(session, account.id, change.current):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The current password is wrong")
    await set_password(session, account.id, change.new, keep_session=request.cookies.get(SESSION_COOKIE))
