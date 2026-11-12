from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from mcp.client.auth import OAuthClientProvider
from pydantic import BaseModel, Field

from sirin.agent import list_server_tools
from sirin.config import McpServerConfig
from sirin.personal_servers import (
    NameTaken,
    all_personal,
    create_personal,
    delete_personal,
    get_personal,
    owners,
    personal_of,
    update_personal,
)
from sirin.probes import ProbeResult, probe_mcp
from sirin.status import ItemStatus
from sirin.web.deps import AccountDep, RuntimeDep, SessionDep, current_account, require_admin
from sirin.web.routers.oauth import AuthorizeResponse

router = APIRouter(dependencies=[Depends(current_account)])


class MyServer(McpServerConfig):
    id: int
    status: ItemStatus


class PersonalOverview(BaseModel):
    id: int
    owner: str = Field(description="The person's name")
    name: str
    status: ItemStatus


def _reconnect(runtime: RuntimeDep, background: BackgroundTasks) -> None:
    """The agent connects the changed servers once this request is committed."""
    background.add_task(runtime.reconnect_tools)


@router.get("/api/my/mcp-servers")
async def get_my_servers(account: AccountDep, session: SessionDep, runtime: RuntimeDep) -> list[MyServer]:
    return [
        MyServer(id=s.id, status=runtime.personal_status(s.id), **s.cfg.model_dump())
        for s in await personal_of(session, account.id)
    ]


@router.post("/api/my/mcp-servers", status_code=201)
async def post_my_server(
    cfg: McpServerConfig, account: AccountDep, session: SessionDep, runtime: RuntimeDep, background: BackgroundTasks
) -> MyServer:
    try:
        server_id = await create_personal(session, account.id, cfg)
    except NameTaken as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    _reconnect(runtime, background)
    return MyServer(id=server_id, status=ItemStatus(state="pending", detail="connecting"), **cfg.model_dump())


@router.put("/api/my/mcp-servers/{server_id}", status_code=204)
async def put_my_server(
    server_id: int,
    cfg: McpServerConfig,
    account: AccountDep,
    session: SessionDep,
    runtime: RuntimeDep,
    background: BackgroundTasks,
) -> None:
    try:
        found = await update_personal(session, account.id, server_id, cfg)
    except NameTaken as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    if not found:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    _reconnect(runtime, background)


@router.delete("/api/my/mcp-servers/{server_id}", status_code=204)
async def delete_my_server(
    server_id: int, account: AccountDep, session: SessionDep, runtime: RuntimeDep, background: BackgroundTasks
) -> None:
    if not await delete_personal(session, account.id, server_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    _reconnect(runtime, background)


@router.post("/api/my/mcp-servers/probe")
async def post_my_probe(cfg: McpServerConfig, account: AccountDep, runtime: RuntimeDep) -> ProbeResult:
    """Tests unsaved values, with the stored tokens of the person's server of that name and URL."""
    return await probe_mcp(cfg, runtime.oauth, account.id)


@router.post("/api/my/mcp-servers/{server_id}/authorize")
async def post_my_authorize(
    server_id: int, account: AccountDep, session: SessionDep, runtime: RuntimeDep, request: Request
) -> AuthorizeResponse:
    server = await get_personal(session, account.id, server_id)
    if server is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    if not server.cfg.oauth:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "OAuth is off for this server")

    async def connect(provider: OAuthClientProvider) -> None:
        await list_server_tools(server.cfg, runtime.oauth, provider, owner_id=account.id)

    try:
        redirect_uri = str(request.url_for("oauth_callback"))
        url = await runtime.oauth.authorize(server.cfg, redirect_uri, connect, account.id)
    except Exception as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"{type(exc).__name__}: {exc}") from exc
    return AuthorizeResponse(url=url)


@router.get("/api/mcp-servers/personal", dependencies=[Depends(require_admin)])
async def get_personal_overview(session: SessionDep, runtime: RuntimeDep) -> list[PersonalOverview]:
    """Everyone's personal servers by name and state; their URLs and headers stay with their owners."""
    names = await owners(session)
    return [
        PersonalOverview(
            id=s.id, owner=names.get(s.owner_id, "?"), name=s.cfg.name, status=runtime.personal_status(s.id)
        )
        for s in await all_personal(session)
    ]
