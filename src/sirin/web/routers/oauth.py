from html import escape

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from mcp.client.auth import OAuthClientProvider
from pydantic import BaseModel, Field

from sirin.agent import list_server_tools
from sirin.config import McpServerConfig
from sirin.web.deps import RuntimeDep, require_admin

router = APIRouter()


class AuthorizeResponse(BaseModel):
    url: str | None = Field(description="Login page to open; None when the stored tokens already work")


@router.post("/api/oauth/authorize", dependencies=[Depends(require_admin)])
async def post_oauth_authorize(cfg: McpServerConfig, runtime: RuntimeDep, request: Request) -> AuthorizeResponse:
    oauth = runtime.oauth
    if not cfg.oauth:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "OAuth is off for this server")

    async def connect(provider: OAuthClientProvider) -> None:
        await list_server_tools(cfg, oauth, provider)

    try:
        # The callback address as the user's browser sees this service.
        redirect_uri = str(request.url_for("oauth_callback"))
        return AuthorizeResponse(url=await oauth.authorize(cfg, redirect_uri, connect))
    except Exception as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"{type(exc).__name__}: {exc}") from exc


# The OAuth provider sends the browser here; the one-time state ties it to a started login.
@router.get("/oauth/callback", response_class=HTMLResponse)
async def oauth_callback(
    runtime: RuntimeDep, state: str = "", code: str = "", iss: str | None = None, error: str | None = None
) -> str:
    if error:
        return f"<p>Authorization failed: {escape(error)}</p>"
    if not runtime.oauth.complete(state, code, iss):
        return "<p>This login is unknown or already finished. Start it again from the settings page.</p>"
    return "<p>Authorized. You can close this tab.</p>"
