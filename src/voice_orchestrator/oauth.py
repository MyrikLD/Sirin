import asyncio
import hashlib
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx2
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import (
    AuthorizationCodeResult,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthMetadata,
    OAuthToken,
)
from pydantic import AnyUrl

from voice_orchestrator.config import McpServerConfig
from voice_orchestrator.isolation import run_isolated

_log = logging.getLogger(__name__)

APP_NAME = "voice-orchestrator"

# How long a started authorization waits for the user to finish it in the browser.
AUTHORIZE_TIMEOUT_S = 600


class OAuthRequired(RuntimeError):
    def __init__(self, server: str) -> None:
        super().__init__(f"OAuth authorization required: press Authorize for {server!r}")


class FileTokenStorage(TokenStorage):
    """Tokens and the registered client of one MCP server, in one JSON file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    async def get_tokens(self) -> OAuthToken | None:
        data = self._read()
        if "tokens" not in data:
            return None
        tokens = OAuthToken.model_validate(data["tokens"])
        if (expires_at := data.get("expires_at")) is not None:
            tokens.expires_in = max(0, int(expires_at - time.time()))
        return tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        data = self._read()
        data["tokens"] = tokens.model_dump(mode="json", exclude_none=True)
        data["expires_at"] = time.time() + tokens.expires_in if tokens.expires_in is not None else None
        self._write(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        data = self._read()
        return OAuthClientInformationFull.model_validate(data["client"]) if "client" in data else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        data = self._read()
        data["client"] = client_info.model_dump(mode="json", exclude_none=True)
        self._write(data)

    def has_tokens(self) -> bool:
        return "tokens" in self._read()

    def redirect_uris(self) -> list[str]:
        return [str(u) for u in self._read().get("client", {}).get("redirect_uris", [])]

    def client_id(self) -> str | None:
        return self._read().get("client", {}).get("client_id")

    def reset(self) -> None:
        self._path.unlink(missing_ok=True)

    def server_metadata(self) -> dict | None:
        return self._read().get("server_metadata")

    def set_server_metadata(self, metadata: dict) -> None:
        data = self._read()
        data["server_metadata"] = metadata
        self._write(data)

    def _read(self) -> dict:
        return json.loads(self._path.read_text()) if self._path.exists() else {}

    def _write(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.chmod(0o600)
        os.replace(tmp, self._path)


class _Provider(OAuthClientProvider):
    async def _initialize(self) -> None:
        await super()._initialize()
        # The SDK treats loaded tokens as never expiring, so an expired one would get a 401 and a new
        # browser login instead of a refresh. Stored tokens carry their remaining lifetime.
        if self.context.current_tokens is not None:
            self.context.update_token_expiry(self.context.current_tokens)
        # After a restart the SDK knows no server metadata and refreshes at <origin>/token, which is
        # wrong for servers like Home Assistant (/auth/token); the refresh fails and asks for a login.
        if self.context.oauth_metadata is None and isinstance(self.context.storage, FileTokenStorage):
            metadata = self.context.storage.server_metadata()
            if metadata is None:
                metadata = await _authorization_server_metadata(self.context.server_url)
                if metadata is not None:
                    self.context.storage.set_server_metadata(metadata)
            if metadata is not None:
                self.context.oauth_metadata = OAuthMetadata.model_validate(metadata)


class _Flow:
    def __init__(self) -> None:
        loop = asyncio.get_running_loop()
        self.url: asyncio.Future[str] = loop.create_future()
        self.code: asyncio.Future[AuthorizationCodeResult] = loop.create_future()


class OAuthManager:
    """OAuth for HTTP MCP servers on a headless host: the login runs in the settings page's browser.

    The redirect URI is the address that browser used to open the page, so the provider sends it
    back to a URL it can reach. Connections made by the running agent never start a login; they
    only refresh tokens and fail with OAuthRequired until the user authorizes from the page.
    """

    def __init__(self, data_dir: Path, on_authorized: Callable[[str], Awaitable[None]]) -> None:
        self._dir = data_dir / "oauth"
        self._on_authorized = on_authorized
        self._flows: dict[str, _Flow] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def provider(self, server: McpServerConfig) -> OAuthClientProvider:
        if not self.is_authorized(server):
            raise OAuthRequired(server.name)

        async def refuse(_: str) -> None:
            raise OAuthRequired(server.name)

        async def never() -> AuthorizationCodeResult:
            raise OAuthRequired(server.name)

        redirect_uris = self._storage(server).redirect_uris()
        if not redirect_uris:
            raise OAuthRequired(server.name)
        return self._provider(server, redirect_uris[0], refuse, never)

    def is_authorized(self, server: McpServerConfig) -> bool:
        return self._storage(server).has_tokens()

    async def authorize(
        self,
        server: McpServerConfig,
        redirect_uri: str,
        connect: Callable[[OAuthClientProvider], Awaitable[None]],
    ) -> str | None:
        """Starts a login and returns the URL to open, or None if the stored tokens already work."""
        storage = self._storage(server)
        stale_address = storage.redirect_uris() and redirect_uri not in storage.redirect_uris()
        stale_url_id = (storage.client_id() or "").startswith(_origin(redirect_uri)) and (
            storage.client_id() != _url_client_id(redirect_uri)
        )
        if stale_address or stale_url_id:
            # The client was set up for another address of this service, or under an older URL client id.
            storage.reset()
        await self._preregister(server, redirect_uri)
        flow = _Flow()

        async def redirect(url: str) -> None:
            state = parse_qs(urlparse(url).query).get("state", [""])[0]
            self._flows[state] = flow
            flow.url.set_result(url)

        async def callback() -> AuthorizationCodeResult:
            return await flow.code

        provider = self._provider(server, redirect_uri, redirect, callback)

        async def login() -> None:
            try:
                await asyncio.wait_for(run_isolated(connect(provider)), AUTHORIZE_TIMEOUT_S)
            except Exception as exc:  # noqa: BLE001
                if not flow.url.done():
                    flow.url.set_exception(exc)
                else:
                    _log.error("OAuth for MCP server %s failed: %s", server.name, exc)
                return
            finally:
                self._flows = {k: v for k, v in self._flows.items() if v is not flow}
            if not flow.url.done():
                flow.url.set_result("")
            _log.info("OAuth for MCP server %s is done", server.name)
            await self._on_authorized(server.name)

        task = asyncio.create_task(login())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return await flow.url or None

    def complete(self, state: str, code: str, iss: str | None) -> bool:
        flow = self._flows.get(state)
        if flow is None or flow.code.done():
            return False
        flow.code.set_result(AuthorizationCodeResult(code=code, state=state, iss=iss))
        return True

    def _provider(
        self,
        server: McpServerConfig,
        redirect_uri: str,
        redirect: Callable[[str], Awaitable[None]],
        callback: Callable[[], Awaitable[AuthorizationCodeResult]],
    ) -> OAuthClientProvider:
        metadata = OAuthClientMetadata(
            client_name=APP_NAME,
            redirect_uris=[AnyUrl(redirect_uri)],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
        )
        return _Provider(
            server_url=server.url,
            client_metadata=metadata,
            storage=self._storage(server),
            redirect_handler=redirect,
            callback_handler=callback,
        )

    async def _preregister(self, server: McpServerConfig, redirect_uri: str) -> None:
        """Stores a URL client id when the server has no client registration endpoint.

        This is IndieAuth, as Home Assistant does it: the client id is our origin, accepted because
        the redirect URI shares its scheme and host. The SDK does this by itself only for HTTPS
        client ids, so a plain-HTTP orchestrator would otherwise POST to a /register that does not exist.
        """
        storage = self._storage(server)
        if await storage.get_client_info() is not None:
            return
        metadata = await _authorization_server_metadata(server.url)
        if metadata is not None:
            storage.set_server_metadata(metadata)
        if metadata is None or metadata.get("registration_endpoint"):
            return
        client_id = _url_client_id(redirect_uri)
        await storage.set_client_info(
            OAuthClientInformationFull(
                client_id=client_id,
                redirect_uris=[AnyUrl(redirect_uri)],
                token_endpoint_auth_method="none",
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
            )
        )
        _log.info("MCP server %s has no client registration; using client id %s", server.name, client_id)

    def _storage(self, server: McpServerConfig) -> FileTokenStorage:
        # Keyed by name and URL, so pointing a server at another URL starts from a clean login.
        key = hashlib.sha256(f"{server.name}\n{server.url}".encode()).hexdigest()[:16]
        return FileTokenStorage(self._dir / f"{key}.json")


def _origin(url: str) -> str:
    parts = urlparse(url)
    return f"{parts.scheme}://{parts.netloc}"


def _url_client_id(redirect_uri: str) -> str:
    # The authorization server shows this URL on its consent page, so it carries the app name.
    return f"{_origin(redirect_uri)}/{APP_NAME}"


async def _authorization_server_metadata(server_url: str) -> dict | None:
    """RFC 9728 resource metadata, then RFC 8414 server metadata; None when neither is published."""
    url = urlparse(server_url)
    origin = f"{url.scheme}://{url.netloc}"
    async with httpx2.AsyncClient(timeout=10) as client:
        issuer = origin
        for path in (f"/.well-known/oauth-protected-resource{url.path}", "/.well-known/oauth-protected-resource"):
            response = await client.get(origin + path)
            if response.status_code == 200:
                issuer = (response.json().get("authorization_servers") or [origin])[0].rstrip("/")
                break
        issuer_url = urlparse(issuer)
        well_known = f"{issuer_url.scheme}://{issuer_url.netloc}/.well-known/oauth-authorization-server"
        response = await client.get(well_known + issuer_url.path.rstrip("/"))
        return response.json() if response.status_code == 200 else None
