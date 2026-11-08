import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx2
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    build_protected_resource_metadata_discovery_urls,
    create_oauth_metadata_request,
    handle_auth_metadata_response,
    handle_protected_resource_response,
)
from mcp.shared.auth import (
    AuthorizationCodeResult,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthMetadata,
    OAuthToken,
)
from pydantic import AnyUrl

from voice_orchestrator.config import McpServerConfig
from voice_orchestrator.files import write_private

_log = logging.getLogger(__name__)

APP_NAME = "voice-orchestrator"

# How long a started authorization waits for the user to finish it in the browser.
AUTHORIZE_TIMEOUT_S = 600


class OAuthRequired(RuntimeError):
    def __init__(self, server: str) -> None:
        super().__init__(f"OAuth authorization required: press Authorize for {server!r}")


class FileTokenStorage(TokenStorage):
    """Tokens, the registered client and the server metadata of one MCP server, in one JSON file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    async def get_tokens(self) -> OAuthToken | None:
        data = self.read()
        if "tokens" not in data:
            return None
        tokens = OAuthToken.model_validate(data["tokens"])
        if (expires_at := data.get("expires_at")) is not None:
            tokens.expires_in = max(0, int(expires_at - time.time()))
        return tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        data = self.read()
        data["tokens"] = tokens.model_dump(mode="json", exclude_none=True)
        data["expires_at"] = time.time() + tokens.expires_in if tokens.expires_in is not None else None
        self._write(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        data = self.read()
        return OAuthClientInformationFull.model_validate(data["client"]) if "client" in data else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        data = self.read()
        data["client"] = client_info.model_dump(mode="json", exclude_none=True)
        self._write(data)

    async def server_metadata(self, server_url: str) -> OAuthMetadata | None:
        """The stored authorization server metadata, discovered and stored on first use."""
        data = self.read()
        if "server_metadata" in data:
            return OAuthMetadata.model_validate(data["server_metadata"])
        metadata = await _discover_server_metadata(server_url)
        if metadata is not None:
            data["server_metadata"] = metadata.model_dump(mode="json", exclude_none=True)
            self._write(data)
        return metadata

    def read(self) -> dict:
        return json.loads(self._path.read_text()) if self._path.exists() else {}

    def reset(self) -> None:
        self._path.unlink(missing_ok=True)

    def _write(self, data: dict) -> None:
        write_private(self._path, json.dumps(data, indent=2))


class _Provider(OAuthClientProvider):
    def __init__(self, *, storage: FileTokenStorage, **kwargs) -> None:
        super().__init__(storage=storage, **kwargs)
        self._file_storage = storage

    async def _initialize(self) -> None:
        await super()._initialize()
        # The SDK treats loaded tokens as never expiring, so an expired one would get a 401 and a new
        # browser login instead of a refresh. Stored tokens carry their remaining lifetime.
        if self.context.current_tokens is not None:
            self.context.update_token_expiry(self.context.current_tokens)
        # After a restart the SDK knows no server metadata and refreshes at <origin>/token, which is
        # wrong for servers like Home Assistant (/auth/token); the refresh fails and asks for a login.
        if self.context.oauth_metadata is None:
            self.context.oauth_metadata = await self._file_storage.server_metadata(self.context.server_url)


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
        data = self._storage(server).read()
        redirect_uris = data.get("client", {}).get("redirect_uris")
        if "tokens" not in data or not redirect_uris:
            raise OAuthRequired(server.name)

        async def refuse(_: str) -> None:
            raise OAuthRequired(server.name)

        async def never() -> AuthorizationCodeResult:
            raise OAuthRequired(server.name)

        return self._provider(server, str(redirect_uris[0]), refuse, never)

    async def authorize(
        self,
        server: McpServerConfig,
        redirect_uri: str,
        connect: Callable[[OAuthClientProvider], Awaitable[object]],
    ) -> str | None:
        """Starts a login and returns the URL to open, or None if the stored tokens already work.

        connect opens the server with the given provider; it must run outside the caller's task.
        """
        storage = self._storage(server)
        client = storage.read().get("client", {})
        stale_address = client.get("redirect_uris") and redirect_uri not in client["redirect_uris"]
        stale_url_id = client.get("client_id", "").startswith(_origin(redirect_uri)) and (
            client["client_id"] != _url_client_id(redirect_uri)
        )
        if stale_address or stale_url_id:
            # The client was set up for another address of this service, or with a URL client id that lacks the
            # app name; the provider would reject the first and show the second on its consent page.
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
                await asyncio.wait_for(connect(provider), AUTHORIZE_TIMEOUT_S)
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
        metadata = await storage.server_metadata(server.url)
        if metadata is None or metadata.registration_endpoint:
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


async def _discover_server_metadata(server_url: str) -> OAuthMetadata | None:
    """RFC 9728 resource metadata, then RFC 8414 / OpenID server metadata, as the SDK discovers them."""
    async with httpx2.AsyncClient(timeout=10) as client:
        issuer = None
        for url in build_protected_resource_metadata_discovery_urls(None, server_url):
            resource = await handle_protected_resource_response(await client.send(create_oauth_metadata_request(url)))
            if resource is not None:
                issuer = str(resource.authorization_servers[0]) if resource.authorization_servers else None
                break
        for url in build_oauth_authorization_server_metadata_discovery_urls(issuer, server_url):
            keep_trying, metadata = await handle_auth_metadata_response(
                await client.send(create_oauth_metadata_request(url))
            )
            if metadata is not None or not keep_trying:
                return metadata
    return None
