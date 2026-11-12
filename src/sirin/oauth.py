import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse

import httpx2
import sqlalchemy as sa
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
from sqlalchemy.dialects.sqlite import insert

from sirin.config import McpServerConfig
from sirin.database import Database
from sirin.tables import McpServer, McpToken

_log = logging.getLogger(__name__)

APP_NAME = "sirin"

# How long a started authorization waits for the user to finish it in the browser.
AUTHORIZE_TIMEOUT_S = 600


class OAuthRequired(RuntimeError):
    def __init__(self, server: str) -> None:
        super().__init__(f"OAuth authorization required: press Authorize for {server!r}")


class ServerNotSaved(RuntimeError):
    def __init__(self, server: str) -> None:
        super().__init__(f"MCP server {server!r} is not saved with this URL: save the settings first")


class DbTokenStorage(TokenStorage):
    """The `mcp_tokens` row of one shared MCP server, found by its name and URL.

    A server not saved under that name and URL has no stored state; storing for it fails.
    """

    def __init__(self, db: Database, server: McpServerConfig) -> None:
        self._db = db
        self._server = server

    async def get_tokens(self) -> OAuthToken | None:
        row = await self.read()
        if row is None or row.tokens is None:
            return None
        tokens = OAuthToken.model_validate(row.tokens)
        if row.expires_at is not None:
            tokens.expires_in = max(0, int((row.expires_at - datetime.now(UTC)).total_seconds()))
        return tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        expires_at = datetime.now(UTC) + timedelta(seconds=tokens.expires_in) if tokens.expires_in is not None else None
        await self._write(tokens=tokens.model_dump(mode="json", exclude_none=True), expires_at=expires_at)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        row = await self.read()
        if row is None or row.client_info is None:
            return None
        return OAuthClientInformationFull.model_validate(row.client_info)

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        await self._write(client_info=client_info.model_dump(mode="json", exclude_none=True))

    async def server_metadata(self, server_url: str) -> OAuthMetadata | None:
        """The stored authorization server metadata, discovered and stored on first use."""
        row = await self.read()
        if row is not None and row.server_metadata is not None:
            return OAuthMetadata.model_validate(row.server_metadata)
        metadata = await _discover_server_metadata(server_url)
        if metadata is not None:
            await self._write(server_metadata=metadata.model_dump(mode="json", exclude_none=True))
        return metadata

    async def read(self) -> sa.Row | None:
        async with self._db.session() as session:
            rows = await session.execute(
                sa.select(McpToken.tokens, McpToken.expires_at, McpToken.client_info, McpToken.server_metadata)
                .join(McpServer, McpServer.id == McpToken.server_id)
                .where(*self._server_is())
            )
            return rows.one_or_none()

    async def server_id(self) -> int | None:
        async with self._db.session() as session:
            return await session.scalar(sa.select(McpServer.id).where(*self._server_is()))

    async def reset(self) -> None:
        async with self._db.session() as session, session.begin():
            server_id = sa.select(McpServer.id).where(*self._server_is()).scalar_subquery()
            await session.execute(sa.delete(McpToken).where(McpToken.server_id == server_id))

    async def _write(self, **values: object) -> None:
        if (server_id := await self.server_id()) is None:
            raise ServerNotSaved(self._server.name)
        async with self._db.session() as session, session.begin():
            stmt = insert(McpToken).values(server_id=server_id, **values)
            await session.execute(stmt.on_conflict_do_update(index_elements=[McpToken.server_id], set_=values))

    def _server_is(self) -> list[sa.ColumnElement[bool]]:
        # Name and URL: a server pointed at another URL has its tokens dropped when saved, and until then
        # its unsaved URL must not reach the tokens of the old one.
        return [McpServer.owner_id.is_(None), McpServer.name == self._server.name, McpServer.url == self._server.url]


class _Provider(OAuthClientProvider):
    def __init__(self, *, storage: DbTokenStorage, **kwargs) -> None:
        super().__init__(storage=storage, **kwargs)
        self._stored = storage

    async def _initialize(self) -> None:
        await super()._initialize()
        # The SDK treats loaded tokens as never expiring, so an expired one would get a 401 and a new
        # browser login instead of a refresh. Stored tokens carry their remaining lifetime.
        if self.context.current_tokens is not None:
            self.context.update_token_expiry(self.context.current_tokens)
        # After a restart the SDK knows no server metadata and refreshes at <origin>/token, which is
        # wrong for servers like Home Assistant (/auth/token); the refresh fails and asks for a login.
        if self.context.oauth_metadata is None:
            self.context.oauth_metadata = await self._stored.server_metadata(self.context.server_url)


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

    def __init__(self, db: Database, on_authorized: Callable[[str], Awaitable[None]]) -> None:
        self._db = db
        self._on_authorized = on_authorized
        self._flows: dict[str, _Flow] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    async def provider(self, server: McpServerConfig) -> OAuthClientProvider:
        row = await self._storage(server).read()
        redirect_uris = (row.client_info or {}).get("redirect_uris") if row is not None else None
        if row is None or row.tokens is None or not redirect_uris:
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
        row = await storage.read()
        if row is None and await storage.server_id() is None:
            raise ServerNotSaved(server.name)
        registered = (row.client_info or {}).get("redirect_uris") if row is not None else None
        if registered and redirect_uri not in registered:
            # The client was set up for another address of this service; the provider would reject it.
            await storage.reset()
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
            except Exception as exc:
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
        client ids, so a plain-HTTP Sirin would otherwise POST to a /register that does not exist.
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

    def _storage(self, server: McpServerConfig) -> DbTokenStorage:
        return DbTokenStorage(self._db, server)


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
