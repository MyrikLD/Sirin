import ipaddress
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from voice_orchestrator.clips import ClipStore
from voice_orchestrator.config import AppSettings, SettingsStore
from voice_orchestrator.oauth import OAuthManager
from voice_orchestrator.runtime import Runtime
from voice_orchestrator.web.routers import health, oauth, probes, settings, tts

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

_log = logging.getLogger(__name__)


def _check_public_url(app_settings: AppSettings) -> None:
    url = app_settings.public_url
    if url is None:
        return
    if url.host == "localhost" or _is_loopback_ip(url.host or ""):
        _log.warning("VO_PUBLIC_URL %s is a loopback address: satellites cannot download answers from it", url)
    if url.port != app_settings.port:
        _log.warning(
            "VO_PUBLIC_URL %s points to port %s, but the server listens on %s: satellites reach it "
            "only through a proxy on that port",
            url,
            url.port,
            app_settings.port,
        )


def _is_loopback_ip(host: str) -> bool:
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def create_app(app_settings: AppSettings) -> FastAPI:
    public_url = str(app_settings.public_url) if app_settings.public_url else None
    clips = ClipStore(public_url, app_settings.port, app_settings.audio_ttl_s)
    store = SettingsStore(app_settings.data_dir)

    async def on_authorized(server_name: str) -> None:
        if any(m.name == server_name for m in runtime.settings.mcp_servers):
            await runtime.reconnect_tools()

    oauth_manager = OAuthManager(app_settings.data_dir, on_authorized)
    runtime = Runtime(clips, oauth_manager)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        _check_public_url(app_settings)
        await runtime.apply(store.load())
        yield
        await runtime.stop()

    app = FastAPI(title="voice-orchestrator", lifespan=lifespan)
    app.state.clips = clips
    app.state.store = store
    app.state.runtime = runtime
    app.state.oauth = oauth_manager

    for module in (health, tts, settings, probes, oauth):
        app.include_router(module.router)
    # Mounted last: a mount at "/" would otherwise shadow the routes above.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
