import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

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
    if app_settings.public_url is None:
        return
    url = urlparse(app_settings.public_url)
    port = url.port or {"http": 80, "https": 443}.get(url.scheme)
    if port != app_settings.port:
        _log.warning(
            "VO_PUBLIC_URL %s points to port %s, but the server listens on %s: the satellite reaches it "
            "only through a proxy on that port",
            app_settings.public_url,
            port,
            app_settings.port,
        )


def create_app(app_settings: AppSettings) -> FastAPI:
    clips = ClipStore(app_settings.public_url, app_settings.port, app_settings.audio_ttl_s)
    store = SettingsStore(app_settings.data_dir)

    async def on_authorized(server_name: str) -> None:
        if any(m.name == server_name for m in runtime.settings.mcp_servers):
            await runtime.apply(runtime.settings)

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
