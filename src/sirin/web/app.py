import ipaddress
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles

from sirin.config import AppSettings
from sirin.runtime import Runtime
from sirin.web.deps import require_admin
from sirin.web.routers import auth, oauth, probes, settings, status, tts, users

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

_log = logging.getLogger(__name__)


def _check_public_url(app_settings: AppSettings) -> None:
    url = app_settings.public_url
    if url is None:
        return
    if url.host == "localhost" or _is_loopback_ip(url.host or ""):
        _log.warning("SIRIN_PUBLIC_URL %s is a loopback address: satellites cannot download answers from it", url)
    if url.port != app_settings.port:
        _log.warning(
            "SIRIN_PUBLIC_URL %s points to port %s, but the server listens on %s: satellites reach it "
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
    runtime = Runtime(app_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        _check_public_url(app_settings)
        await runtime.start()
        yield
        await runtime.stop()

    app = FastAPI(title="sirin", lifespan=lifespan)
    app.state.runtime = runtime

    # Open: the satellite downloads answers without credentials, and the OAuth provider sends the browser back.
    for module in (tts, oauth, auth, users):
        app.include_router(module.router)
    for module in (status, settings, probes):
        app.include_router(module.router, dependencies=[Depends(require_admin)])
    # Mounted last: a mount at "/" would otherwise shadow the routes above.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
