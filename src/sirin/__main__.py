import argparse
import asyncio
import contextlib
import logging
import os
from types import FrameType

import uvicorn
from pydantic import ValidationError

from sirin.accounts import reset_password
from sirin.config import AppSettings
from sirin.database import Database
from sirin.runtime import Runtime
from sirin.web import create_app


class _Server(uvicorn.Server):
    """Closes the status and history streams on the exit signal; otherwise shutdown would wait for every open page."""

    def __init__(self, config: uvicorn.Config, runtime: Runtime) -> None:
        super().__init__(config)
        self._runtime = runtime

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        super().handle_exit(sig, frame)
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(self._runtime.board.close)
        loop.call_soon_threadsafe(self._runtime.history.close)


async def _reset_password(settings: AppSettings, login: str) -> None:
    db = Database.in_dir(settings.data_dir)
    try:
        await db.migrate()
        async with db.session() as session, session.begin():
            password = await reset_password(session, login)
    finally:
        await db.close()
    if password is None:
        raise SystemExit(f"There is no account {login!r}")
    print(f"New password for {login!r}: {password}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice pipeline for ESPHome satellites")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log protocol events")
    parser.add_argument(
        "--reset-password", metavar="LOGIN", help="Set a new generated password for LOGIN, print it and exit"
    )
    args = parser.parse_args()
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:
        logging.getLogger("aioesphomeapi").setLevel(logging.WARNING)

    try:
        settings = AppSettings()
    except ValidationError as exc:
        raise SystemExit(f"Bad environment (SIRIN_* variables or .env):\n{exc}") from exc
    if args.reset_password is not None:
        asyncio.run(_reset_password(settings, args.reset_password))
        return
    app = create_app(settings)
    config = uvicorn.Config(app, host=settings.host, port=settings.port, log_config=None, timeout_graceful_shutdown=3)
    # uvicorn re-raises the exit signal once shut down; uvicorn.run() swallows it the same way.
    with contextlib.suppress(KeyboardInterrupt):
        _Server(config, app.state.runtime).run()


if __name__ == "__main__":
    main()
