import argparse
import asyncio
import contextlib
import logging
import os
from types import FrameType

import uvicorn
from pydantic import ValidationError

from voice_orchestrator.config import AppSettings
from voice_orchestrator.status import StatusBoard
from voice_orchestrator.web import create_app


class _Server(uvicorn.Server):
    """Closes the status streams on the exit signal; otherwise shutdown would wait for every open settings page."""

    def __init__(self, config: uvicorn.Config, board: StatusBoard) -> None:
        super().__init__(config)
        self._board = board

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        super().handle_exit(sig, frame)
        asyncio.get_event_loop().call_soon_threadsafe(self._board.close)


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice pipeline for ESPHome satellites")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log protocol events")
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
        raise SystemExit(f"Bad environment (VO_* variables or .env):\n{exc}") from exc
    app = create_app(settings)
    config = uvicorn.Config(app, host=settings.host, port=settings.port, log_config=None, timeout_graceful_shutdown=3)
    # uvicorn re-raises the exit signal once shut down; uvicorn.run() swallows it the same way.
    with contextlib.suppress(KeyboardInterrupt):
        _Server(config, app.state.runtime.board).run()


if __name__ == "__main__":
    main()
