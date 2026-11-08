import argparse
import logging
import os

import uvicorn
from pydantic import ValidationError

from voice_orchestrator.config import AppSettings
from voice_orchestrator.web import create_app


class _SkipHealthcheck(logging.Filter):
    """Drops access-log lines for /healthcheck, which the settings page polls every few seconds."""

    def filter(self, record: logging.LogRecord) -> bool:
        # uvicorn passes (client, method, path, http_version, status) as the record args.
        args = record.args
        return not (isinstance(args, tuple) and len(args) > 2 and str(args[2]).startswith("/healthcheck"))


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
    logging.getLogger("uvicorn.access").addFilter(_SkipHealthcheck())

    try:
        settings = AppSettings()
    except ValidationError as exc:
        raise SystemExit(f"Bad environment (VO_* variables or .env):\n{exc}") from exc
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_config=None)


if __name__ == "__main__":
    main()
