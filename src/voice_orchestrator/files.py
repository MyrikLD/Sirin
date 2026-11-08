import os
from pathlib import Path


def write_private(path: Path, text: str) -> None:
    """Atomically replaces path with text, readable only by the owner: these files hold secrets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.chmod(0o600)
    os.replace(tmp, path)
