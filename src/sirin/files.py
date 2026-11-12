import os
import tempfile
from pathlib import Path


def write_private(path: Path, content: str | bytes) -> None:
    """Atomically replaces path with content, readable only by the owner: these files hold secrets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp creates the file 0600 under a unique name: it is never readable by others, even before the rename.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w" if isinstance(content, str) else "wb") as file:
            file.write(content)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
