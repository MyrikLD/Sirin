from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse

from sirin.status import Status
from sirin.web.deps import RuntimeDep

router = APIRouter()


@router.get("/api/status/stream", response_class=EventSourceResponse)
async def stream_status(runtime: RuntimeDep) -> AsyncIterator[Status]:
    """Server-sent events: the full status now and after every change."""
    board = runtime.board
    while not board.closed:
        changed = board.changed
        yield runtime.status()
        await changed.wait()
