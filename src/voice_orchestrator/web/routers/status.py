import asyncio
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from voice_orchestrator.web.deps import RuntimeDep

router = APIRouter()

# A quiet stream still repeats the status this often, so proxies keep it open and a lost update heals.
RESEND_S = 15


@router.get("/api/status/stream")
async def stream_status(runtime: RuntimeDep, request: Request) -> StreamingResponse:
    """Server-sent events: the full status now and after every change."""

    async def events() -> AsyncIterator[str]:
        board = runtime.board
        while not board.closed and not await request.is_disconnected():
            changed = board.changed
            yield f"data: {runtime.status().model_dump_json()}\n\n"
            try:
                await asyncio.wait_for(changed.wait(), RESEND_S)
            except TimeoutError:
                pass

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
