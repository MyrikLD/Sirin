from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.sse import EventSourceResponse
from pydantic import AwareDatetime

from sirin.history import TurnOut, get_turn, list_turns, turn_devices
from sirin.web.deps import RuntimeDep, SessionDep, current_account

# Every logged-in person reads the history; nobody edits it.
router = APIRouter(prefix="/api/history", dependencies=[Depends(current_account)])


@router.get("")
async def get_history(
    session: SessionDep,
    device: str | None = None,
    since: AwareDatetime | None = None,
    until: AwareDatetime | None = None,
    before_id: Annotated[int | None, Query(description="Continue a page: turns older than this one")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[TurnOut]:
    return await list_turns(session, device, since, until, before_id, limit)


@router.get("/devices")
async def get_history_devices(session: SessionDep) -> list[str]:
    return await turn_devices(session)


@router.get("/stream", response_class=EventSourceResponse)
async def stream_history(runtime: RuntimeDep) -> AsyncIterator[TurnOut]:
    """Server-sent events: each turn as it is added and again when its answer is stored."""
    queue = runtime.history.watch()
    try:
        while (turn_id := await queue.get()) is not None:
            async with runtime.db.session() as session:
                turn = await get_turn(session, turn_id)
            if turn is not None:
                yield turn
    finally:
        runtime.history.unwatch(queue)
