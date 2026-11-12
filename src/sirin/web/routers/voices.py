from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field

from sirin.voice_links import (
    NotAllowed,
    VoiceNotFound,
    VoiceOut,
    claim_turn,
    delete_voice,
    list_voices,
    release_turn,
    update_voice,
)
from sirin.web.deps import AccountDep, RuntimeDep, SessionDep, current_account

router = APIRouter(dependencies=[Depends(current_account)])


class VoiceChange(BaseModel):
    person_id: int | None = Field(default=None, description="Link to this person (admin only)")
    unlink: bool = Field(default=False, description="Take the voice from its person")


def _refresh(runtime: RuntimeDep, background: BackgroundTasks) -> None:
    """The gate's profiles are read again once this request is committed."""
    if runtime.voices is not None:
        background.add_task(runtime.voices.reload)


async def _guarded(call):
    try:
        return await call
    except VoiceNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except NotAllowed as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc


@router.get("/api/voices")
async def get_voices(account: AccountDep, session: SessionDep) -> list[VoiceOut]:
    return await list_voices(session, account)


@router.patch("/api/voices/{voice_id}", status_code=204)
async def patch_voice(
    voice_id: int,
    change: VoiceChange,
    account: AccountDep,
    session: SessionDep,
    runtime: RuntimeDep,
    background: BackgroundTasks,
) -> None:
    await _guarded(update_voice(session, account, voice_id, change.person_id, change.unlink))
    _refresh(runtime, background)


@router.delete("/api/voices/{voice_id}", status_code=204)
async def delete_voice_route(
    voice_id: int, account: AccountDep, session: SessionDep, runtime: RuntimeDep, background: BackgroundTasks
) -> None:
    await _guarded(delete_voice(session, account, voice_id))
    _refresh(runtime, background)


@router.post("/api/history/{turn_id}/mine", status_code=204)
async def post_mine(
    turn_id: int, account: AccountDep, session: SessionDep, runtime: RuntimeDep, background: BackgroundTasks
) -> None:
    """ "This is me": the turn's voice becomes the caller's, past turns included."""
    await _guarded(claim_turn(session, account, turn_id))
    _refresh(runtime, background)


@router.delete("/api/history/{turn_id}/voice", status_code=204)
async def delete_turn_voice(
    turn_id: int, account: AccountDep, session: SessionDep, runtime: RuntimeDep, background: BackgroundTasks
) -> None:
    """ "Not this voice": takes a wrongly matched turn out of its voice."""
    await _guarded(release_turn(session, account, turn_id))
    _refresh(runtime, background)
