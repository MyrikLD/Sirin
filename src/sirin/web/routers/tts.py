from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from sirin.web.deps import RuntimeDep

router = APIRouter(prefix="/tts")


# The satellite fetches answers without credentials; clip ids are random and short-lived.
@router.get("/{clip_id}.wav")
async def tts_audio(clip_id: str, runtime: RuntimeDep) -> StreamingResponse:
    clip = runtime.clips.get(clip_id)
    if clip is None or not await clip.wait_audio():
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return StreamingResponse(clip.read(), media_type="audio/wav")
