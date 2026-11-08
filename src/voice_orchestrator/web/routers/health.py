from fastapi import APIRouter

from voice_orchestrator.status import Status
from voice_orchestrator.web.deps import RuntimeDep

router = APIRouter()


@router.get("/healthcheck")
async def get_status(runtime: RuntimeDep) -> Status:
    return runtime.status()
