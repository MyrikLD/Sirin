from typing import Any

from fastapi import APIRouter

from voice_orchestrator.config import RuntimeSettings
from voice_orchestrator.web.deps import RuntimeDep

router = APIRouter(prefix="/api/settings")


@router.get("")
async def get_settings(runtime: RuntimeDep) -> RuntimeSettings:
    return runtime.settings


@router.get("/schema")
async def get_settings_schema() -> dict[str, Any]:
    return RuntimeSettings.model_json_schema()


# The page learns the resulting status from /api/status/stream.
@router.put("", status_code=204)
async def put_settings(settings: RuntimeSettings, runtime: RuntimeDep) -> None:
    await runtime.apply(settings)
