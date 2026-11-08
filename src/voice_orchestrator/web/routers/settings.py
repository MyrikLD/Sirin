from typing import Any

from fastapi import APIRouter

from voice_orchestrator.config import RuntimeSettings
from voice_orchestrator.status import Status
from voice_orchestrator.web.deps import RuntimeDep, StoreDep

router = APIRouter(prefix="/api/settings")


@router.get("")
async def get_settings(runtime: RuntimeDep) -> RuntimeSettings:
    return runtime.settings


@router.get("/schema")
async def get_settings_schema() -> dict[str, Any]:
    return RuntimeSettings.model_json_schema()


@router.put("")
async def put_settings(settings: RuntimeSettings, runtime: RuntimeDep, store: StoreDep) -> Status:
    store.save(settings)
    await runtime.apply(settings)
    return runtime.status()
