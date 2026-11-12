from typing import Any

from fastapi import APIRouter, HTTPException, status

from sirin.config import RuntimeSettings
from sirin.personal_servers import personal_names
from sirin.web.deps import RuntimeDep

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
    async with runtime.db.session() as session:
        taken = {m.name for m in settings.mcp_servers} & await personal_names(session)
    if taken:
        # A tool prefix has to stay unique among the servers a person's turn sees.
        raise HTTPException(status.HTTP_409_CONFLICT, f"personal MCP servers already use: {', '.join(sorted(taken))}")
    await runtime.apply(settings)
