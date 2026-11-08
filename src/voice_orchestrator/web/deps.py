from typing import Annotated

from fastapi import Depends, Request

from voice_orchestrator.clips import ClipStore
from voice_orchestrator.config import SettingsStore
from voice_orchestrator.oauth import OAuthManager
from voice_orchestrator.runtime import Runtime
from voice_orchestrator.status import StatusBoard


def get_clips(request: Request) -> ClipStore:
    return request.app.state.clips


def get_store(request: Request) -> SettingsStore:
    return request.app.state.store


def get_runtime(request: Request) -> Runtime:
    return request.app.state.runtime


def get_oauth(request: Request) -> OAuthManager:
    return request.app.state.oauth


def get_board(request: Request) -> StatusBoard:
    return request.app.state.board


ClipsDep = Annotated[ClipStore, Depends(get_clips)]
StoreDep = Annotated[SettingsStore, Depends(get_store)]
RuntimeDep = Annotated[Runtime, Depends(get_runtime)]
OAuthDep = Annotated[OAuthManager, Depends(get_oauth)]
BoardDep = Annotated[StatusBoard, Depends(get_board)]
