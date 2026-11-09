from typing import Annotated

from fastapi import Depends, Request

from sirin.runtime import Runtime


def get_runtime(request: Request) -> Runtime:
    return request.app.state.runtime


RuntimeDep = Annotated[Runtime, Depends(get_runtime)]
