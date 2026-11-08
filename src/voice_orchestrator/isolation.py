import asyncio
from collections.abc import Awaitable


async def run_isolated[T](coro: Awaitable[T]) -> T:
    """Runs an MCP connection in a throwaway task and reports back through a future.

    A failed anyio transport leaks its cancel scope into the task that ran it, which would
    otherwise silently cancel the caller.
    """
    result: asyncio.Future[T] = asyncio.get_running_loop().create_future()

    async def runner() -> None:
        try:
            result.set_result(await coro)
        except BaseException as exc:  # noqa: BLE001
            if not result.done():
                result.set_exception(exc if isinstance(exc, Exception) else RuntimeError("cancelled"))

    task = asyncio.create_task(runner())
    try:
        return await result
    finally:
        task.cancel()
