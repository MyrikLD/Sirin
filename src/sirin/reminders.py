import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import AwareDatetime, BaseModel, TypeAdapter
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset

from sirin.agent import TurnDeps
from sirin.files import write_private

_log = logging.getLogger(__name__)

# How long a reminder is retried while its device is offline before it is dropped unannounced.
GRACE = timedelta(hours=1)
# Longest single sleep while a reminder waits for its due time.
MAX_SLEEP_S = 60


class Reminder(BaseModel):
    id: str
    device: str
    due: AwareDatetime
    prompt: str
    created: AwareDatetime

    def instruction(self) -> str:
        """What the agent receives at the due time."""
        created = self.created.astimezone().strftime("%Y-%m-%d %H:%M")
        return (
            f"A reminder you set at {created} is due now. Carry out your note to yourself; "
            f"your answer is announced on the device: {self.prompt}"
        )


# True once announced, False if the device cannot take it now and the reminder should be retried.
Deliver = Callable[[Reminder], Awaitable[bool]]

_reminder_list = TypeAdapter(list[Reminder])


class Reminders:
    """Pending reminders, kept in data/reminders.json so they survive a restart; one task waits for each."""

    def __init__(self, data_dir: Path, deliver: Deliver, retry_s: float = 30) -> None:
        self._path = data_dir / "reminders.json"
        self._deliver = deliver
        self._retry_s = retry_s
        self._items: dict[str, Reminder] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def start(self) -> None:
        if not self._path.exists():
            return
        saved = _reminder_list.validate_json(self._path.read_text())
        now = datetime.now(UTC)
        for reminder in saved:
            if reminder.due <= now:
                _log.warning("reminder %s for %s is dropped: missed while down", reminder.id, reminder.device)
                continue
            self._items[reminder.id] = reminder
            self._schedule(reminder)
        if len(self._items) != len(saved):
            self._save()

    async def stop(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def add(self, device: str, due: datetime, prompt: str) -> Reminder:
        reminder = Reminder(id=uuid.uuid4().hex[:8], device=device, due=due, prompt=prompt, created=datetime.now(UTC))
        self._items[reminder.id] = reminder
        self._save()
        self._schedule(reminder)
        return reminder

    def pending(self, device: str) -> list[Reminder]:
        return sorted((r for r in self._items.values() if r.device == device), key=lambda r: r.due)

    def cancel(self, device: str, reminder_id: str) -> bool:
        """Its task is left to find the reminder gone: the task may be the very run that calls this."""
        reminder = self._items.get(reminder_id)
        if reminder is None or reminder.device != device:
            return False
        self._remove(reminder_id)
        return True

    def _schedule(self, reminder: Reminder) -> None:
        task = asyncio.create_task(self._run(reminder))
        self._tasks[reminder.id] = task
        task.add_done_callback(lambda _: self._tasks.pop(reminder.id, None))

    def _remove(self, reminder_id: str) -> None:
        if self._items.pop(reminder_id, None) is not None:
            self._save()

    def _save(self) -> None:
        write_private(self._path, _reminder_list.dump_json(list(self._items.values()), indent=2).decode())

    async def _run(self, reminder: Reminder) -> None:
        # Short sleeps re-read the wall clock, so a clock step does not delay the reminder.
        while (left := (reminder.due - datetime.now(UTC)).total_seconds()) > 0:  # noqa: ASYNC110
            await asyncio.sleep(min(left, MAX_SLEEP_S))
        while reminder.id in self._items:
            if datetime.now(UTC) - reminder.due > GRACE:
                _log.warning("reminder %s for %s is dropped: due %s", reminder.id, reminder.device, reminder.due)
                break
            if await self._fire(reminder):
                break
            _log.info("reminder %s: %s is unavailable, retrying", reminder.id, reminder.device)
            await asyncio.sleep(self._retry_s)
        self._remove(reminder.id)

    async def _fire(self, reminder: Reminder) -> bool:
        _log.info("reminder %s for %s: %r", reminder.id, reminder.device, reminder.prompt)
        try:
            return await self._deliver(reminder)
        except Exception:
            # Retrying would run the agent again for an error that is not the device's absence.
            _log.exception("reminder %s failed and is dropped", reminder.id)
            return True


async def _with_local_time(ctx: RunContext[TurnDeps], tool_def: ToolDefinition) -> ToolDefinition:
    now = datetime.now().astimezone()
    return replace(tool_def, description=f"{tool_def.description}\n\nCurrent local time: {now:%Y-%m-%d %H:%M, %A}.")


def reminder_tools(reminders: Reminders) -> FunctionToolset[TurnDeps]:
    # The tools are async: pydantic-ai runs a plain def in a worker thread, and Reminders belongs to the event loop.
    tools = FunctionToolset[TurnDeps]()

    @tools.tool(prepare=_with_local_time)
    async def set_reminder(
        ctx: RunContext[TurnDeps], prompt: str, in_minutes: float | None = None, at: datetime | None = None
    ) -> str:
        """Schedules a reminder on the device the user speaks to. Give exactly one of in_minutes and at.

        At the due time you receive the prompt as an instruction, and your answer is announced on this device.

        Args:
            prompt: A note to yourself in the user's language, e.g. "Remind the user to turn off the stove".
            in_minutes: Delay from now, for relative times such as "in 10 minutes".
            at: Local date and time, for absolute times such as "tomorrow at 9".
        """
        now = datetime.now().astimezone()
        if in_minutes is not None and at is None:
            due = now + timedelta(minutes=in_minutes)
        elif at is not None and in_minutes is None:
            # A time without an offset is local time, the same clock the prompt's now() shows.
            due = at.astimezone()
        else:
            raise ModelRetry("Give exactly one of in_minutes and at.")
        if due <= now:
            raise ModelRetry(f"{due:%Y-%m-%d %H:%M} is in the past; the local time is {now:%Y-%m-%d %H:%M}.")
        reminder = reminders.add(ctx.deps.device, due.astimezone(UTC), prompt)
        return f"Reminder {reminder.id} is set for {due:%Y-%m-%d %H:%M, %A}."

    @tools.tool(prepare=_with_local_time)
    async def list_reminders(ctx: RunContext[TurnDeps]) -> list[dict[str, str]]:
        """Lists the pending reminders of this device, soonest first."""
        return [
            {"id": r.id, "due": f"{r.due.astimezone():%Y-%m-%d %H:%M, %A}", "prompt": r.prompt}
            for r in reminders.pending(ctx.deps.device)
        ]

    @tools.tool
    async def cancel_reminder(ctx: RunContext[TurnDeps], reminder_id: str) -> str:
        """Cancels a pending reminder of this device; the id comes from list_reminders."""
        if not reminders.cancel(ctx.deps.device, reminder_id):
            raise ModelRetry(f"There is no reminder {reminder_id}; call list_reminders.")
        return f"Reminder {reminder_id} is cancelled."

    return tools
