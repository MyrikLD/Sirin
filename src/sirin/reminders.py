import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cronsim import CronSim, CronSimError
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
    schedule: str | None = None
    until: AwareDatetime | None = None

    def instruction(self) -> str:
        """What the agent receives at the due time."""
        created = self.created.astimezone().strftime("%Y-%m-%d %H:%M")
        kind = f"recurring reminder (cron {self.schedule!r})" if self.schedule else "reminder"
        return (
            f"A {kind} you set at {created} is due now. Carry out your note to yourself; "
            f"your answer is announced on the device: {self.prompt}"
        )

    def next_due(self, now: datetime) -> datetime | None:
        """The occurrence after this one and after now, or None for a one-off reminder or a schedule that has ended."""
        if self.schedule is None:
            return None
        due = cron_next(self.schedule, max(self.due, now))
        return None if self.until is not None and due > self.until else due


def cron_next(schedule: str, after: datetime) -> datetime:
    """The first time of a cron schedule, read in local time, that is later than after; in UTC."""
    # Local wall times keep "0 8 * * *" at 8 across a DST change; the comparison skips a repeated hour.
    for local in CronSim(schedule, after.astimezone().replace(tzinfo=None)):
        if (due := local.astimezone(UTC)) > after:
            return due
    raise CronSimError(f"{schedule!r} has no time after {after}")


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
        changed = False
        for reminder in saved:
            if reminder.due <= now:
                changed = True
                if (due := reminder.next_due(now)) is None:
                    _log.warning("reminder %s for %s is dropped: missed while down", reminder.id, reminder.device)
                    continue
                _log.warning("reminder %s for %s skips an occurrence missed while down", reminder.id, reminder.device)
                reminder = reminder.model_copy(update={"due": due})
            self._items[reminder.id] = reminder
            self._schedule(reminder)
        if changed:
            self._save()

    async def stop(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def add(
        self, device: str, due: datetime, prompt: str, schedule: str | None = None, until: datetime | None = None
    ) -> Reminder:
        reminder = Reminder(
            id=uuid.uuid4().hex[:8],
            device=device,
            due=due,
            prompt=prompt,
            created=datetime.now(UTC),
            schedule=schedule,
            until=until,
        )
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
        # A recurring reminder keeps this one task: a new task under the same id would be popped by this one's callback.
        while True:
            await self._occur(reminder)
            current = self._items.get(reminder.id)
            if current is None:
                return
            if (due := current.next_due(datetime.now(UTC))) is None:
                self._remove(reminder.id)
                return
            reminder = current.model_copy(update={"due": due})
            self._items[reminder.id] = reminder
            self._save()

    async def _occur(self, reminder: Reminder) -> None:
        """Waits for the due time and announces this occurrence, retrying while the device is offline."""
        # Short sleeps re-read the wall clock, so a clock step does not delay the reminder.
        while reminder.id in self._items and (left := (reminder.due - datetime.now(UTC)).total_seconds()) > 0:  # noqa: ASYNC110
            await asyncio.sleep(min(left, MAX_SLEEP_S))
        while reminder.id in self._items:
            if datetime.now(UTC) - reminder.due > GRACE:
                _log.warning("reminder %s for %s is dropped: due %s", reminder.id, reminder.device, reminder.due)
                return
            if await self._fire(reminder):
                return
            _log.info("reminder %s: %s is unavailable, retrying", reminder.id, reminder.device)
            await asyncio.sleep(self._retry_s)

    async def _fire(self, reminder: Reminder) -> bool:
        _log.info("reminder %s for %s: %r", reminder.id, reminder.device, reminder.prompt)
        try:
            return await self._deliver(reminder)
        except Exception:
            # Retrying would run the agent again for an error that is not the device's absence.
            _log.exception("reminder %s failed and is not retried", reminder.id)
            return True


async def _with_local_time(ctx: RunContext[TurnDeps], tool_def: ToolDefinition) -> ToolDefinition:
    now = datetime.now().astimezone()
    return replace(tool_def, description=f"{tool_def.description}\n\nCurrent local time: {now:%Y-%m-%d %H:%M, %A}.")


def reminder_tools(reminders: Reminders) -> FunctionToolset[TurnDeps]:
    # The tools are async: pydantic-ai runs a plain def in a worker thread, and Reminders belongs to the event loop.
    tools = FunctionToolset[TurnDeps]()

    @tools.tool(prepare=_with_local_time)
    async def set_reminder(
        ctx: RunContext[TurnDeps],
        prompt: str,
        in_minutes: float | None = None,
        at: datetime | None = None,
        repeat: str | None = None,
        until: datetime | None = None,
    ) -> str:
        """Schedules a reminder on the device the user speaks to.

        At the due time you receive the prompt as an instruction, and your answer is announced on this device.
        A one-off reminder takes exactly one of in_minutes and at. A recurring one takes repeat, and at only to start
        later than now.

        Args:
            prompt: A note to yourself in the user's language, e.g. "Remind the user to turn off the stove".
            in_minutes: Delay from now, for relative times such as "in 10 minutes".
            at: Local date and time, for absolute times such as "tomorrow at 9". With repeat, the schedule starts
                at this time instead of now.
            repeat: A 5-field cron expression in local time (minute hour day-of-month month day-of-week) for
                recurring reminders, e.g. "0 8 * * *" every day at 8, "0 8-10 * * *" every hour from 8 to 10,
                "30 7 * * 1-5" on weekdays at 7:30. It repeats until cancelled or until the until time.
            until: Local date and time after which a recurring reminder stops, e.g. "until Friday".
        """
        now = datetime.now().astimezone()
        # A time without an offset is local time, the same clock the prompt's now() shows.
        start, end = (t.astimezone() if t is not None else None for t in (at, until))
        if repeat is not None:
            if in_minutes is not None:
                raise ModelRetry("A recurring reminder takes no in_minutes; give at to start later.")
            try:
                # A second before start lets the schedule's first time be start itself.
                due = cron_next(repeat, max(start - timedelta(seconds=1), now) if start is not None else now)
            except CronSimError as e:
                raise ModelRetry(f"repeat {repeat!r} is not a valid 5-field cron expression: {e}") from e
            if end is not None and due > end:
                raise ModelRetry(f"The first time, {due.astimezone():%Y-%m-%d %H:%M}, is after until.")
        elif until is not None:
            raise ModelRetry("until is only for a recurring reminder; give repeat too.")
        elif in_minutes is not None and at is None:
            due = now + timedelta(minutes=in_minutes)
        elif start is not None and in_minutes is None:
            due = start
        else:
            raise ModelRetry("Give exactly one of in_minutes and at.")
        if due <= now:
            raise ModelRetry(f"{due:%Y-%m-%d %H:%M} is in the past; the local time is {now:%Y-%m-%d %H:%M}.")
        reminder = reminders.add(ctx.deps.device, due.astimezone(UTC), prompt, repeat, end)
        first = f"{due.astimezone():%Y-%m-%d %H:%M, %A}"
        if repeat is not None:
            ends = f", until {end:%Y-%m-%d %H:%M}" if end is not None else ""
            return f"Recurring reminder {reminder.id} ({repeat}{ends}) is set; first at {first}."
        return f"Reminder {reminder.id} is set for {first}."

    @tools.tool(prepare=_with_local_time)
    async def list_reminders(ctx: RunContext[TurnDeps]) -> list[dict[str, str]]:
        """Lists the pending reminders of this device, soonest first; a recurring one shows its next time."""
        return [
            {"id": r.id, "due": f"{r.due.astimezone():%Y-%m-%d %H:%M, %A}", "prompt": r.prompt}
            | ({"repeat": r.schedule} if r.schedule else {})
            | ({"until": f"{r.until.astimezone():%Y-%m-%d %H:%M}"} if r.until else {})
            for r in reminders.pending(ctx.deps.device)
        ]

    @tools.tool
    async def cancel_reminder(ctx: RunContext[TurnDeps], reminder_id: str) -> str:
        """Cancels a pending reminder of this device, a recurring one for good; the id comes from list_reminders."""
        if not reminders.cancel(ctx.deps.device, reminder_id):
            raise ModelRetry(f"There is no reminder {reminder_id}; call list_reminders.")
        return f"Reminder {reminder_id} is cancelled."

    return tools
