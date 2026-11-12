import asyncio
from datetime import UTC, datetime, timedelta

import sqlalchemy as sa
from pydantic_ai import Agent, ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from sirin import tables
from sirin.agent import TurnDeps
from sirin.database import Database
from sirin.reminders import Reminder, Reminders, cron_next, reminder_tools


async def _store(db: Database, saved: list[Reminder]) -> None:
    async with db.session() as session, session.begin():
        await session.execute(sa.insert(tables.Reminder), [r.model_dump() for r in saved])


async def _stored(db: Database) -> list[Reminder]:
    async with db.session() as session:
        rows = await session.execute(sa.select(*(getattr(tables.Reminder, f) for f in Reminder.model_fields)))
        return [Reminder.model_validate(dict(r)) for r in rows.mappings()]


async def _wait_for(condition) -> None:
    async with asyncio.timeout(2):
        while not condition():  # noqa: ASYNC110 - the state lives in the scheduler, which offers no event
            await asyncio.sleep(0.01)


async def test_due_reminder_is_announced_once_and_forgotten(db):
    announced: list[Reminder] = []

    async def deliver(reminder: Reminder) -> bool:
        announced.append(reminder)
        return True

    reminders = Reminders(db, deliver)
    await reminders.start()
    try:
        await reminders.add("kitchen", datetime.now(UTC) + timedelta(milliseconds=100), "stove")
        assert [r.prompt for r in await _stored(db)] == ["stove"]
        await _wait_for(lambda: not reminders.pending("kitchen"))
    finally:
        await reminders.stop()
    assert [r.prompt for r in announced] == ["stove"]
    assert await _stored(db) == []


async def test_offline_device_is_retried(db):
    attempts = 0

    async def deliver(reminder: Reminder) -> bool:
        nonlocal attempts
        attempts += 1
        return attempts == 3

    reminders = Reminders(db, deliver, retry_s=0.05)
    await reminders.start()
    try:
        await reminders.add("kitchen", datetime.now(UTC), "stove")
        await _wait_for(lambda: not reminders.pending("kitchen"))
    finally:
        await reminders.stop()
    assert attempts == 3


async def test_restart_keeps_pending_and_drops_missed(db):
    now = datetime.now(UTC)
    saved = [
        Reminder(id="eat", device="kitchen", due=now + timedelta(hours=1), prompt="eat", created=now),
        Reminder(id="missed", device="kitchen", due=now - timedelta(seconds=1), prompt="missed", created=now),
    ]
    await _store(db, saved)

    announced: list[Reminder] = []

    async def deliver(reminder: Reminder) -> bool:
        announced.append(reminder)
        return True

    reminders = Reminders(db, deliver)
    await reminders.start()
    await reminders.stop()
    assert [r.prompt for r in reminders.pending("kitchen")] == ["eat"]
    assert [r.id for r in await _stored(db)] == ["eat"]
    assert announced == []


def _calling(tool: str, args: dict):
    """A model that calls one tool once and then answers with what the tool said."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        said = [p for m in messages for p in getattr(m, "parts", []) if isinstance(p, ToolReturnPart | RetryPromptPart)]
        if said:
            return ModelResponse(parts=[TextPart(str(said[-1].content))])
        return ModelResponse(parts=[ToolCallPart(tool, args)])

    return FunctionModel(model)


async def test_tools_take_local_time_and_keep_devices_apart(db):
    async def deliver(reminder: Reminder) -> bool:
        return True

    reminders = Reminders(db, deliver)
    agent = Agent(deps_type=TurnDeps, toolsets=[reminder_tools(reminders)])
    try:
        kitchen = TurnDeps(device="kitchen")

        local = (datetime.now() + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        at = {"prompt": "eat", "at": local.isoformat()}
        await agent.run("remind", model=_calling("set_reminder", at), deps=kitchen)
        (reminder,) = reminders.pending("kitchen")
        assert reminder.due == local.astimezone()

        past = {"prompt": "eat", "in_minutes": -5}
        result = await agent.run("remind", model=_calling("set_reminder", past), deps=kitchen)
        assert "in the past" in result.output
        assert len(reminders.pending("kitchen")) == 1

        other = TurnDeps(device="bedroom")
        await agent.run("cancel", model=_calling("cancel_reminder", {"reminder_id": reminder.id}), deps=other)
        assert len(reminders.pending("kitchen")) == 1
        await agent.run("cancel", model=_calling("cancel_reminder", {"reminder_id": reminder.id}), deps=kitchen)
        assert reminders.pending("kitchen") == []
    finally:
        await reminders.stop()


async def test_recurring_reminder_moves_to_next_time(db):
    announced: list[Reminder] = []

    async def deliver(reminder: Reminder) -> bool:
        announced.append(reminder)
        return True

    reminders = Reminders(db, deliver)
    await reminders.start()
    try:
        due = datetime.now(UTC) + timedelta(milliseconds=100)
        reminder = await reminders.add("kitchen", due, "weather", schedule="0 8-10 * * *")
        await _wait_for(lambda: announced)
        await _wait_for(lambda: reminders.pending("kitchen")[0].due != due)
        (pending,) = reminders.pending("kitchen")
        assert pending.id == reminder.id
        assert pending.due == cron_next("0 8-10 * * *", datetime.now(UTC))
        assert [r.due for r in await _stored(db)] == [pending.due]
        assert await reminders.cancel("kitchen", reminder.id)
    finally:
        await reminders.stop()
    assert len(announced) == 1


async def test_recurring_reminder_ends_at_until(db):
    async def deliver(reminder: Reminder) -> bool:
        return True

    reminders = Reminders(db, deliver)
    await reminders.start()
    try:
        now = datetime.now(UTC)
        await reminders.add("kitchen", now + timedelta(milliseconds=100), "x", schedule="0 8 * * *", until=now)
        await _wait_for(lambda: not reminders.pending("kitchen"))
    finally:
        await reminders.stop()
    assert await _stored(db) == []


async def test_restart_moves_missed_recurring_to_next_time(db):
    now = datetime.now(UTC)
    missed = Reminder(
        id="d", device="kitchen", due=now - timedelta(days=2), prompt="x", created=now, schedule="0 8 * * *"
    )
    await _store(db, [missed])

    async def deliver(reminder: Reminder) -> bool:
        raise AssertionError("a missed occurrence is not announced")

    reminders = Reminders(db, deliver)
    await reminders.start()
    await reminders.stop()
    (pending,) = reminders.pending("kitchen")
    assert pending.due == cron_next("0 8 * * *", now)


def test_cron_is_read_in_local_time():
    after = datetime.now().astimezone().replace(hour=12, minute=0, second=0, microsecond=0)
    due = cron_next("30 9 * * *", after).astimezone()
    assert (due.hour, due.minute) == (9, 30)
    assert due.date() == (after + timedelta(days=1)).date()


async def test_repeat_tool(db):
    async def deliver(reminder: Reminder) -> bool:
        return True

    reminders = Reminders(db, deliver)
    agent = Agent(deps_type=TurnDeps, toolsets=[reminder_tools(reminders)])
    kitchen = TurnDeps(device="kitchen")
    try:
        bad = {"prompt": "x", "repeat": "every day"}
        result = await agent.run("remind", model=_calling("set_reminder", bad), deps=kitchen)
        assert "not a valid" in result.output
        assert reminders.pending("kitchen") == []

        start = (datetime.now() + timedelta(days=3)).replace(hour=0, minute=0, second=0, microsecond=0)
        args = {"prompt": "weather", "repeat": "0 8-10 * * *", "at": start.isoformat()}
        result = await agent.run("remind", model=_calling("set_reminder", args), deps=kitchen)
        assert "Recurring reminder" in result.output
        (reminder,) = reminders.pending("kitchen")
        assert reminder.schedule == "0 8-10 * * *"
        assert reminder.due == start.replace(hour=8).astimezone()
    finally:
        await reminders.stop()
