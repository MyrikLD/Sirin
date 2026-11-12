from datetime import UTC, datetime, timedelta

from sirin.accounts import Account, Role
from sirin.history import History, list_turns
from sirin.reminders import Reminders

ADMIN = Account(id=0, login="admin", name="Admin", role=Role.ADMIN, created_at="2026-01-01T00:00:00Z")


async def test_retention_keeps_the_turn_of_a_pending_reminder(db):
    history = History(db, None)
    long_ago = datetime.now(UTC) - timedelta(days=40)
    old = await history.open_turn("kitchen", "c", long_ago, None, "remind me", None, None)
    kept = await history.open_turn("kitchen", "c", long_ago, None, "remind me later", None, None)
    recent = await history.open_turn("kitchen", "c", datetime.now(UTC), None, "hello", None, None)

    async def deliver(_) -> bool:
        return True

    reminders = Reminders(db, deliver)
    await reminders.add("kitchen", datetime.now(UTC) + timedelta(days=1), "x", turn_id=kept)
    await reminders.stop()

    assert await history.clean_up() == 1
    async with db.session() as session:
        assert [t.id for t in await list_turns(session, ADMIN)] == [recent, kept]
    assert old not in (recent, kept)
