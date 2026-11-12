from datetime import UTC, datetime, timedelta

from sirin.history import History, list_turns
from sirin.reminders import Reminders


async def test_retention_keeps_the_turn_of_a_pending_reminder(db):
    history = History(db, embed=None)
    long_ago = datetime.now(UTC) - timedelta(days=40)
    old = await history.open_turn("kitchen", "c", long_ago, "remind me", None, None)
    kept = await history.open_turn("kitchen", "c", long_ago, "remind me later", None, None)
    recent = await history.open_turn("kitchen", "c", datetime.now(UTC), "hello", None, None)

    async def deliver(_) -> bool:
        return True

    reminders = Reminders(db, deliver)
    await reminders.add("kitchen", datetime.now(UTC) + timedelta(days=1), "x", turn_id=kept)
    await reminders.stop()

    assert await history.clean_up() == 1
    async with db.session() as session:
        assert [t.id for t in await list_turns(session)] == [recent, kept]
    assert old not in (recent, kept)
