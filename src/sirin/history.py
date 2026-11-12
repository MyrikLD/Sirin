import asyncio
import logging
from datetime import UTC, datetime, timedelta

import numpy as np
import sqlalchemy as sa
from pydantic import AwareDatetime, BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from sirin.accounts import Account, Role
from sirin.agent import ToolCall
from sirin.database import Database
from sirin.tables import Reminder, Turn, User, Voice
from sirin.voices import Voices

# How often turns older than the retention are deleted.
CLEANUP_INTERVAL_S = 3600

_log = logging.getLogger(__name__)


class TurnOut(BaseModel):
    id: int
    started_at: AwareDatetime
    device: str
    conversation_id: str
    voice_id: int | None = Field(description="The voice that spoke; None when it was not recognized or too short")
    voice_label: str | None
    person_id: int | None = Field(description="The person the voice is linked to")
    person_name: str | None
    voice_confirmed: bool = Field(description="The person said this turn is theirs")
    similarity: float | None = Field(description="Similarity with the closest known voice")
    speech_ms: int | None
    text: str = Field(description="What was recognized; empty when a question was met with silence")
    answer: str | None = Field(description="What was spoken back; None while the answer is being made")
    tools: list[ToolCall]
    timings: dict[str, int] = Field(description="Milliseconds from the start of the run to each stage")
    error: str | None


_SELECT = (
    sa.select(
        Turn.id,
        Turn.started_at,
        Turn.device,
        Turn.conversation_id,
        Turn.voice_id,
        Voice.label.label("voice_label"),
        Voice.user_id.label("person_id"),
        User.name.label("person_name"),
        Turn.voice_confirmed,
        Turn.similarity,
        Turn.speech_ms,
        Turn.text,
        Turn.answer,
        Turn.tools,
        Turn.timings,
        Turn.error,
    )
    .outerjoin(Voice, Voice.id == Turn.voice_id)
    .outerjoin(User, User.id == Voice.user_id)
)


def visible_to(account: Account) -> sa.ColumnElement[bool]:
    """The admin sees every turn; a member sees their own and the ones of no person."""
    if account.role == Role.ADMIN:
        return sa.true()
    return sa.or_(Voice.user_id.is_(None), Voice.user_id == account.id)


async def list_turns(
    session: AsyncSession,
    viewer: Account,
    device: str | None = None,
    person_id: int | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    before_id: int | None = None,
    limit: int = 50,
) -> list[TurnOut]:
    """Newest first; before_id continues a page."""
    stmt = _SELECT.where(visible_to(viewer)).order_by(Turn.id.desc()).limit(limit)
    if device is not None:
        stmt = stmt.where(Turn.device == device)
    if person_id is not None:
        stmt = stmt.where(Voice.user_id == person_id)
    if since is not None:
        stmt = stmt.where(Turn.started_at >= since)
    if until is not None:
        stmt = stmt.where(Turn.started_at < until)
    if before_id is not None:
        stmt = stmt.where(Turn.id < before_id)
    rows = await session.execute(stmt)
    return [TurnOut.model_validate(dict(r)) for r in rows.mappings()]


async def get_turn(session: AsyncSession, turn_id: int, viewer: Account) -> TurnOut | None:
    rows = await session.execute(_SELECT.where(Turn.id == turn_id, visible_to(viewer)))
    row = rows.mappings().one_or_none()
    return TurnOut.model_validate(dict(row)) if row is not None else None


async def turn_devices(session: AsyncSession) -> list[str]:
    return list(await session.scalars(sa.select(Turn.device).distinct().order_by(Turn.device)))


class History:
    """Writes one `turns` row per utterance, tells the open history pages, and deletes turns past the retention."""

    def __init__(self, db: Database, voices: Voices | None) -> None:
        """voices: None without voice models; then turns get no voice and no embedding."""
        self._db = db
        self._voices = voices
        self.retention = timedelta(days=30)
        self._watchers: set[asyncio.Queue[int | None]] = set()
        self._writes: set[asyncio.Task[None]] = set()
        self._cleanup: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._cleanup = asyncio.create_task(self._clean_up_regularly())

    async def stop(self) -> None:
        if self._cleanup is not None:
            self._cleanup.cancel()
        # Pending writes finish: they hold the end of turns already spoken.
        await asyncio.gather(*self._writes, *([self._cleanup] if self._cleanup else []), return_exceptions=True)

    async def open_turn(
        self,
        device: str,
        conversation_id: str,
        started_at: datetime,
        voice_id: int | None,
        text: str,
        similarity: float | None,
        speech_ms: int | None,
    ) -> int:
        """Stores what was heard, before the answer exists, so tools of the answer can refer to the turn."""
        async with self._db.session() as session, session.begin():
            result = await session.execute(
                sa.insert(Turn)
                .values(
                    started_at=started_at,
                    device=device,
                    conversation_id=conversation_id,
                    # NULL when the voice was deleted while the user spoke.
                    voice_id=sa.select(Voice.id).where(Voice.id == voice_id).scalar_subquery(),
                    similarity=similarity,
                    speech_ms=speech_ms,
                    text=text,
                    tools=[],
                    timings={},
                )
                .returning(Turn.id)
            )
            turn_id = result.scalar_one()
        self._notify(turn_id)
        return turn_id

    def close_turn(
        self,
        turn_id: int,
        voice_id: int | None,
        answer: str,
        tools: list[ToolCall],
        timings: dict[str, int],
        error: str | None,
        speech: np.ndarray | None,
    ) -> None:
        """Stores the answer in the background, so the end of the run does not wait for the speaker network."""
        task = asyncio.create_task(self._close(turn_id, voice_id, answer, tools, timings, error, speech))
        self._writes.add(task)
        task.add_done_callback(self._writes.discard)

    async def _close(
        self,
        turn_id: int,
        voice_id: int | None,
        answer: str,
        tools: list[ToolCall],
        timings: dict[str, int],
        error: str | None,
        speech: np.ndarray | None,
    ) -> None:
        async with self._db.session() as session, session.begin():
            await session.execute(
                sa.update(Turn)
                .where(Turn.id == turn_id)
                .values(
                    answer=answer,
                    tools=[call.model_dump(mode="json") for call in tools],
                    timings=timings,
                    error=error,
                )
            )
        if self._voices is not None and speech is not None:
            try:
                await self._voices.learn(turn_id, voice_id, speech)
            except Exception:
                _log.exception("turn %d: its voice is not learned", turn_id)
        self._notify(turn_id)

    def watch(self) -> asyncio.Queue[int | None]:
        """Ids of turns as they are added or changed; None once the service stops. Pair with unwatch()."""
        queue: asyncio.Queue[int | None] = asyncio.Queue()
        self._watchers.add(queue)
        return queue

    def unwatch(self, queue: asyncio.Queue[int | None]) -> None:
        self._watchers.discard(queue)

    def close(self) -> None:
        """Ends the watchers' streams, so a server shutting down does not wait for open history pages."""
        for queue in self._watchers:
            queue.put_nowait(None)

    def _notify(self, turn_id: int) -> None:
        for queue in self._watchers:
            queue.put_nowait(turn_id)

    async def clean_up(self) -> int:
        """Deletes the turns older than the retention; a turn a pending reminder refers to stays."""
        cutoff = datetime.now(UTC) - self.retention
        referenced = sa.select(Reminder.turn_id).where(Reminder.turn_id.is_not(None))
        async with self._db.session() as session, session.begin():
            deleted = await session.scalars(
                sa.delete(Turn).where(Turn.started_at < cutoff, Turn.id.not_in(referenced)).returning(Turn.id)
            )
            count = len(deleted.all())
        if count:
            _log.info("deleted %d turn(s) older than %s", count, cutoff)
        if self._voices is not None:
            await self._voices.clean_up()
        return count

    async def _clean_up_regularly(self) -> None:
        while True:
            try:
                await self.clean_up()
            except Exception:
                _log.exception("history cleanup failed")
            await asyncio.sleep(CLEANUP_INTERVAL_S)
