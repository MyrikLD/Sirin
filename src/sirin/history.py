import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import numpy as np
import sqlalchemy as sa
from pydantic import AwareDatetime, BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from sirin.agent import ToolCall
from sirin.database import Database
from sirin.tables import Reminder, Turn

# How often turns older than the retention are deleted.
CLEANUP_INTERVAL_S = 3600
# Less speech than this gives no useful speaker embedding.
MIN_EMBED_SAMPLES = 8000

_log = logging.getLogger(__name__)


class TurnOut(BaseModel):
    id: int
    started_at: AwareDatetime
    device: str
    conversation_id: str
    similarity: float | None = Field(description="Similarity with the closest remembered voice")
    speech_ms: int | None
    text: str = Field(description="What was recognized; empty when a question was met with silence")
    answer: str | None = Field(description="What was spoken back; None while the answer is being made")
    tools: list[ToolCall]
    timings: dict[str, int] = Field(description="Milliseconds from the start of the run to each stage")
    error: str | None


_COLUMNS = [getattr(Turn, name) for name in TurnOut.model_fields]


async def list_turns(
    session: AsyncSession,
    device: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    before_id: int | None = None,
    limit: int = 50,
) -> list[TurnOut]:
    """Newest first; before_id continues a page."""
    stmt = sa.select(*_COLUMNS).order_by(Turn.id.desc()).limit(limit)
    if device is not None:
        stmt = stmt.where(Turn.device == device)
    if since is not None:
        stmt = stmt.where(Turn.started_at >= since)
    if until is not None:
        stmt = stmt.where(Turn.started_at < until)
    if before_id is not None:
        stmt = stmt.where(Turn.id < before_id)
    rows = await session.execute(stmt)
    return [TurnOut.model_validate(dict(r)) for r in rows.mappings()]


async def get_turn(session: AsyncSession, turn_id: int) -> TurnOut | None:
    row = (await session.execute(sa.select(*_COLUMNS).where(Turn.id == turn_id))).mappings().one_or_none()
    return TurnOut.model_validate(dict(row)) if row is not None else None


async def turn_devices(session: AsyncSession) -> list[str]:
    return list(await session.scalars(sa.select(Turn.device).distinct().order_by(Turn.device)))


class History:
    """Writes one `turns` row per utterance, tells the open history pages, and deletes turns past the retention."""

    def __init__(self, db: Database, embed: Callable[[np.ndarray], np.ndarray] | None) -> None:
        """embed: the speaker embedding of speech, or None without voice models."""
        self._db = db
        self._embed = embed
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
        answer: str,
        tools: list[ToolCall],
        timings: dict[str, int],
        error: str | None,
        speech: np.ndarray | None,
    ) -> None:
        """Stores the answer in the background, so the end of the run does not wait for the speaker network."""
        task = asyncio.create_task(self._close(turn_id, answer, tools, timings, error, speech))
        self._writes.add(task)
        task.add_done_callback(self._writes.discard)

    async def _close(
        self,
        turn_id: int,
        answer: str,
        tools: list[ToolCall],
        timings: dict[str, int],
        error: str | None,
        speech: np.ndarray | None,
    ) -> None:
        embedding = None
        if self._embed is not None and speech is not None and len(speech) >= MIN_EMBED_SAMPLES:
            try:
                embedding = (await asyncio.to_thread(self._embed, speech)).astype(np.float32).tobytes()
            except Exception:
                _log.exception("turn %d: no speaker embedding", turn_id)
        async with self._db.session() as session, session.begin():
            await session.execute(
                sa.update(Turn)
                .where(Turn.id == turn_id)
                .values(
                    answer=answer,
                    tools=[call.model_dump(mode="json") for call in tools],
                    timings=timings,
                    error=error,
                    embedding=embedding,
                )
            )
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
        return count

    async def _clean_up_regularly(self) -> None:
        while True:
            try:
                await self.clean_up()
            except Exception:
                _log.exception("history cleanup failed")
            await asyncio.sleep(CLEANUP_INTERVAL_S)
