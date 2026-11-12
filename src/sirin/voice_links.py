"""What people do with voices on the page: claim a turn as theirs, link a voice to a person, delete it.

A person may have several voices: each of them is recognized as that person.
"""

import sqlalchemy as sa
from pydantic import AwareDatetime, BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from sirin.accounts import Account, Role
from sirin.tables import Turn, User, Voice
from sirin.voices import recompute_profile


class VoiceNotFound(LookupError):
    pass


class NotAllowed(PermissionError):
    pass


class VoiceOut(BaseModel):
    id: int
    person_id: int | None = Field(description="The person the voice is linked to; None for a voice nobody claimed")
    person_name: str | None
    created_at: AwareDatetime
    turns: int = Field(description="History turns in this voice")
    last_heard: AwareDatetime | None
    isolates: bool = Field(description="Its voiceprint exists, so once linked it mutes other speech")


async def list_voices(session: AsyncSession, viewer: Account) -> list[VoiceOut]:
    """The admin sees every voice; a member sees the voices linked to them."""
    usage = (
        sa.select(
            Turn.voice_id,
            sa.func.count().label("turns"),
            sa.func.max(Turn.started_at).label("last_heard"),
        )
        .group_by(Turn.voice_id)
        .subquery()
    )
    stmt = (
        sa.select(
            Voice.id,
            Voice.user_id.label("person_id"),
            User.name.label("person_name"),
            Voice.created_at,
            sa.func.coalesce(usage.c.turns, 0).label("turns"),
            usage.c.last_heard,
            Voice.voiceprint.is_not(None).label("isolates"),
        )
        .outerjoin(User, User.id == Voice.user_id)
        .outerjoin(usage, usage.c.voice_id == Voice.id)
        .order_by(Voice.id)
    )
    if viewer.role != Role.ADMIN:
        stmt = stmt.where(Voice.user_id == viewer.id)
    rows = await session.execute(stmt)
    return [VoiceOut.model_validate(dict(r)) for r in rows.mappings()]


async def claim_turn(session: AsyncSession, viewer: Account, turn_id: int) -> int:
    """ "This is me": links the turn's voice to viewer, if nobody else has it, and makes the turn an anchor.

    Returns the voice id. The link covers every turn of the voice, past ones included.
    """
    voice_id, owner = await _turn_voice(session, turn_id)
    if voice_id is None:
        raise VoiceNotFound("this turn has no voice: it was too short to tell")
    if owner is not None and owner != viewer.id:
        raise NotAllowed("this voice belongs to another person")
    await session.execute(sa.update(Voice).where(Voice.id == voice_id).values(user_id=viewer.id))
    await session.execute(sa.update(Turn).where(Turn.id == turn_id).values(voice_confirmed=True))
    await recompute_profile(session, voice_id)
    return voice_id


async def release_turn(session: AsyncSession, viewer: Account, turn_id: int) -> int:
    """ "Not my voice": takes the turn out of its voice, for a turn the gate matched wrongly."""
    voice_id, owner = await _turn_voice(session, turn_id)
    if voice_id is None:
        raise VoiceNotFound("this turn has no voice")
    _check_owner(viewer, owner)
    await session.execute(sa.update(Turn).where(Turn.id == turn_id).values(voice_id=None, voice_confirmed=False))
    await recompute_profile(session, voice_id)
    return voice_id


async def update_voice(
    session: AsyncSession,
    viewer: Account,
    voice_id: int,
    person_id: int | None = None,
    unlink: bool = False,
) -> None:
    """A member unlinks their own voice; the admin links any voice to any person or to nobody."""
    owner = await _voice_owner(session, voice_id)
    _check_owner(viewer, owner)
    values: dict[str, object] = {}
    if person_id is not None:
        if viewer.role != Role.ADMIN:
            raise NotAllowed("only an admin links a voice to another person")
        if await session.scalar(sa.select(User.id).where(User.id == person_id)) is None:
            raise VoiceNotFound(f"there is no person {person_id}")
        values["user_id"] = person_id
    if unlink:
        values["user_id"] = None
    if values:
        await session.execute(sa.update(Voice).where(Voice.id == voice_id).values(**values))


async def delete_voice(session: AsyncSession, viewer: Account, voice_id: int) -> None:
    """Its turns stay in the history without a voice."""
    _check_owner(viewer, await _voice_owner(session, voice_id))
    await session.execute(sa.delete(Voice).where(Voice.id == voice_id))


async def _turn_voice(session: AsyncSession, turn_id: int) -> tuple[int | None, int | None]:
    row = (
        await session.execute(
            sa.select(Turn.voice_id, Voice.user_id)
            .outerjoin(Voice, Voice.id == Turn.voice_id)
            .where(Turn.id == turn_id)
        )
    ).one_or_none()
    if row is None:
        raise VoiceNotFound(f"there is no turn {turn_id}")
    return row.voice_id, row.user_id


async def _voice_owner(session: AsyncSession, voice_id: int) -> int | None:
    row = (await session.execute(sa.select(Voice.user_id).where(Voice.id == voice_id))).one_or_none()
    if row is None:
        raise VoiceNotFound(f"there is no voice {voice_id}")
    return row.user_id


def _check_owner(viewer: Account, owner: int | None) -> None:
    """A member acts on their own voices; a voice of nobody is the admin's to handle."""
    if viewer.role == Role.ADMIN:
        return
    if owner is None:
        raise NotAllowed("only an admin changes a voice nobody has claimed")
    if owner != viewer.id:
        raise NotAllowed("this voice belongs to another person")
