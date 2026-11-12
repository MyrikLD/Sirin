from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import AwareDatetime, BaseModel, Field

from sirin.accounts import Role
from sirin.reminders import reminder_people
from sirin.web.deps import AccountDep, RuntimeDep, SessionDep, current_account

# As with the history: the admin sees every reminder, a member their own and the ones of no person.
router = APIRouter(prefix="/api/reminders", dependencies=[Depends(current_account)])


class ReminderOut(BaseModel):
    id: str
    due: AwareDatetime = Field(description="The next time it is announced")
    prompt: str = Field(description="The note the assistant wrote to itself")
    schedule: str | None = Field(description="Cron schedule in local time, for a recurring reminder")
    until: AwareDatetime | None
    created: AwareDatetime
    person_id: int | None = Field(description="Who set it: the person of the voice that asked; None if unknown")
    person_name: str | None


@router.get("")
async def get_reminders(account: AccountDep, session: SessionDep, runtime: RuntimeDep) -> list[ReminderOut]:
    people = await reminder_people(session)
    shown = []
    for reminder in runtime.reminders.pending():
        person_id, name = people.get(reminder.id, (None, None))
        if account.role == Role.ADMIN or person_id in (None, account.id):
            shown.append(ReminderOut(person_id=person_id, person_name=name, **reminder.model_dump()))
    return shown


@router.delete("/{reminder_id}", status_code=204)
async def delete_reminder(reminder_id: str, account: AccountDep, session: SessionDep, runtime: RuntimeDep) -> None:
    """A member cancels their own reminders; one nobody is known to have set is the admin's."""
    reminder = runtime.reminders.get(reminder_id)
    if reminder is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    owner = (await reminder_people(session)).get(reminder_id, (None, None))[0]
    if account.role != Role.ADMIN and owner != account.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only its person or an admin cancels this reminder")
    await runtime.reminders.cancel(reminder.device, reminder_id)
