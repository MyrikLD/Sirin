from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import httpx2
import pytest
import sqlalchemy as sa

from sirin.accounts import FIRST_ADMIN_LOGIN, reset_password
from sirin.tables import Voice
from sirin.web import create_app


@asynccontextmanager
async def _app(app_settings):
    app = create_app(app_settings)
    async with app.router.lifespan_context(app):
        yield app


def _client(app) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://test")


async def _password(app, login: str = FIRST_ADMIN_LOGIN) -> str:
    async with app.state.runtime.db.session() as session, session.begin():
        return await reset_password(session, login)


@asynccontextmanager
async def _logged_in(app, login: str, password: str):
    async with _client(app) as client:
        response = await client.post("/api/auth/login", json={"login": login, "password": password})
        assert response.status_code == 204
        yield client


@pytest.fixture
async def app(app_settings):
    async with _app(app_settings) as app:
        yield app


@pytest.fixture
async def client(app):
    async with _logged_in(app, FIRST_ADMIN_LOGIN, await _password(app)) as client:
        yield client


async def test_schema_describes_the_page(client):
    schema = (await client.get("/api/settings/schema")).json()
    assert schema["properties"]["stt"]["title"] == "Speech-to-text"
    assert schema["$defs"]["SttConfig"]["x-probe"] == "/api/probe/stt"
    assert schema["$defs"]["DeviceConfig"]["properties"]["noise_psk"]["format"] == "password"


async def test_invalid_settings_are_rejected(client):
    response = await client.put("/api/settings", json={"mcp_servers": [{"name": "x"}]})
    assert response.status_code == 422


async def test_saved_settings_survive_a_restart_and_stay_private(app_settings, client):
    response = await client.put("/api/settings", json={"tts": {"voice": "xenia"}})
    assert response.status_code == 204
    path = app_settings.data_dir / "sirin.db"
    assert path.stat().st_mode & 0o777 == 0o600

    async with (
        _app(app_settings) as restarted,
        _logged_in(restarted, FIRST_ADMIN_LOGIN, await _password(restarted)) as again,
    ):
        assert (await again.get("/api/settings")).json()["tts"]["voice"] == "xenia"


async def test_settings_are_for_the_admin_and_answers_for_anyone(app, client):
    member = {"login": "Anna", "name": "Anna", "password": "anna-secret"}
    assert (await client.post("/api/users", json=member)).status_code == 201
    assert (await client.post("/api/users", json=member | {"login": " anna "})).status_code == 409

    async with _client(app) as anonymous:
        assert (await anonymous.get("/api/settings")).status_code == 401
        assert (await anonymous.post("/api/probe/stt", json={})).status_code == 401
        assert (await anonymous.get("/tts/unknown.wav")).status_code == 404
        assert (await anonymous.get("/oauth/callback")).status_code == 200
        wrong = await anonymous.post("/api/auth/login", json={"login": "anna", "password": "nope-nope"})
        assert wrong.status_code == 401

    async with _logged_in(app, "anna", "anna-secret") as anna:
        assert (await anna.get("/api/auth/me")).json()["role"] == "member"
        assert (await anna.get("/api/settings")).status_code == 403
        assert (await anna.get("/api/users")).status_code == 403


async def test_password_change_ends_other_sessions(app):
    password = await _password(app)
    async with (
        _logged_in(app, FIRST_ADMIN_LOGIN, password) as this,
        _logged_in(app, FIRST_ADMIN_LOGIN, password) as other,
    ):
        change = {"current": password, "new": "a-new-password"}
        assert (await this.post("/api/auth/password", json=change)).status_code == 204
        assert (await this.get("/api/auth/me")).status_code == 200
        assert (await other.get("/api/auth/me")).status_code == 401

        assert (await this.post("/api/auth/logout")).status_code == 204
        assert (await this.get("/api/auth/me")).status_code == 401


async def test_the_last_admin_stays(client):
    (admin,) = (await client.get("/api/users")).json()
    assert (await client.patch(f"/api/users/{admin['id']}", json={"role": "member"})).status_code == 409
    assert (await client.delete(f"/api/users/{admin['id']}")).status_code == 409

    second = {"login": "boss", "name": "Boss", "password": "boss-secret", "role": "admin"}
    assert (await client.post("/api/users", json=second)).status_code == 201
    assert (await client.patch(f"/api/users/{admin['id']}", json={"role": "member"})).status_code == 200


async def test_history_is_read_by_anyone_logged_in(app, client):
    await app.state.runtime.history.open_turn("kitchen", "c", datetime.now(UTC), None, "hello", 0.2, 1500)
    member = {"login": "anna", "name": "Anna", "password": "x"}
    assert (await client.post("/api/users", json=member)).status_code == 201

    async with _client(app) as anonymous:
        assert (await anonymous.get("/api/history")).status_code == 401
    async with _logged_in(app, "anna", "x") as anna:
        (turn,) = (await anna.get("/api/history", params={"device": "kitchen"})).json()
        assert (turn["text"], turn["answer"]) == ("hello", None)
        assert (await anna.get("/api/history", params={"device": "hall"})).json() == []
        assert (await anna.get("/api/history/devices")).json() == ["kitchen"]


async def test_a_member_sees_their_turns_and_unclaimed_ones(app, client):
    db = app.state.runtime.db
    for login in ("anna", "boris"):
        assert (
            await client.post("/api/users", json={"login": login, "name": login.title(), "password": "x"})
        ).status_code == 201
    async with db.session() as session, session.begin():
        voices = (
            (
                await session.execute(
                    sa.insert(Voice).returning(Voice.id),
                    [{"embedding": b"\0" * 8, "created_at": datetime.now(UTC)} for _ in range(2)],
                )
            )
            .scalars()
            .all()
        )
    history = app.state.runtime.history
    anna_turn = await history.open_turn("kitchen", "c", datetime.now(UTC), voices[0], "anna speaks", None, None)
    boris_turn = await history.open_turn("kitchen", "c", datetime.now(UTC), voices[1], "boris speaks", None, None)
    nobody = await history.open_turn("kitchen", "c", datetime.now(UTC), None, "short", None, None)

    async with _logged_in(app, "anna", "x") as anna, _logged_in(app, "boris", "x") as boris:
        assert (await anna.post(f"/api/history/{anna_turn}/mine")).status_code == 204
        assert (await boris.post(f"/api/history/{anna_turn}/mine")).status_code == 403
        assert (await boris.post(f"/api/history/{nobody}/mine")).status_code == 404
        assert (await boris.delete(f"/api/voices/{voices[1]}")).status_code == 403

        seen = {t["id"]: t for t in (await anna.get("/api/history")).json()}
        assert set(seen) == {anna_turn, boris_turn, nobody}
        assert seen[anna_turn]["person_name"] == "Anna" and seen[anna_turn]["voice_confirmed"]
        assert {t["id"] for t in (await boris.get("/api/history")).json()} == {boris_turn, nobody}
        (mine,) = (await anna.get("/api/voices")).json()
        assert (mine["id"], mine["turns"]) == (voices[0], 1)
        assert (await boris.patch(f"/api/voices/{voices[0]}", json={"unlink": True})).status_code == 403
    assert len((await client.get("/api/voices")).json()) == 2
    (only_anna,) = (await client.get("/api/history", params={"person_id": mine["person_id"]})).json()
    assert (only_anna["id"], only_anna["person_name"]) == (anna_turn, "Anna")


async def test_personal_servers_belong_to_their_person(app, client):
    for login in ("anna", "boris"):
        response = await client.post("/api/users", json={"login": login, "name": login.title(), "password": "x"})
        assert response.status_code == 201
    shared = {"name": "home", "url": "http://127.0.0.1:1/mcp"}
    assert (await client.put("/api/settings", json={"mcp_servers": [shared]})).status_code == 204
    mail = {"name": "mail", "url": "http://127.0.0.1:1/mail", "headers": {"Authorization": "Bearer secret"}}

    async with _logged_in(app, "anna", "x") as anna, _logged_in(app, "boris", "x") as boris:
        assert (await anna.post("/api/my/mcp-servers", json=shared)).status_code == 409
        created = (await anna.post("/api/my/mcp-servers", json=mail)).json()
        assert (await anna.post("/api/my/mcp-servers", json=mail)).status_code == 409
        assert (await boris.post("/api/my/mcp-servers", json=mail)).status_code == 201

        (mine,) = (await anna.get("/api/my/mcp-servers")).json()
        assert mine["headers"] == mail["headers"]
        assert (await boris.delete(f"/api/my/mcp-servers/{created['id']}")).status_code == 404
        moved = mail | {"url": "http://127.0.0.1:1/mail2"}
        assert (await anna.put(f"/api/my/mcp-servers/{created['id']}", json=moved)).status_code == 204
        assert (await anna.get("/api/mcp-servers/personal")).status_code == 403

    overview = (await client.get("/api/mcp-servers/personal")).json()
    assert sorted(s["owner"] for s in overview) == ["Anna", "Boris"]
    assert "headers" not in overview[0] and "url" not in overview[0]
    taken = await client.put("/api/settings", json={"mcp_servers": [shared, {"name": "mail", "url": "http://x/"}]})
    assert taken.status_code == 409


async def test_reminders_are_shown_and_cancelled_by_their_person(app, client):
    runtime = app.state.runtime
    for login in ("anna", "boris"):
        response = await client.post("/api/users", json={"login": login, "name": login.title(), "password": "x"})
        assert response.status_code == 201
    async with runtime.db.session() as session, session.begin():
        voice = (
            await session.execute(
                sa.insert(Voice).values(embedding=b"\0" * 8, created_at=datetime.now(UTC)).returning(Voice.id)
            )
        ).scalar_one()
    turn = await runtime.history.open_turn("kitchen", "c", datetime.now(UTC), voice, "remind me", None, None)
    later = datetime.now(UTC) + timedelta(days=1)
    hers = await runtime.reminders.add("kitchen", later, "stove", turn_id=turn)
    nobodys = await runtime.reminders.add("kitchen", later + timedelta(hours=1), "door")

    async with _logged_in(app, "anna", "x") as anna, _logged_in(app, "boris", "x") as boris:
        assert (await anna.post(f"/api/history/{turn}/mine")).status_code == 204
        assert [r["prompt"] for r in (await anna.get("/api/reminders")).json()] == ["stove", "door"]
        assert [r["prompt"] for r in (await boris.get("/api/reminders")).json()] == ["door"]
        assert (await boris.delete(f"/api/reminders/{hers.id}")).status_code == 403
        assert (await boris.delete(f"/api/reminders/{nobodys.id}")).status_code == 403
        assert (await anna.delete(f"/api/reminders/{hers.id}")).status_code == 204
    (left,) = (await client.get("/api/reminders")).json()
    assert (left["prompt"], left["person_name"]) == ("door", None)
    assert (await client.delete(f"/api/reminders/{nobodys.id}")).status_code == 204
