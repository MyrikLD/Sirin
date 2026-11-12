from contextlib import asynccontextmanager

import httpx2
import pytest

from sirin.accounts import FIRST_ADMIN_LOGIN, reset_password
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
