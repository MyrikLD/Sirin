from contextlib import asynccontextmanager

import httpx2
import pytest

from sirin.web import create_app


@asynccontextmanager
async def _client(app_settings):
    app = create_app(app_settings)
    async with app.router.lifespan_context(app):
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.fixture
async def client(app_settings):
    async with _client(app_settings) as client:
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

    async with _client(app_settings) as restarted:
        assert (await restarted.get("/api/settings")).json()["tts"]["voice"] == "xenia"
