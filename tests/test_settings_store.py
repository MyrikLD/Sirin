import hashlib
import json

from mcp.shared.auth import OAuthToken

from sirin.config import McpServerConfig, RuntimeSettings
from sirin.legacy import import_files
from sirin.oauth import DbTokenStorage
from sirin.reminders import Reminders
from sirin.settings_store import SettingsStore

SERVER = {"name": "home", "url": "http://ha.local/mcp", "oauth": True}


async def test_url_change_drops_the_tokens_and_other_changes_keep_them(db):
    store = SettingsStore(db)
    await store.save(RuntimeSettings.model_validate({"mcp_servers": [SERVER]}))
    await DbTokenStorage(db, McpServerConfig(**SERVER)).set_tokens(OAuthToken(access_token="a"))

    renamed_header = SERVER | {"headers": {"X-A": "1"}}
    await store.save(RuntimeSettings.model_validate({"mcp_servers": [renamed_header]}))
    assert await DbTokenStorage(db, McpServerConfig(**renamed_header)).get_tokens() is not None

    moved = SERVER | {"url": "http://other.local/mcp"}
    await store.save(RuntimeSettings.model_validate({"mcp_servers": [moved]}))
    assert await DbTokenStorage(db, McpServerConfig(**moved)).get_tokens() is None
    assert (await store.load()).mcp_servers == [McpServerConfig(**moved)]


async def test_old_files_are_imported_once_and_kept_renamed(db, tmp_path):
    settings = RuntimeSettings.model_validate({"tts": {"voice": "xenia"}, "mcp_servers": [SERVER]})
    (tmp_path / "settings.json").write_text(settings.model_dump_json())
    key = hashlib.sha256(f"{SERVER['name']}\n{SERVER['url']}".encode()).hexdigest()[:16]
    (tmp_path / "oauth").mkdir()
    (tmp_path / "oauth" / f"{key}.json").write_text(
        json.dumps({"tokens": {"access_token": "a", "token_type": "Bearer"}, "expires_at": 4102444800})
    )
    reminder = {
        "id": "r1",
        "device": "kitchen",
        "due": "2100-01-01T08:00:00Z",
        "prompt": "stove",
        "created": "2026-01-01T00:00:00Z",
    }
    (tmp_path / "reminders.json").write_text(json.dumps([reminder]))

    await import_files(db, tmp_path)

    assert await SettingsStore(db).load() == settings
    tokens = await DbTokenStorage(db, McpServerConfig(**SERVER)).get_tokens()
    assert tokens is not None and tokens.access_token == "a" and tokens.expires_in > 0
    reminders = Reminders(db, deliver=None)
    await reminders.start()
    await reminders.stop()
    assert [r.prompt for r in reminders.pending("kitchen")] == ["stove"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "oauth.imported",
        "reminders.json.imported",
        "settings.json.imported",
    ]

    (tmp_path / "settings.json").write_text(RuntimeSettings().model_dump_json())
    await import_files(db, tmp_path)
    assert await SettingsStore(db).load() == settings
