from mcp.shared.auth import OAuthToken

from sirin.config import McpServerConfig, RuntimeSettings
from sirin.oauth import DbTokenStorage
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
