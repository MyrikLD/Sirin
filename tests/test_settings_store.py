import pytest
import sqlalchemy as sa
from mcp.shared.auth import OAuthToken
from pydantic import ValidationError

from sirin.config import LLMConfig, McpServerConfig, RuntimeSettings
from sirin.oauth import DbTokenStorage
from sirin.settings_store import SettingsStore
from sirin.tables import Setting

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


def test_the_active_model_is_one_of_the_list():
    two = [{"name": "local"}, {"name": "cloud", "base_url": "https://api.example/v1", "model": "big"}]
    assert LLMConfig.model_validate({"models": two}).current.name == "local"
    assert LLMConfig.model_validate({"models": two, "active": "cloud"}).current.model == "big"
    for bad in ({"models": two, "active": "gone"}, {"models": two + [{"name": "local"}]}, {"models": []}):
        with pytest.raises(ValidationError):
            LLMConfig.model_validate(bad)


async def test_a_stored_block_of_another_version_falls_back_to_defaults(db):
    async with db.session() as session, session.begin():
        await session.execute(sa.insert(Setting).values(name="llm", value={"base_url": "http://old/v1", "model": "x"}))
        await session.execute(sa.insert(Setting).values(name="tts", value={"voice": "xenia"}))
    settings = await SettingsStore(db).load()
    assert settings.llm == LLMConfig()
    assert settings.tts.voice == "xenia"
