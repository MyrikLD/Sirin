import pytest
from jinja2.exceptions import SecurityError
from pydantic import ValidationError

from sirin.config import LLMConfig, RuntimeSettings
from sirin.prompt import compile_prompt
from sirin.speech import split_sentences


async def _deltas(*parts: str):
    for part in parts:
        yield part


async def test_sentences_are_cut_at_punctuation_and_line_breaks():
    text = _deltas("Свет на кух", "не включён. Хочешь", ", ещё что-нибудь?\nГотово")
    assert [s async for s in split_sentences(text)] == ["Свет на кухне включён.", "Хочешь, ещё что-нибудь?", "Готово"]


def test_prompt_renders_the_home_assistant_now_helper():
    assert compile_prompt("{{ now().year }}").render().isdigit()


def test_prompt_template_is_sandboxed():
    with pytest.raises(SecurityError):
        compile_prompt("{{ ''.__class__.__mro__ }}").render()


def test_broken_prompt_template_is_rejected_on_save():
    with pytest.raises(ValidationError, match="template error"):
        LLMConfig(system_prompt="{{ now() }")


def test_device_names_must_be_unique():
    devices = [{"name": "a", "host": "h1"}, {"name": "a", "host": "h2"}]
    with pytest.raises(ValidationError, match="names must be unique"):
        RuntimeSettings.model_validate({"devices": devices})
