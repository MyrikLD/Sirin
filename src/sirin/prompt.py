from datetime import datetime

from jinja2 import Template
from jinja2.sandbox import ImmutableSandboxedEnvironment

# The prompt is edited on the settings page, so templates run sandboxed: no attribute access to
# internals and no way to call into the host.
_env = ImmutableSandboxedEnvironment()
# The same helper Home Assistant offers, so prompts written for HA render here unchanged.
_env.globals["now"] = lambda: datetime.now().astimezone()


def compile_prompt(source: str) -> Template:
    """Raises jinja2.TemplateSyntaxError for a broken template."""
    return _env.from_string(source)
