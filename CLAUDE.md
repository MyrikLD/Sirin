# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A voice pipeline for ESPHome satellites that replaces Home Assistant in the voice path. It connects to each
device over the ESPHome native API (as HA does), answers its `voice_assistant` events, and runs
STT (Wyoming) → LLM agent (pydantic-ai, OpenAI-compatible) with MCP tools → TTS (Wyoming) → WAV over HTTP to the
device's `media_player`. See README.md for the user-facing description.

## Commands

The Python environment is already active; do not use `uv run`.

```sh
pip install -e .              # install (src layout, hatchling)
voice-orchestrator -v         # run; -v enables DEBUG and protocol event logs
python -m voice_orchestrator  # same entry point
ruff check src && ruff format src   # lint/format (line-length 120)
docker compose up -d --build  # container; host networking, settings in ./data
```

CI (`.github/workflows/ci.yml`) runs `uv lock --check`, ruff and an import check, then builds the image for
amd64 and arm64 and pushes it to GHCR on `main` and `v*` tags. The image is pinned to Python 3.12 because
`pymicro-vad` has wheels only for that version.

There is no test suite. The settings page at `http://<host>:8765/` stores config in `data/settings.json`
(git-ignored); process-level config comes from `VO_*` env vars or `.env` (`AppSettings`).

## Architecture

Two config layers in `config.py`:
- `AppSettings` (env, read once at start): bind address, `public_url`, clip TTL, `data_dir`.
- `RuntimeSettings` (edited on the web page, persisted by `SettingsStore`): devices, STT, TTS, VAD, LLM, MCP servers.
  `PUT /api/settings` saves it and calls `Runtime.apply()`, which under a lock rebuilds only what changed, with no
  restart needed:
  - the `Agent` (and its MCP connections) when `llm` or `mcp_servers` changed;
  - `SpeechToText` / `TextToSpeech` when their block changed;
  - a `Satellite` when its own `DeviceConfig` changed. Unchanged satellites keep their connection and any dialog in
    progress and only receive the new `Services`.
  `Runtime.reconnect_tools()` rebuilds just the agent (used after an OAuth login).

Object graph built by `Runtime.apply()` (`runtime.py`): one `Agent` + `SpeechToText` + `TextToSpeech` +
the shared `ClipStore` are bundled into `Services` (`pipeline.py`), and one `Satellite` per device receives them.

Request flow for one utterance:
1. `Satellite` (`satellite.py`) holds an `aioesphomeapi` connection with `ReconnectLogic`. On the device's
   voice-assistant start it cancels any previous run (awaited, so `RUN_END` precedes the next `RUN_START`),
   creates a `PipelineRun`, and returns port `0` so audio comes over the API connection rather than UDP.
2. `PipelineRun` (`pipeline.py`) speaks HA's event protocol (`VoiceAssistantEventType`). It creates a `LiveClip`
   and announces its URL in `RUN_START` before anything is synthesized.
3. `_listen()` streams mic chunks to Wyoming ASR while `vad.Endpointer` decides end-of-utterance. The endpointer is
   a port of HA's microVAD segmenter; its constants mirror HA's and are intentionally not configurable.
4. `think()` / `speak()` / `start_playback()` run concurrently in a `TaskGroup`: agent text deltas flow through a
   queue into `TextToSpeech.speak()`, which writes PCM into the `LiveClip`; once the first audio exists,
   `INTENT_PROGRESS tts_start_streaming=1` tells the device to start downloading.
5. `GET /tts/{id}.wav` (`web/routers/tts.py`) streams the `LiveClip` with unknown-length WAV headers while it is
   still being written. `ClipStore.base_url_for()` picks the URL host as the local IP routed toward that device
   unless `VO_PUBLIC_URL` is set.

`Agent` (`agent.py`):
- Wraps pydantic-ai `run_stream_events`, yielding only text. It emits `"\n"` at each `TextPart` end so text
  said before a tool call is spoken while the tool runs.
- History is kept in memory per `conversation_id` with a TTL. Satellites that send no id use `device:<name>`.
- Each MCP server connection is held open in its own task (`_hold`), because a failed anyio transport leaks its
  cancel scope into the caller. `isolation.run_isolated` applies the same workaround for one-shot MCP calls.
  Keep MCP connects out of the caller's task.
- Tools are prefixed `<server>_`. A server that fails to connect is skipped and reported in
  `tool_status`, not raised.

Status (`status.py`): `Status.items[<settings list>][<item name>]` is an `ItemStatus` (`ok | pending | error` +
`detail`). `Satellite.status` and `Agent.tool_status` keep their own entries current, including an MCP connection
lost after startup, and call `StatusBoard.touch()` only on a real change. The page listens to
`GET /api/status/stream` (server-sent events: the full status now and after each change) and shows badges on list
items without knowing which list is which. On the exit signal `__main__._Server` closes the board so open settings
pages do not hold up shutdown.
- The system prompt is a Jinja template (`prompt.py`) re-rendered before each request.

`TextToSpeech` (`speech.py`) checks once whether the Wyoming server supports streaming input. If yes, it sends
deltas as `SynthesizeChunk`; if not, `split_sentences` cuts the text and each sentence is a separate `Synthesize`.

Web layer (`web/`): FastAPI app built in `create_app()`. Shared objects live on `app.state` and reach routes via
`Annotated` dependencies in `web/deps.py`. The static mount at `/` must stay last.

Settings page (`static/index.html`) is a single vanilla-JS file that renders forms from
`GET /api/settings/schema` (the `RuntimeSettings` JSON schema). It hard-codes no block or field name; behaviour comes
from the schema, so changing `config.py` changes the page:
- `title` on a `RuntimeSettings` field is the section heading; field `description`s become help text.
- `model_config = probed("/api/probe/...")` (`x-probe`) gives a block a Test button that POSTs its unsaved values
  there (`probes.py`).
- `json_schema_extra=CHOICES` (`x-choices`) makes a field a dropdown filled from that block's probe suggestions.
- `json_schema_extra=SECRET` (`format: password`) masks a field.
- `x-authorize` on a boolean field shows an Authorize button while the field is on (OAuth for MCP servers).

OAuth for HTTP MCP servers (`oauth.py`): `OAuthManager` runs the login flow from the settings page (redirect back
to `/oauth/callback` on the origin the page was opened from), stores tokens per server in `data/oauth/`, and on
success calls `on_authorized`, which reconnects the MCP servers (`Runtime.reconnect_tools()`). During normal runs
the provider refuses to start an interactive login and raises `OAuthRequired` instead.
