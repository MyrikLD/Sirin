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
sirin -v                      # run; -v enables DEBUG and protocol event logs
python -m sirin               # same entry point
ruff check src tests && ruff format src tests   # lint/format (line-length 120)
pytest                        # basic offline tests (tests/), about a second
docker compose up -d --build  # container; bridge network, needs SIRIN_PUBLIC_URL in .env, settings in ./data
```

CI (`.github/workflows/ci.yml`) runs `uv lock --check`, ruff, an import check and pytest, then builds the image for
amd64 and arm64 and pushes it to GHCR on `main` and `v*` tags. Python 3.14 only.

Tests cover a few basics only (sentence splitting, prompt sandbox, settings validation, live clips, VAD timeout,
MCP isolation and reconnect, history cap, partial rebuild, settings API, reminder scheduling and tools). Tests need network once: `conftest.py`
downloads the VAD model into the pytest cache.

The settings page at `http://<host>:8765/` stores config in `data/settings.json` (git-ignored); process-level config
comes from `SIRIN_*` env vars or `.env` (`AppSettings`).

## Architecture

Two config layers in `config.py`:
- `AppSettings` (env, read once at start): bind address, `public_url`, clip TTL, `data_dir`, `vad_model`.
- `RuntimeSettings` (edited on the web page, persisted by `SettingsStore`): devices, STT, TTS, VAD, LLM, MCP servers.
  `PUT /api/settings` calls `Runtime.apply()`, which saves it and under a lock rebuilds only what changed, with no
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
   HA's segmenter on top of the Silero VAD ONNX model, run with onnxruntime: one `SileroModel` per process,
   recurrent state per utterance. The code does not download the model: the Dockerfile adds it to the image
   (pinned release, SHA-256 checked) and sets `SIRIN_VAD_MODEL`; `Runtime` loads it at start and fails if the file is
   missing. Its timings mirror HA's and its thresholds are Silero's defaults; they are intentionally not configurable.
4. `think()` / `speak()` / `start_playback()` run concurrently in a `TaskGroup`: agent text deltas flow through a
   queue into `TextToSpeech.speak()`, which writes PCM into the `LiveClip`; once the first audio exists,
   `INTENT_PROGRESS tts_start_streaming=1` tells the device to start downloading.
5. `GET /tts/{id}.wav` (`web/routers/tts.py`) streams the `LiveClip` with unknown-length WAV headers while it is
   still being written. `ClipStore.base_url_for()` picks the URL host as the local IP routed toward that device
   unless `SIRIN_PUBLIC_URL` is set.

`Agent` (`agent.py`):
- Wraps pydantic-ai `run_stream_events`, yielding only text. It emits `"\n"` at each `TextPart` end so text
  said before a tool call is spoken while the tool runs.
- History is kept in memory per `conversation_id` with a TTL and capped at `MAX_HISTORY_TURNS` user turns.
  Satellites that send no id use `device:<name>`.
- `build_toolset()` returns an `IsolatedToolset` (`isolation.py`): the MCP connection is opened and held in a task
  of its own, because a failed anyio transport leaks its cancel scope into the task that opened it. Entering is
  re-entrant (the agent holds each toolset open; every agent run enters it again without reconnecting). Open MCP
  servers only through `build_toolset()` / `list_server_tools()`.
- One `_hold()` task per server keeps it connected for the agent's life: a server that fails to connect or loses
  its connection is reported in `tool_status`, not raised, and tried again with a doubling pause. A tool call that
  finds the server gone (`httpx2.TransportError`) counts as a lost connection and the model gets `ToolFailed`: the
  MCP client does not recover by itself after the server restarts. Each run gets the
  toolsets connected at that moment (`run_stream_events(toolsets=...)`), MCP first and built-in tools last.
- Tools are prefixed `<server>_`.

Reminders (`reminders.py`): built-in agent tools (`reminder_tools()`, passed to `Agent` with no prefix) read the
device from `TurnDeps`, the pydantic-ai deps of every run. The LLM writes a prompt for itself; `Reminders` (owned by
`Runtime`, so it outlives agent rebuilds) keeps them in `data/reminders.json` and at the due time calls
`Runtime._deliver_reminder()` → `Satellite.announce()`: the agent answers the prompt in the `device:<name>`
conversation, TTS writes a `LiveClip`, and the device plays it via the voice_assistant announce API (`ANNOUNCE`
feature). An offline device is retried for up to `GRACE`; reminders missed while the service was down are dropped at
start. A recurring reminder has a cron `schedule` (`cronsim`, read in local wall time by `cron_next()`) and an optional
`until`; it keeps its id and its one task: after each occurrence (announced or dropped) `_run` moves `due` to the next
time, and at start a missed one moves forward instead of being dropped. Times are the process's local zone.

Status (`status.py`): `Status.items[<settings list>][<item name>]` is an `ItemStatus` (`ok | pending | error` +
`detail`). `Satellite.status` and `Agent.tool_status` keep their own entries current, including an MCP connection
lost after startup, and call `StatusBoard.touch()` only on a real change. The page listens to
`GET /api/status/stream` (server-sent events: the full status now and after each change) and shows badges on list
items without knowing which list is which. On the exit signal `__main__._Server` closes the board so open settings
pages do not hold up shutdown.
- The system prompt is a Jinja template (`prompt.py`) re-rendered before each request.

`TextToSpeech` (`speech.py`) checks once whether the Wyoming server supports streaming input. If yes, it sends
deltas as `SynthesizeChunk`; if not, `split_sentences` cuts the text and each sentence is a separate `Synthesize`.

Web layer (`web/`): FastAPI app built in `create_app()`. `Runtime` is the only object on `app.state`; routes reach
it through `RuntimeDep` (`web/deps.py`) and use its `clips`, `oauth` and `board`. The static mount at `/` must stay
last.

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
success calls back into `Runtime`, which reconnects the MCP servers (`Runtime.reconnect_tools()`). During normal runs
the provider refuses to start an interactive login and raises `OAuthRequired` instead.
