import asyncio
import logging
import re
from collections.abc import AsyncIterator

from aioesphomeapi import APIClient, APIConnectionError
from aioesphomeapi import VoiceAssistantEventType as Ev
from pydantic import BaseModel, ConfigDict

from voice_orchestrator.agent import Agent
from voice_orchestrator.clips import ClipStore, LiveClip
from voice_orchestrator.config import VadConfig
from voice_orchestrator.speech import SpeechToText, TextToSpeech
from voice_orchestrator.vad import Endpointer, VadEvent

_log = logging.getLogger(__name__)

# The satellite sends audio every ~32 ms; a longer gap means it stopped streaming.
AUDIO_GAP_TIMEOUT_S = 3.0


class Services(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    stt: SpeechToText
    tts: TextToSpeech
    agent: Agent
    clips: ClipStore
    vad: VadConfig


class PipelineError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PipelineRun:
    """One wake-word-to-answer cycle, speaking the event protocol the satellite expects from HA.

    Every stage streams: microphone audio goes to the ASR while the user speaks, LLM text goes
    to the TTS while it is generated, and the device plays the answer URL while it is synthesized.
    """

    def __init__(
        self, client: APIClient, services: Services, conversation_id: str, log_name: str, audio_base_url: str
    ) -> None:
        self._client = client
        self._audio_base_url = audio_base_url
        self._s = services
        self._conversation_id = conversation_id
        self._log_name = log_name
        self._audio: asyncio.Queue[bytes | None] = asyncio.Queue()

    def feed(self, chunk: bytes) -> None:
        self._audio.put_nowait(chunk)

    def end_audio(self) -> None:
        self._audio.put_nowait(None)

    async def run(self) -> None:
        clip, url = self._s.clips.create(self._audio_base_url)
        # The URL is announced up front so the device can start playback before synthesis ends.
        self._send(Ev.VOICE_ASSISTANT_RUN_START, {"url": url})
        try:
            await self._run(clip, url)
        except PipelineError as exc:
            _log.warning("[%s] %s: %s", self._log_name, exc.code, exc)
            self._send(Ev.VOICE_ASSISTANT_ERROR, {"code": exc.code, "message": str(exc)})
        except Exception as exc:  # noqa: BLE001
            _log.exception("[%s] pipeline failed", self._log_name)
            self._send(Ev.VOICE_ASSISTANT_ERROR, {"code": "pipeline-error", "message": str(exc)[:200]})
        finally:
            await clip.close()
            self._send(Ev.VOICE_ASSISTANT_RUN_END)

    async def _run(self, clip: LiveClip, url: str) -> None:
        self._send(Ev.VOICE_ASSISTANT_STT_START)
        text = await self._listen()
        if not text:
            raise PipelineError("stt-no-text-recognized", "No text recognized")
        self._send(Ev.VOICE_ASSISTANT_STT_END, {"text": text})

        self._send(Ev.VOICE_ASSISTANT_INTENT_START)
        deltas: asyncio.Queue[str | None] = asyncio.Queue()
        answer: list[str] = []

        async def think() -> None:
            try:
                async for delta in self._s.agent.stream_reply(self._conversation_id, text):
                    answer.append(delta)
                    deltas.put_nowait(delta)
            finally:
                deltas.put_nowait(None)
            self._send(
                Ev.VOICE_ASSISTANT_INTENT_END,
                {"conversation_id": self._conversation_id, "continue_conversation": "0"},
            )
            if spoken := re.sub(r"\s+", " ", "".join(answer)).strip():
                self._send(Ev.VOICE_ASSISTANT_TTS_START, {"text": spoken})

        async def text_stream() -> AsyncIterator[str]:
            while (delta := await deltas.get()) is not None:
                yield delta

        async def start_playback() -> None:
            if await clip.wait_audio():
                _log.info("[%s] the device plays the answer from %s", self._log_name, url)
                self._send(Ev.VOICE_ASSISTANT_INTENT_PROGRESS, {"tts_start_streaming": "1"})

        async def speak() -> None:
            try:
                await self._s.tts.speak(text_stream(), clip)
            finally:
                await clip.close()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(think())
            tg.create_task(speak())
            tg.create_task(start_playback())

        if await clip.wait_audio():
            self._send(Ev.VOICE_ASSISTANT_TTS_END, {"url": url})

    async def _listen(self) -> str:
        endpointer = Endpointer(self._s.vad)
        async with self._s.stt.stream() as stt:
            while True:
                try:
                    chunk = await asyncio.wait_for(self._audio.get(), AUDIO_GAP_TIMEOUT_S)
                except TimeoutError:
                    chunk = None
                if chunk is None:
                    break
                await stt.send(chunk)
                event = endpointer.feed(chunk)
                if event is VadEvent.SPEECH_START:
                    self._send(Ev.VOICE_ASSISTANT_STT_VAD_START)
                elif event in (VadEvent.SPEECH_END, VadEvent.TIMEOUT):
                    break

            if not endpointer.heard_speech:
                raise PipelineError("stt-no-text-recognized", "No speech detected")
            self._send(Ev.VOICE_ASSISTANT_STT_VAD_END)
            return await stt.finish()

    def _send(self, event: Ev, data: dict[str, str] | None = None) -> None:
        _log.debug("[%s] -> %s %s", self._log_name, event.name, data or "")
        try:
            self._client.send_voice_assistant_event(event, data)
        except APIConnectionError:
            _log.debug("[%s] device gone, %s dropped", self._log_name, event.name)
