import asyncio
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from aioesphomeapi import APIClient, APIConnectionError
from aioesphomeapi import VoiceAssistantEventType as Ev

from sirin.agent import Agent
from sirin.clips import ClipStore, LiveClip
from sirin.config import VadConfig
from sirin.speech import SpeechToText, TextToSpeech
from sirin.vad import Endpointer, SileroModel, VadEvent

_log = logging.getLogger(__name__)

# The satellite sends audio every ~32 ms; a longer gap means it stopped streaming.
AUDIO_GAP_TIMEOUT_S = 3.0


@dataclass(frozen=True)
class Services:
    stt: SpeechToText
    tts: TextToSpeech
    agent: Agent
    clips: ClipStore
    vad: VadConfig
    vad_model: SileroModel


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
        self, client: APIClient, services: Services, conversation_id: str, device: str, audio_base_url: str
    ) -> None:
        self._client = client
        self._audio_base_url = audio_base_url
        self._s = services
        self._conversation_id = conversation_id
        self._device = device
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
        except Exception as exc:
            # A stage that fails inside a TaskGroup arrives as a group; its first member is the cause.
            while isinstance(exc, ExceptionGroup):
                exc = exc.exceptions[0]
            if isinstance(exc, PipelineError):
                code = exc.code
                _log.warning("[%s] %s: %s", self._device, code, exc)
            else:
                code = "pipeline-error"
                _log.error("[%s] pipeline failed", self._device, exc_info=exc)
            self._send(Ev.VOICE_ASSISTANT_ERROR, {"code": code, "message": str(exc)[:200]})
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
                async for delta in self._s.agent.stream_reply(self._conversation_id, text, self._device):
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
                _log.info("[%s] the device plays the answer from %s", self._device, url)
                self._send(Ev.VOICE_ASSISTANT_INTENT_PROGRESS, {"tts_start_streaming": "1"})

        async with asyncio.TaskGroup() as tg:
            tg.create_task(think())
            tg.create_task(self._s.tts.speak(text_stream(), clip))
            tg.create_task(start_playback())

        if await clip.wait_audio():
            self._send(Ev.VOICE_ASSISTANT_TTS_END, {"url": url})

    async def _listen(self) -> str:
        endpointer = Endpointer(self._s.vad, self._s.vad_model)
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
        _log.debug("[%s] -> %s %s", self._device, event.name, data or "")
        try:
            self._client.send_voice_assistant_event(event, data)
        except APIConnectionError:
            _log.debug("[%s] device gone, %s dropped", self._device, event.name)
