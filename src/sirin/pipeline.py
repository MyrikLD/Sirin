import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime

from aioesphomeapi import APIClient, APIConnectionError
from aioesphomeapi import VoiceAssistantEventType as Ev

from sirin.agent import Agent, TurnDeps
from sirin.clips import ClipStore, LiveClip
from sirin.config import VadConfig
from sirin.history import History
from sirin.speech import SpeechToText, TextToSpeech
from sirin.vad import SAMPLE_RATE, Endpointer, SileroModel, VadEvent
from sirin.voices import SpeakerGate, Voices

_log = logging.getLogger(__name__)

# The satellite sends audio every ~32 ms; a longer gap means it stopped streaming.
AUDIO_GAP_TIMEOUT_S = 3.0
# What the agent gets as the user's turn when its question was met with silence.
SILENCE = "(The user did not answer.)"


@dataclass(frozen=True)
class Services:
    stt: SpeechToText
    tts: TextToSpeech
    agent: Agent
    clips: ClipStore
    vad: VadConfig
    vad_model: SileroModel
    # None when the voice models are not installed: nobody is recognized and all audio is recognized as it comes.
    voices: Voices | None = None
    history: History | None = None


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
        self,
        client: APIClient,
        services: Services,
        conversation_id: str,
        device: str,
        audio_base_url: str,
        answers_question: bool = False,
    ) -> None:
        """answers_question: the device listens because the previous answer ended with a question."""
        self._client = client
        self._audio_base_url = audio_base_url
        self._s = services
        self._conversation_id = conversation_id
        self._device = device
        self._answers_question = answers_question
        # Set once the answer asks the user something, so the device listens again without the wake word.
        self.awaits_reply = False
        self._audio: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._turn = TurnDeps(device=device)
        # What the history keeps of this run.
        self._started = time.monotonic()
        self._started_at = datetime.now(UTC)
        self._timings: dict[str, int] = {}
        self._similarity: float | None = None
        self._spoken = ""
        self._error: str | None = None

    def feed(self, chunk: bytes) -> None:
        self._audio.put_nowait(chunk)

    def end_audio(self) -> None:
        self._audio.put_nowait(None)

    async def run(self) -> None:
        self._started, self._started_at = time.monotonic(), datetime.now(UTC)
        clip, url = self._s.clips.create(self._audio_base_url)
        # The URL is announced up front so the device can start playback before synthesis ends.
        self._send(Ev.VOICE_ASSISTANT_RUN_START, {"url": url})
        try:
            await self._run(clip, url)
        except asyncio.CancelledError:
            self._error = "cancelled: the device started another run or stopped this one"
            raise
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
            self._error = f"{code}: {exc}"
        finally:
            self._mark("done")
            if self._s.history is not None and self._turn.turn_id is not None:
                self._s.history.close_turn(
                    self._turn.turn_id,
                    self._turn.voice_id,
                    self._spoken,
                    self._turn.calls,
                    self._timings,
                    self._error,
                    self._turn.speech,
                )
            await clip.close()
            self._send(Ev.VOICE_ASSISTANT_RUN_END)

    def _mark(self, stage: str) -> None:
        """Times stage from the start of the run; the first time counts."""
        self._timings.setdefault(stage, round((time.monotonic() - self._started) * 1000))

    async def _run(self, clip: LiveClip, url: str) -> None:
        self._send(Ev.VOICE_ASSISTANT_STT_START)
        heard = await self._listen()
        self._mark("heard")
        if heard:
            self._send(Ev.VOICE_ASSISTANT_STT_END, {"text": heard})
        elif not self._answers_question:
            raise PipelineError("stt-no-text-recognized", "No text recognized")
        # Silence after a question is an answer too: the agent decides what follows from it.
        text = heard or SILENCE
        if self._s.history is not None:
            speech = self._turn.speech
            turn_id = await self._s.history.open_turn(
                self._device,
                self._conversation_id,
                self._started_at,
                self._turn.voice_id,
                heard,
                self._similarity,
                round(len(speech) * 1000 / SAMPLE_RATE) if speech is not None else None,
            )
            self._turn = self._turn.model_copy(update={"turn_id": turn_id})

        self._send(Ev.VOICE_ASSISTANT_INTENT_START)
        deltas: asyncio.Queue[str | None] = asyncio.Queue()
        answer: list[str] = []

        def say(delta: str) -> None:
            if delta.strip():
                self._mark("first_text")
            answer.append(delta)
            deltas.put_nowait(delta)

        async def think() -> None:
            try:
                async for delta in self._s.agent.stream_reply(self._conversation_id, text, self._turn):
                    say(delta)
            except Exception as exc:
                if not self._s.agent.error_reply:
                    raise
                _log.exception("[%s] the agent failed, the device says the error reply", self._device)
                self._error = f"agent: {type(exc).__name__}: {exc}"
                # The newline ends whatever sentence the failure interrupted.
                say("\n")
                say(self._s.agent.error_reply)
            finally:
                deltas.put_nowait(None)
            self._spoken = spoken = re.sub(r"\s+", " ", "".join(answer)).strip()
            # As in Home Assistant: after an answer that ends with a question the device listens again
            # without the wake word. A question that follows silence is not waited on, or an empty room
            # would be asked forever.
            self.awaits_reply = bool(heard) and spoken.endswith(("?", "？"))
            self._send(
                Ev.VOICE_ASSISTANT_INTENT_END,
                {
                    "conversation_id": self._conversation_id,
                    "continue_conversation": "1" if self.awaits_reply else "0",
                },
            )
            if spoken:
                self._send(Ev.VOICE_ASSISTANT_TTS_START, {"text": spoken})

        async def text_stream() -> AsyncIterator[str]:
            while (delta := await deltas.get()) is not None:
                yield delta

        async def start_playback() -> None:
            if await clip.wait_audio():
                self._mark("first_audio")
                _log.info("[%s] the device plays the answer from %s", self._device, url)
                self._send(Ev.VOICE_ASSISTANT_INTENT_PROGRESS, {"tts_start_streaming": "1"})

        async with asyncio.TaskGroup() as tg:
            tg.create_task(think())
            tg.create_task(self._s.tts.speak(text_stream(), clip))
            tg.create_task(start_playback())

        if await clip.wait_audio():
            self._send(Ev.VOICE_ASSISTANT_TTS_END, {"url": url})

    async def _listen(self) -> str:
        """The recognized text; empty when a question was met with silence."""
        endpointer = Endpointer(self._s.vad, self._s.vad_model)
        gate = SpeakerGate(self._s.voices, self._s.vad_model) if self._s.voices is not None else None
        async with self._s.stt.stream() as stt:

            async def hear(audio: bytes) -> bool:
                """True once the utterance is over."""
                if not audio:
                    return False
                await stt.send(audio)
                event = endpointer.feed(audio)
                if event is VadEvent.SPEECH_START:
                    self._send(Ev.VOICE_ASSISTANT_STT_VAD_START)
                return event in (VadEvent.SPEECH_END, VadEvent.TIMEOUT)

            while True:
                try:
                    chunk = await asyncio.wait_for(self._audio.get(), AUDIO_GAP_TIMEOUT_S)
                except TimeoutError:
                    chunk = None
                if chunk is None:
                    if gate is not None:
                        await hear(await asyncio.to_thread(gate.finish))
                    break
                # The networks behind the gate take a good part of real time; a thread keeps the event loop free.
                if await hear(await asyncio.to_thread(gate.feed, chunk) if gate is not None else chunk):
                    break

            if gate is not None:
                speaker = await self._s.voices.person_name(gate.speaker)
                self._turn = TurnDeps(device=self._device, voice_id=gate.speaker, speaker=speaker, speech=gate.speech)
                seconds = len(gate.speech) / SAMPLE_RATE
                if gate.closest is None:
                    _log.info("[%s] the voice was not compared: %.1f s of speech", self._device, seconds)
                else:
                    voice_id, score, ms = gate.closest
                    self._similarity = score
                    who = f"voice {voice_id}" + (f" ({speaker})" if speaker else "")
                    verdict = f"is {who}" if gate.speaker else f"is not recognized, the closest is voice {voice_id}"
                    _log.info(
                        "[%s] the voice %s (similarity %.2f on %.1f s; %.1f s of speech in all)",
                        self._device,
                        verdict,
                        score,
                        ms / 1000,
                        seconds,
                    )
            if not endpointer.heard_speech and not self._answers_question:
                raise PipelineError("stt-no-text-recognized", "No speech detected")
            self._send(Ev.VOICE_ASSISTANT_STT_VAD_END)
            return await stt.finish() if endpointer.heard_speech else ""

    def _send(self, event: Ev, data: dict[str, str] | None = None) -> None:
        _log.debug("[%s] -> %s %s", self._device, event.name, data or "")
        try:
            self._client.send_voice_assistant_event(event, data)
        except APIConnectionError:
            _log.debug("[%s] device gone, %s dropped", self._device, event.name)
