import asyncio
import logging
import re
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

from wyoming.asr import Transcribe, Transcript, TranscriptChunk
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeVoice,
)

from voice_orchestrator.clips import LiveClip
from voice_orchestrator.config import SttConfig, TtsConfig
from voice_orchestrator.vad import SAMPLE_RATE, SAMPLE_WIDTH

_log = logging.getLogger(__name__)

_MIC_FORMAT = {"rate": SAMPLE_RATE, "width": SAMPLE_WIDTH, "channels": 1}
# Sentence end: terminal punctuation followed by whitespace, or a line break.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")


class WyomingError(RuntimeError):
    pass


async def _read(client: AsyncTcpClient, what: str) -> Event:
    event = await client.read_event()
    if event is None:
        raise WyomingError(f"{what} closed the connection")
    if Error.is_type(event.type):
        raise WyomingError(f"{what} error: {Error.from_event(event).text}")
    return event


async def describe(host: str, port: int) -> Info:
    async with AsyncTcpClient(host, port) as client:
        await client.write_event(Describe().event())
        while True:
            event = await _read(client, "Wyoming server")
            if Info.is_type(event.type):
                return Info.from_event(event)


class SttStream:
    def __init__(self, client: AsyncTcpClient) -> None:
        self._client = client

    async def send(self, pcm: bytes) -> None:
        await self._client.write_event(AudioChunk(audio=pcm, **_MIC_FORMAT).event())

    async def finish(self) -> str:
        await self._client.write_event(AudioStop().event())
        while True:
            event = await _read(self._client, "ASR")
            if TranscriptChunk.is_type(event.type):
                _log.debug("partial transcript %r", TranscriptChunk.from_event(event).text)
            elif Transcript.is_type(event.type):
                return Transcript.from_event(event).text.strip()


class SpeechToText:
    """Microphone audio goes to the ASR as it arrives, so recognition overlaps with speech."""

    def __init__(self, cfg: SttConfig) -> None:
        self._cfg = cfg

    @asynccontextmanager
    async def stream(self) -> AsyncGenerator[SttStream]:
        async with AsyncTcpClient(self._cfg.host, self._cfg.port) as client:
            await client.write_event(Transcribe(language=self._cfg.language).event())
            await client.write_event(AudioStart(**_MIC_FORMAT).event())
            yield SttStream(client)


class TextToSpeech:
    """Speaks a stream of text deltas into a LiveClip while the text is still being generated.

    Servers with streaming input get the deltas directly; for the others the text is cut into
    sentences and each sentence is synthesized as soon as it is complete.
    """

    def __init__(self, cfg: TtsConfig) -> None:
        self._cfg = cfg
        self._voice = SynthesizeVoice(name=cfg.voice) if cfg.voice else None
        self._server_streams: bool | None = None

    async def speak(self, text: AsyncIterator[str], clip: LiveClip) -> None:
        if await self._supports_streaming():
            await self._speak_streaming(text, clip)
        else:
            async for sentence in split_sentences(text):
                await self._speak_sentence(sentence, clip)

    async def _supports_streaming(self) -> bool:
        if self._server_streams is None:
            info = await describe(self._cfg.host, self._cfg.port)
            self._server_streams = any(p.supports_synthesize_streaming for p in info.tts if p.installed)
            _log.info("TTS input streaming: %s", self._server_streams)
        return self._server_streams

    async def _speak_streaming(self, text: AsyncIterator[str], clip: LiveClip) -> None:
        async with AsyncTcpClient(self._cfg.host, self._cfg.port) as client:
            reader = asyncio.create_task(self._pump_audio(client, clip, until=SynthesizeStopped))
            try:
                await client.write_event(SynthesizeStart(voice=self._voice).event())
                async for delta in text:
                    await client.write_event(SynthesizeChunk(text=delta).event())
                await client.write_event(SynthesizeStop().event())
                await reader
            finally:
                reader.cancel()

    async def _speak_sentence(self, sentence: str, clip: LiveClip) -> None:
        async with AsyncTcpClient(self._cfg.host, self._cfg.port) as client:
            await client.write_event(Synthesize(text=sentence, voice=self._voice).event())
            await self._pump_audio(client, clip, until=AudioStop)

    async def _pump_audio(self, client: AsyncTcpClient, clip: LiveClip, until: type) -> None:
        while True:
            event = await _read(client, "TTS")
            if AudioStart.is_type(event.type):
                start = AudioStart.from_event(event)
                await clip.set_format(start.rate, start.width, start.channels)
            elif AudioChunk.is_type(event.type):
                await clip.write(AudioChunk.from_event(event).audio)
            if until.is_type(event.type):
                return


async def split_sentences(text: AsyncIterator[str]) -> AsyncIterator[str]:
    buffer = ""
    async for delta in text:
        buffer += delta
        parts = _SENTENCE_END.split(buffer)
        buffer = parts.pop()
        for sentence in parts:
            if sentence.strip():
                yield sentence.strip()
    if buffer.strip():
        yield buffer.strip()
