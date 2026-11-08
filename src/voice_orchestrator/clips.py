import asyncio
import ipaddress
import logging
import socket
import struct
import time
import uuid
from collections.abc import AsyncIterator
from urllib.parse import urlparse

_log = logging.getLogger(__name__)

# Length fields of a WAV whose size is unknown while it is being streamed.
_UNKNOWN_SIZE = 0xFFFFFFFF


def wav_header(rate: int, width: int, channels: int) -> bytes:
    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", _UNKNOWN_SIZE),
            b"WAVEfmt ",
            struct.pack("<IHHIIHH", 16, 1, channels, rate, rate * width * channels, width * channels, width * 8),
            b"data",
            struct.pack("<I", _UNKNOWN_SIZE),
        ]
    )


class LiveClip:
    """PCM that is still being synthesized; readers get what exists and then wait for more."""

    def __init__(self) -> None:
        self._header: bytes | None = None
        self._format: tuple[int, int, int] | None = None
        self._chunks: list[bytes] = []
        self._closed = False
        self._changed = asyncio.Condition()
        self.closed_at: float | None = None

    async def set_format(self, rate: int, width: int, channels: int) -> None:
        fmt = (rate, width, channels)
        if self._format is not None and self._format != fmt:
            raise ValueError(f"TTS changed audio format mid-answer: {self._format} -> {fmt}")
        self._format = fmt
        self._header = wav_header(*fmt)

    async def write(self, pcm: bytes) -> None:
        if self._header is None:
            raise RuntimeError("set_format() must come before write()")
        if not pcm:
            return
        async with self._changed:
            self._chunks.append(pcm)
            self._changed.notify_all()

    async def close(self) -> None:
        async with self._changed:
            if not self._closed:
                self._closed = True
                self.closed_at = time.monotonic()
            self._changed.notify_all()

    async def wait_audio(self) -> bool:
        """True once the first audio is written, False if the clip closed without audio."""
        async with self._changed:
            await self._changed.wait_for(lambda: bool(self._chunks) or self._closed)
            return bool(self._chunks)

    async def read(self) -> AsyncIterator[bytes]:
        if not await self.wait_audio() or self._header is None:
            return
        yield self._header
        sent = 0
        while True:
            async with self._changed:
                await self._changed.wait_for(lambda n=sent: n < len(self._chunks) or self._closed)
                pending = self._chunks[sent:]
                finished = self._closed
            for chunk in pending:
                yield chunk
            sent += len(pending)
            if finished and sent == len(self._chunks):
                return


def source_address(host: str) -> str:
    """The local IP the kernel routes to host from, which is the address host can reach us at."""
    family, _, _, _, address = socket.getaddrinfo(host, 9, type=socket.SOCK_DGRAM)[0]
    with socket.socket(family, socket.SOCK_DGRAM) as sock:
        # Connecting a UDP socket sends nothing; it only selects the route and the source address.
        sock.connect(address)
        ip = sock.getsockname()[0]
    return f"[{ip}]" if family == socket.AF_INET6 else ip


def _is_loopback(url: str) -> bool:
    host = urlparse(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class ClipStore:
    def __init__(self, public_url: str | None, port: int, ttl_s: int) -> None:
        self._public_url = public_url.rstrip("/") if public_url else None
        self._port = port
        self._ttl = ttl_s
        self._clips: dict[str, LiveClip] = {}

    async def base_url_for(self, device_host: str) -> str:
        """Where a device downloads answers: the configured public URL, else our address as seen from it."""
        if self._public_url:
            if _is_loopback(self._public_url) and not _is_loopback(f"http://{device_host}"):
                _log.warning(
                    "VO_PUBLIC_URL %s is a loopback address: satellite %s cannot download answers from it",
                    self._public_url,
                    device_host,
                )
            return self._public_url
        return f"http://{await asyncio.to_thread(source_address, device_host)}:{self._port}"

    def create(self, base_url: str) -> tuple[LiveClip, str]:
        self._evict()
        clip_id = uuid.uuid4().hex
        clip = LiveClip()
        self._clips[clip_id] = clip
        return clip, f"{base_url}/tts/{clip_id}.wav"

    def get(self, clip_id: str) -> LiveClip | None:
        return self._clips.get(clip_id)

    def _evict(self) -> None:
        now = time.monotonic()
        expired = [k for k, c in self._clips.items() if c.closed_at is not None and now - c.closed_at > self._ttl]
        for clip_id in expired:
            del self._clips[clip_id]
