import asyncio

from sirin.clips import LiveClip


async def test_reader_gets_the_header_and_audio_written_while_it_reads():
    clip = LiveClip()
    clip.set_format(16000, 2, 1)
    await clip.write(b"\x01\x00" * 10)

    async def produce():
        await asyncio.sleep(0.01)
        await clip.write(b"\x02\x00" * 10)
        await clip.close()

    producer = asyncio.create_task(produce())
    data = b"".join([chunk async for chunk in clip.read()])
    await producer
    assert data[:4] == b"RIFF"
    assert data[44:] == b"\x01\x00" * 10 + b"\x02\x00" * 10


async def test_clip_closed_without_audio_has_nothing_to_play():
    clip = LiveClip()
    await clip.close()
    assert await clip.wait_audio() is False
    assert [chunk async for chunk in clip.read()] == []
