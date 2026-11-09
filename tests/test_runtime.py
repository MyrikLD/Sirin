from sirin.config import RuntimeSettings
from sirin.runtime import Runtime

# A closed local port: satellites stay disconnected, and nothing leaves the machine.
DEVICES = [{"name": "a", "host": "127.0.0.1", "port": 1}, {"name": "b", "host": "127.0.0.1", "port": 2}]


def _parts(runtime: Runtime) -> dict:
    services = runtime._services
    return {
        "agent": services.agent,
        "stt": services.stt,
        "tts": services.tts,
        "satellites": list(runtime._satellites),
    }


async def test_apply_rebuilds_only_what_changed(app_settings):
    runtime = Runtime(app_settings)
    try:
        await runtime.apply(RuntimeSettings.model_validate({"devices": DEVICES}))
        before = _parts(runtime)

        await runtime.apply(RuntimeSettings.model_validate({"devices": DEVICES, "vad": {"silence_ms": 400}}))
        after_vad = _parts(runtime)
        assert after_vad == before
        assert all(s.services.vad.silence_ms == 400 for s in runtime._satellites)

        changed = [DEVICES[0], {**DEVICES[1], "port": 3}]
        await runtime.apply(RuntimeSettings.model_validate({"devices": changed, "vad": {"silence_ms": 400}}))
        after_device = _parts(runtime)
        assert after_device["satellites"][0] is before["satellites"][0]
        assert after_device["satellites"][1] is not before["satellites"][1]
        assert after_device["agent"] is before["agent"]

        await runtime.reconnect_tools()
        assert _parts(runtime)["agent"] is not before["agent"]
        assert _parts(runtime)["satellites"] == after_device["satellites"]
    finally:
        await runtime.stop()
