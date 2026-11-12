# syntax=docker/dockerfile:1
FROM python:3.14-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.14-slim
# Silero VAD, pinned release; the checksum keeps a changed or truncated download out of the image.
ADD --checksum=sha256:1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3 \
    https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/src/silero_vad/data/silero_vad.onnx \
    /app/models/silero_vad.onnx
# Voice models: WeSpeaker ResNet34 for who speaks, VoxLock (pinned revision) to keep only that voice.
ADD --checksum=sha256:e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012 \
    https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx \
    /app/models/voices/speaker.onnx
ADD --checksum=sha256:81f305c9b7a32f13d1401f53bb4c6301826a607c9e807e60912f36ef1b9fae29 \
    https://huggingface.co/arbazsiddiqui/voxlock/resolve/2413b292bc2c337bfc19a9802aaf3e011e68e9c4/onnx/v2/enroll.onnx \
    /app/models/voices/enroll.onnx
ADD --checksum=sha256:c50ffdc8d723fd4f91cb08889f3d0143491f95967b705f35c444a0b4afe88172 \
    https://huggingface.co/arbazsiddiqui/voxlock/resolve/2413b292bc2c337bfc19a9802aaf3e011e68e9c4/onnx/v2/chunk_int8.onnx \
    /app/models/voices/chunk.onnx
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    SIRIN_DATA_DIR=/data \
    SIRIN_VAD_MODEL=/app/models/silero_vad.onnx \
    SIRIN_VOICE_MODELS=/app/models/voices
VOLUME /data
EXPOSE 8765
ENTRYPOINT ["sirin"]
