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
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    SIRIN_DATA_DIR=/data \
    SIRIN_VAD_MODEL=/app/models/silero_vad.onnx
VOLUME /data
EXPOSE 8765
ENTRYPOINT ["sirin"]
