# syntax=docker/dockerfile:1.7

# ---------------------------------------------------------------- builder
# Resolves and installs runtime dependencies into a self-contained virtualenv
# at /opt/venv. Dev dependencies are never installed here, so they cannot leak
# into the runtime image.
FROM python:3.12-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies before copying the source so that a code-only change
# does not invalidate the dependency layer.
COPY pyproject.toml ./
RUN uv venv "$VIRTUAL_ENV" --python 3.12 \
 && uv pip install --python "$VIRTUAL_ENV/bin/python" -r pyproject.toml

# Now add the application itself and install it (no extras).
COPY app ./app
COPY alembic.ini ./
COPY alembic ./alembic
RUN uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps .

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim AS runtime

# PORT is read by the CMD below; platforms like Cloud Run inject their own.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/venv \
    PORT=8000

# curl is needed by the container healthcheck. Nothing else is added.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder --chown=root:root /opt/venv /opt/venv
COPY --from=builder --chown=root:root /app/app /app/app
COPY --from=builder --chown=root:root /app/alembic /app/alembic
COPY --from=builder --chown=root:root /app/alembic.ini /app/alembic.ini

# Run unprivileged. The code is owned by root and only readable by the app user,
# so the running process cannot modify its own source.
USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

# sh -c so that ${PORT} is expanded at container start rather than build time.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
