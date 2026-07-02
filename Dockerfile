# syntax=docker/dockerfile:1

# =========================================================
# Build stage – resolve & install dependencies with uv,
# using pyproject.toml / uv.lock already in the repo.
# =========================================================
FROM python:3.13-slim AS builder

# uv binary, copied straight from its official image (no pip bootstrap needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System build deps needed by tensorflow/torch wheels & scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 1) Install deps only first -> this layer stays cached unless
#    pyproject.toml / uv.lock change, even if source code changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 2) Now copy the rest of the source and install the project itself
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# =========================================================
# Runtime stage – slim image, no compilers, non-root user
# =========================================================
FROM python:3.13-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Bring over the fully-built venv + application code from the builder
COPY --from=builder --chown=appuser:appuser /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

# 8000 = FastAPI (uvicorn), 8501 = Streamlit dashboard
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# This image is shared by both services in docker-compose.yml, which override
# this command for the Streamlit service. Running the container directly
# (docker run) defaults to the FastAPI app.
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
