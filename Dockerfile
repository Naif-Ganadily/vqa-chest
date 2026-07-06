# syntax=docker/dockerfile:1

# CPU image suitable for CI, batch inference, and reproducible runs.
# For GPU training, switch the base to an NVIDIA CUDA + Python image and install
# the matching CUDA build of torch instead of the default CPU wheel.
FROM python:3.12-slim

# Bring in the uv binary for fast, reproducible installs from uv.lock.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Install dependencies first (lockfiles only) for better layer caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# 2) Copy the source and install the project itself.
COPY src ./src
COPY entrypoint ./entrypoint
COPY config ./config
COPY tests ./tests
RUN uv sync --frozen --no-dev

# Default shows the training CLI help. Override the command to train / predict / test:
#   docker run --rm IMAGE uv run train   --config config/prod.yaml
#   docker run --rm IMAGE uv run predict --config config/prod.yaml --checkpoint data/05-models/prod-run.pt
#   docker run --rm IMAGE uv run pytest -m "not slow"
CMD ["uv", "run", "train", "--help"]
