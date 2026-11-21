# syntax=docker/dockerfile:1
# Multi-stage build: locked deps -> seeded SQLite database -> slim non-root runtime (replay mode).
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM base AS runtime
RUN useradd --create-home --uid 10001 app && mkdir -p /app/var && chown app /app/var
COPY --from=builder /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    HYPERCARE_MODE=replay \
    HYPERCARE_DB_PATH=/app/var/hypercare.db \
    PORT=8501
USER app
# The database is rebuilt from the packaged seed data, so every deploy starts clean.
RUN hypercare seed
EXPOSE 8501
CMD ["hypercare", "serve"]
