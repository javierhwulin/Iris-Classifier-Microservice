# -- Stage 1: Builder ---
FROM python:3.12-slim AS builder

# Install system dependencies for uv
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && curl -LsSf https://astral.sh/uv/install.sh | sh \
 && cp ~/.local/bin/uv /usr/local/bin/uv

# Create project venv and install Python dependencies
WORKDIR /src
COPY pyproject.toml uv.lock ./

ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN uv venv --python 3.12 .venv \
    && . .venv/bin/activate \
    && uv sync --no-dev

# -- Stage 2: Runtime ---
FROM python:3.12-slim AS runtime

# Install minimal system deps for healthcheck
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd -r app && useradd -r -g app app

WORKDIR /app

# Copy the venv from builder
COPY --from=builder /src/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Copy application code and model artifacts
COPY app app
COPY models models

# Switch to non-root user
USER app

EXPOSE 80

# Healthcheck endpoint
HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost/health || exit 1

# Entrypoint
CMD ["/app/.venv/bin/python", "-m", "uvicorn", \
    "app.main:create_app", "--host", "0.0.0.0", "--port", "80"]