# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies for C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies into an isolated virtual environment
COPY pyproject.toml ./
COPY src/ src/

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir .

# Runtime container
FROM python:3.11-slim AS runner

WORKDIR /app

# Install minimal runtime utilities (curl for container healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy prebuilt virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PORT=1775

# Run container as non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Copy application code, assets, data catalog, and populated vector database
COPY src/ src/
COPY assets/ assets/
COPY app.py ./
COPY data/manifests/ data/manifests/
COPY data/raw/ data/raw/
COPY chroma_db/ chroma_db/

# Ensure runtime directories exist and have correct permissions
RUN mkdir -p logs && chown -R appuser:appuser /app

USER appuser

EXPOSE 1775

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail http://127.0.0.1:${PORT}/_stcore/health || exit 1

# Cloud Run injects PORT; Streamlit binds to 0.0.0.0 inside the container
ENTRYPOINT ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
