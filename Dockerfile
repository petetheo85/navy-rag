# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies for C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies into an isolated virtual environment
COPY pyproject.toml ./

# Bootstrap empty package so pip resolves and builds all third-party dependencies
RUN mkdir -p src/navy_rag && touch src/navy_rag/__init__.py && \
    python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir .

# Copy full application source code and install package without reinstalling dependencies
COPY src/ src/
RUN /opt/venv/bin/pip install --no-cache-dir --no-deps .

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

# Copy static vector database and authoritative directives
COPY data/manifests/ data/manifests/
COPY data/raw/ data/raw/
COPY chroma_db/ chroma_db/

# Copy application assets and runtime source code
COPY assets/ assets/
COPY src/ src/
COPY app.py ./

# Ensure runtime directories exist and have correct permissions
RUN mkdir -p logs && chown -R appuser:appuser /app

USER appuser

EXPOSE 1775

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail http://127.0.0.1:${PORT}/_stcore/health || exit 1

# Cloud Run injects PORT; Streamlit binds to 0.0.0.0 inside the container
ENTRYPOINT ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
