# ===========================================================================
# Multi-stage Dockerfile for Fraud Detection API
# ===========================================================================
# Stage 1 (builder): install Python dependencies into an isolated prefix
# Stage 2 (runtime): copy only the dependencies + app code into a slim image
#
# Build:   docker build -t fraud-detection-api .
# Run:     docker run -p 8000:8000 fraud-detection-api
# ===========================================================================

# ── Stage 1: Builder ──────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed by some ML packages (e.g. numpy, xgboost)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install all dependencies into /build/.local so we can copy them cleanly
RUN pip install --no-cache-dir --prefix=/build/.local -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Security: run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Copy only the installed packages from the builder stage
COPY --from=builder /build/.local /usr/local

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Create directories for reports and data (may be mounted as volumes)
RUN mkdir -p reports data/processed data/raw && \
    chown -R appuser:appuser /app

USER appuser

# Expose the FastAPI port
EXPOSE 8000

# Health check for Docker / orchestrators
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command: run FastAPI with Uvicorn
# --workers 2: Use 2 workers for better concurrency on multi-core machines
# --timeout-keep-alive 30: Keep connections alive for Prometheus scraping
CMD ["uvicorn", "src.predict:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--timeout-keep-alive", "30"]
