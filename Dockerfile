# CareerPilot-AI Dockerfile
# Multi-stage build for production deployment

# Stage 1: Python dependencies
FROM python:3.11-slim as python-deps

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production build
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy Python dependencies from first stage
COPY --from=python-deps /root/.local /root/.local

# Add local user for security
RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

# Copy application code
COPY --chown=appuser:appuser backend/ ./backend/
COPY --chown=appuser:appuser ml/ ./ml/
COPY --chown=appuser:appuser knowledge_base/ ./knowledge_base/
COPY --chown=appuser:appuser datasets/ ./datasets/

# Add Python binaries to PATH
ENV PATH=/root/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]