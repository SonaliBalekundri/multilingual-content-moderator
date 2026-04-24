# ============================================
# Multilingual Content Moderator - Dockerfile
# Multi-stage build to keep the image lean
# ============================================

# Stage 1: Install dependencies
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .

# Install CPU-only PyTorch first (skips 2.5GB of CUDA libraries)
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check using Python instead of curl (curl not in slim image)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# Run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]