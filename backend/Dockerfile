FROM python:3.11-slim

WORKDIR /app

# Install system deps needed for psycopg2 build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up user and HuggingFace cache directory
ENV HF_HOME=/home/steamiq/.cache/huggingface
RUN adduser --disabled-password --gecos "" steamiq && \
    mkdir -p /home/steamiq/.cache/huggingface && \
    chown -R steamiq:steamiq /home/steamiq /app

# Copy and install Python deps from pyproject.toml first (layer cache)
COPY pyproject.toml README.md ./
RUN pip install --default-timeout=120 --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu .

# Pre-download sentence transformer model into HF_HOME
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    chown -R steamiq:steamiq /home/steamiq/.cache

# Copy application code
COPY . .
RUN chown -R steamiq:steamiq /app

USER steamiq

# Expose port
EXPOSE 8000

# Health check — polls /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run migrations then start server
CMD ["sh", "-c", "alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port 8000"]
