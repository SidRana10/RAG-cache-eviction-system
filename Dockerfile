# ---------- Builder stage ----------
FROM python:3.11-slim AS builder

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency files first for caching
COPY requirements.txt .

# Create virtualenv to isolate build artifacts
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# ---------- Runtime stage ----------
FROM python:3.11-slim AS runtime

# Create non-root user and app dir
RUN useradd --create-home appuser
WORKDIR /home/appuser/app

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --from=builder /app /home/appuser/app

# Set ownership and switch to non-root user
RUN chown -R appuser:appuser /home/appuser/app
USER appuser

EXPOSE 8501

# HUGGINGFACE
ENV HUGGINGFACE_MODEL_NAME=facebook/bart-large-cnn
ENV HUGGINGFACE_TASK=summarization
ENV HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV HUGGINGFACE_EVAL_MODEL=google/flan-t5-small

# PINECONE
ENV PINECONE_ENVIRONMENT=us-west1-gcp
ENV PINECONE_NAMESPACE=chatbot
ENV PINECONE_INDEX=Motocura


CMD ["uvicorn", "api.server:server", "--host", "0.0.0.0", "--port", "8501", "--loop", "uvloop", "--workers", "1"]