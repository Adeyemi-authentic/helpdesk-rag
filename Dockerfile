# Container image for the Helpdesk RAG API + UI.
# Build context is the repo root (this folder):
#   docker build -t helpdesk-rag .
#   docker run -p 8000:8000 --env-file .env -e ENGINE=pg helpdesk-rag
# On Render: native Docker web service; $PORT is injected at runtime.

FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /code

# Install deps FIRST (own cached layer) so code edits don't re-run pip.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the engine (rag/) and the web app (app/).
COPY rag/ ./rag/
COPY app/ ./app/

# Production talks to an external pgvector store, not the local Qdrant files.
ENV ENGINE=pg

WORKDIR /code/app

# Bind 0.0.0.0 (reachable outside the container); default $PORT=8000 for local runs.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
