FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RSE_APPLY_RERANKING=false

COPY requirements-api.txt ./requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY agent ./agent
COPY agentic ./agentic
COPY multi_agent ./multi_agent
COPY api ./api
COPY full_text ./full_text
COPY ingestion ./ingestion
COPY retrieval ./retrieval
COPY shared ./shared
COPY tools ./tools
COPY data ./data
COPY .env.example ./

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
