FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY agent ./agent
COPY agentic ./agentic
COPY multi_agent ./multi_agent
COPY full_text ./full_text
COPY ingestion ./ingestion
COPY retrieval ./retrieval
COPY shared ./shared
COPY tools ./tools
COPY data ./data
COPY ui ./ui
COPY .streamlit ./.streamlit

EXPOSE 8501

CMD ["sh", "-c", "streamlit run ui/multi_agent_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
