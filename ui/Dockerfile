FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY ui ./ui
COPY shared ./shared
COPY .streamlit ./.streamlit

EXPOSE 8501

CMD ["sh", "-c", "streamlit run ui/streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
