FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-mlflow.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
