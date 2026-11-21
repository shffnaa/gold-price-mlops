FROM python:3.11-slim

WORKDIR /app

COPY docker/requirements-dashboard.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY app/dashboard/ ./app/dashboard/

EXPOSE 8501

CMD ["streamlit", "run", "app/dashboard/main.py", "--server.address=0.0.0.0", "--server.port=8501"]
