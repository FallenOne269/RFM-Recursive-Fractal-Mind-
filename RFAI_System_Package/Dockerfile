FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RFAI_CONFIG_PATH=/app/config/default_config.json \
    RFAI_LOG_LEVEL=INFO

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "rfai_system.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
