FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV ENVIRONMENT=production
ENV MODEL_CACHE_DIR=/opt/models
ENV MODEL_NAME=intfloat/multilingual-e5-large
ENV PROVIDER=CPUExecutionProvider

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

COPY pyproject.toml poetry.lock ./

RUN poetry install --only main --no-root

COPY src/ ./src/

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "embedding_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
