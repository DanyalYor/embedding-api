FROM python:3.13-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

COPY pyproject.toml ./

RUN poetry lock
RUN poetry install --only main --no-root --no-cache

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes --only main

RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.13-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
ENV MODEL_CACHE_DIR=/opt/models
ENV MODEL_NAME=intfloat/multilingual-e5-large
ENV PROVIDER=CPUExecutionProvider
ENV PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash app

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --chown=app:app src/ ./src/

USER app

EXPOSE 8000

CMD ["uvicorn", "embedding_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
