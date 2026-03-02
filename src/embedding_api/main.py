import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from embedding_api.config import settings
from embedding_api.data_models import EmbedRequest, EmbedResponse
from embedding_api.logger import configure_logging
from embedding_api.services import EmbeddingService

configure_logging()

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    start = time.time()
    logger.info("app_starting", model=settings.model_name, provider=settings.provider)

    try:
        app.state.service = EmbeddingService()
        duration = (time.time() - start) * 1000
        logger.info("model_loaded", duration_ms=round(duration, 2))
    except Exception as e:
        logger.error("model_load_failed", error=str(e), exc_info=True)
        raise

    yield

    logger.info("app_shutting_down")
    app.state.service = None


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    path = request.url.path
    method = request.method
    client_ip = request.client.host if request.client else "unknown"

    logger.debug("request_received", path=path, method=method, client_ip=client_ip)

    try:
        response = await call_next(request)
        duration = (time.time() - start) * 1000
        logger.info(
            "request_completed",
            path=path,
            method=method,
            status_code=response.status_code,
            duration_ms=round(duration, 2),
        )
        return response
    except Exception as e:
        duration = (time.time() - start) * 1000
        logger.error(
            "request_failed",
            path=path,
            method=method,
            error_type=type(e).__name__,
            duration_ms=round(duration, 2),
            exc_info=True,
        )
        raise


@app.get("/health")
async def health_check() -> JSONResponse:
    return JSONResponse(content={"status": "healthy"}, status_code=200)


@app.get("/ready")
async def readiness_check() -> JSONResponse:
    ready = hasattr(app.state, "service") and app.state.service is not None
    logger.debug("readiness_check", ready=ready)
    return JSONResponse(
        content={"status": "ready" if ready else "not_ready"},
        status_code=200 if ready else 503,
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(body: EmbedRequest, request: Request) -> EmbedResponse:
    """Embedding endpoint taking a list of texts to embed as well as the task type.

    Args:
        texts: list of strings to embed.
        task_type: 'query' for search queries, 'passage' for documents.

    Returns:
        Embeddings as a list of 1024-dimensional vectors.

    Raises:
        422: if batch size exceed configured limit to avoid overflow.
        500: if generation of embeddings fails.

    """
    service = app.state.service
    if service is None:
        logger.error("inference_attempted_without_model")
        raise HTTPException(status_code=503, detail="Model service not ready")

    client_ip = request.client.host if request.client else "unknown"
    logger.debug(
        "inference_started",
        batch_size=len(body.texts),
        task_type=body.task_type,
        client_ip=client_ip,
    )

    if len(body.texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=f"Request contained {len(body.texts)} which exceeds max_batch_size of {settings.max_batch_size}"
        )

    try:
        start = time.time()
        embeddings = service.embed(body.texts, body.task_type)
        duration = (time.time() - start) * 1000

        logger.info(
            "inference_completed",
            batch_size=len(body.texts),
            embedding_dim=len(embeddings[0]) if embeddings else 0,
            duration_ms=round(duration, 2),
        )

        return EmbedResponse(embeddings=embeddings, model=service.model_name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "inference_failed",
            error_type=type(e).__name__,
            error=str(e),
            batch_size=len(body.texts),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Inference error") from e
