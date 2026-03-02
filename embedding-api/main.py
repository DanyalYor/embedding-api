from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from contextlib import asynccontextmanager
import logging
import httpx
from pydantic import BaseModel, field_validator

logger = logging.getLogger("embedding_api")

MODEL_SERVICE_URL = "http://localhost:8001"
VALID_TASK_TYPES = ["query", "passage"]


class EmbedRequest(BaseModel):
    texts: list[str]
    task_type: str | None = None

    @field_validator("texts")
    @classmethod
    def validate_texts_not_empty(cls, v):
        if not v:
            raise HTTPException(status_code=422, detail="texts cannot be empty")
        return v

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, v):
        if v is not None and v not in VALID_TASK_TYPES:
            raise HTTPException(
                status_code=422, detail=f"Invalid task_type. Must be one of: {VALID_TASK_TYPES}"
            )
        return v


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(filename="api.log", level=logging.INFO)
    logger.info("starting up app")
    logger.info(f"model service url: {MODEL_SERVICE_URL}")
    yield
    logger.info("shutting down app")


app = FastAPI(
    title="Embedding API",
    description="API for generating embeddings using the multilingual-e5-large model.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> JSONResponse:
    return JSONResponse({"status": "healthy"})


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: Request, body: EmbedRequest) -> EmbedResponse:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MODEL_SERVICE_URL}/embed",
            json={"texts": body.texts, "task_type": body.task_type},
        )
        response.raise_for_status()
        data = response.json()

    return EmbedResponse(embeddings=data["embeddings"])
