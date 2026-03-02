from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from pydantic import BaseModel

logger = logging.getLogger("embedding_api")

class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embeddings: list[float]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application lifecycle (Startup and Shutdown events)"""
    logging.basicConfig(filename="api.log", level=logging.INFO)
    logger.info("starting up app")
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
    """Check if the API is running."""
    return JSONResponse({"status": "healthy"})

@app.post("/embed")
async def embed(request: EmbedRequest) -> EmbedResponse:
    """Given text, return list of embeddings"""
    embeddings = [0.0, 0.1]
    return EmbedResponse(embeddings=embeddings) 
