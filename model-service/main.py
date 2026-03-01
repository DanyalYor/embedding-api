from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from pydantic import BaseModel, field_validator
import logging
from typing import Any
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_service")

app = FastAPI(title="Model Service")

MODEL_NAME = "intfloat/multilingual-e5-large"
service: dict[str, Any] | None = None

VALID_TASK_TYPES = ["query", "passage"]
DEFAULT_PREFIX = "query"


class EmbedRequest(BaseModel):
    texts: list[str]
    task_type: str | None = None

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, v):
        if v is not None and v not in VALID_TASK_TYPES:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid task_type. Must be one of: {VALID_TASK_TYPES}"
            )
        return v


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


def add_prefix(texts: list[str], task_type: str | None) -> list[str]:
    prefix = f"{task_type}: " if task_type else f"{DEFAULT_PREFIX}: "
    return [f"{prefix}{text}" for text in texts]


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


@app.on_event("startup")
async def startup():
    global service
    logger.info(f"Loading model: {MODEL_NAME}")
    provider = "CPUExecutionProvider"
    service = {
        "model": ORTModelForFeatureExtraction.from_pretrained(
            MODEL_NAME,
            provider=provider,
        ),
        "tokenizer": AutoTokenizer.from_pretrained(MODEL_NAME),
    }
    logger.info("Model loaded successfully")


@app.get("/health")
async def health():
    return {"status": "healthy", "ready": service is not None}


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    if not request.texts:
        return EmbedResponse(embeddings=[])

    if service is None:
        raise HTTPException(status_code=503, detail="Model service not ready")

    prefixed_texts = add_prefix(request.texts, request.task_type)

    inputs = service["tokenizer"](
        prefixed_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )

    outputs = service["model"](**inputs)
    embeddings = outputs.last_hidden_state.mean(axis=1)
    embeddings = normalize_embeddings(embeddings)

    return EmbedResponse(embeddings=embeddings.tolist())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
