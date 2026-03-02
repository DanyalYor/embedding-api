from pydantic import BaseModel, Field, field_validator

VALID_TASK_TYPES = ["query", "passage"]


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    task_type: str | None = None

    @field_validator("texts")
    @classmethod
    def validate_texts_not_empty(cls, v):
        if not v:
            raise ValueError("texts cannot be empty")
        return v

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, v):
        if v is not None and v not in VALID_TASK_TYPES:
            raise ValueError(f"Invalid task_type. Must be one of: {VALID_TASK_TYPES}")
        return v


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
