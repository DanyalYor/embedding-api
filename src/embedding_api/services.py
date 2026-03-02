import logging
from sentence_transformers import SentenceTransformer
from embedding_api.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.model_name = settings.model_name
        logger.info(f"Loading model: {self.model_name} with ONNX backend")
        
        self.model = SentenceTransformer(
            self.model_name,
            backend="onnx",  
            model_kwargs={
                "provider": settings.provider,  
            }
        )
        logger.info("Model loaded successfully")

    def _add_prefix(self, texts: list[str], task_type: str | None) -> list[str]:
        prefix = f"{task_type}: " if task_type else "query: "
        return [f"{prefix}{text}" for text in texts]

    def embed(self, texts: list[str], task_type: str | None = None) -> list[list[float]]:
        prefixed_texts = self._add_prefix(texts, task_type)
        embeddings = self.model.encode(
            prefixed_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  
        )
        return embeddings.tolist()
