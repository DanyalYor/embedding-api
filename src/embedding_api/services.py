import time
from pathlib import Path

import structlog
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from embedding_api.config import settings

logger = structlog.get_logger()


class EmbeddingService:
    def __init__(self):
        self.model_name = settings.model_name
        self.cache_path = (
            Path(settings.model_cache_dir) / f"{self.model_name.replace('/', '-')}-onnx"
        )
        self.cache_path.mkdir(parents=True, exist_ok=True)

        needs_export = not (self.cache_path / "model.onnx").exists()

        if settings.environment == "production" and needs_export:
            logger.error(
                "production_model_missing",
                cache_path=str(self.cache_path),
                hint="Pre-export model during Docker build",
            )
            raise RuntimeError(
                f"Production mode: pre-exported model not found at {self.cache_path}"
            )

        logger.info(
            "model_loading",
            model=self.model_name,
            cache_path=str(self.cache_path),
            export_needed=needs_export,
            provider=settings.provider,
        )

        start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name if needs_export else str(self.cache_path)
        )

        self.model = SentenceTransformer(
            self.model_name if needs_export else str(self.cache_path),
            backend="onnx",
            model_kwargs={
                "provider": settings.provider,
                "file_name": settings.file_name,
            },
        )

        load_duration = (time.time() - start) * 1000

        if needs_export:
            self.model.save_pretrained(str(self.cache_path))
            logger.info("model_exported", cache_path=str(self.cache_path))
        else:
            logger.info("model_loaded_from_cache", cache_path=str(self.cache_path))

        logger.info("model_ready", load_duration_ms=round(load_duration, 2))

    def _add_prefix(self, texts: list[str], task_type: str | None) -> list[str]:
        prefix = f"{task_type}: " if task_type else "query: "
        return [f"{prefix}{text}" for text in texts]

    def embed(
        self, texts: list[str], task_type: str | None = None
    ) -> list[list[float]]:
        prefixed_texts = self._add_prefix(texts, task_type)

        logger.debug(
            "tokenizing",
            text_count=len(prefixed_texts),
            max_length=settings.max_sequence_length,
        )

        start = time.time()
        embeddings = self.model.encode(
            prefixed_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=settings.max_batch_size,
        )
        infer_duration = (time.time() - start) * 1000

        logger.debug(
            "inference_done",
            output_shape=embeddings.shape,
            duration_ms=round(infer_duration, 2),
        )

        return embeddings.tolist()
