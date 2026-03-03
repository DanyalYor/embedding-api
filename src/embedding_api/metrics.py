from prometheus_client import Gauge, Histogram

EMBED_BATCH_SIZE = Histogram(
    "embed_batch_size",
    "Batch size of embedding requests",
    buckets=[1, 2, 4, 8, 16, 32, 64, 128]
)

EMBED_INFERENCE_DURATION = Histogram(
    "embed_inference_duration_seconds",
    "Model inference duration",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether the model is loaded and ready",
    ["model_name"]
)

MODEL_LOAD_DURATION = Gauge(
    "model_load_duration_seconds",
    "Time taken to load the model",
    ["model_name"]
)


class MetricsCollector:
    @staticmethod
    def record_batch_size(size: int):
        EMBED_BATCH_SIZE.observe(size)

    @staticmethod
    def record_inference_duration(duration: float):
        EMBED_INFERENCE_DURATION.observe(duration)

    @staticmethod
    def set_model_loaded(model_name: str, loaded: bool):
        MODEL_LOADED.labels(model_name=model_name).set(1 if loaded else 0)

    @staticmethod
    def record_model_load_duration(model_name: str, duration: float):
        MODEL_LOAD_DURATION.labels(model_name=model_name).set(duration)
