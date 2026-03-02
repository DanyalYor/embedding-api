from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )

    app_name: str = "Embedding API"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    model_name: str = "intfloat/multilingual-e5-large"
    model_cache_dir: str = "./models"
    provider: str = "CPUExecutionProvider"
    max_sequence_length: int = 512
    max_batch_size: int = 32

    environment: str = "development"


settings = Settings()
