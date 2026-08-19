"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore the many .env vars not declared here yet
    )

    environment: str = "development"

    log_level: str = "INFO"

    cors_allow_origins: str = "http://localhost:3000"

    # OpenAI
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_generation_model: str = "gpt-4.1-mini"
    openai_generation_model_high: str = "gpt-4.1"
    openai_vision_model: str = "gpt-4.1-mini"
    openai_vision_model_high: str = "gpt-4.1"

    # Supabase
    supabase_url: str = ""
    supabase_service_role_key: str = ""
    supabase_storage_bucket: str = "esg-page-images"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "esg_documents"

    # Rerank: Local Cross Encoder
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_top_n: int = 5
    reranker_provider: str = "onnx"

    # Rerank ONNX
    onnx_reranker_filename: str = "onnx/model_quint8_avx2.onnx"

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://jp.cloud.langfuse.com"

    # Rate Limit
    rate_limit_units_per_hour: int = 120
    # Maximum a user can bank up: how large a burst is allowed after being idle.
    rate_limit_burst_units: int = 120

    @property
    def cors_origins(self) -> list[str]:
        """Parse the comma-separated origins string into a list."""
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read env once, reuse everywhere)."""
    return Settings()
