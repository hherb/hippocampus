"""Application configuration via environment variables.

All settings can be overridden with ``HIPPOCAMPUS_<NAME>`` env vars
(e.g. ``HIPPOCAMPUS_DB_URL``).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

SIMILARITY_PRECISION: int = 4
"""Decimal places used when rounding similarity scores in API responses."""


class Settings(BaseSettings):
    """Hippocampus settings, loaded from ``HIPPOCAMPUS_*`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="HIPPOCAMPUS_")

    # Database
    db_url: str = "postgresql://localhost:5432/hippocampus"
    db_min_connections: int = 2
    db_max_connections: int = 10

    # Embeddings
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text-v2-moe"
    embedding_dimensions: int = 768
    embedding_prefix: bool = True
    embedding_request_timeout: float = 120.0

    # API server
    api_host: str = "127.0.0.1"
    api_port: int = 8420

    # MCP SSE transport
    mcp_sse_host: str = "127.0.0.1"
    mcp_sse_port: int = 8421

    # Multi-tenancy
    default_owner: str = "default"


settings = Settings()
