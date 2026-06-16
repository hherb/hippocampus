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

    # Vector search tuning (pgvector HNSW).
    # Because a single HNSW index is shared across tenants, owner_id / type
    # filters are applied after the ANN candidate cut. Iterative scan keeps the
    # executor scanning until LIMIT is satisfied post-filter, and a larger
    # ef_search widens the candidate pool. Requires pgvector >= 0.8 for
    # iterative scan; on older versions it is silently ignored.
    hnsw_iterative_scan: str = "relaxed_order"  # off | relaxed_order | strict_order
    hnsw_ef_search: int = 100

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
