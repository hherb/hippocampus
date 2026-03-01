from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "HIPPOCAMPUS_"}

    # Database
    db_url: str = "postgresql://localhost:5432/hippocampus"
    db_min_connections: int = 2
    db_max_connections: int = 10

    # Embeddings
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text-v2-moe"
    embedding_dimensions: int = 768
    embedding_prefix: bool = True

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8420

    # MCP SSE transport
    mcp_sse_host: str = "0.0.0.0"
    mcp_sse_port: int = 8421


settings = Settings()
