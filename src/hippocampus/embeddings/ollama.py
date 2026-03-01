from __future__ import annotations

import httpx

from hippocampus.config import Settings
from hippocampus.embeddings.base import EmbeddingProvider, TaskType


class OllamaEmbedding(EmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.ollama_url.rstrip("/")
        self.model = settings.embedding_model
        self.use_prefix = settings.embedding_prefix
        self._client = httpx.AsyncClient(timeout=120.0)

    async def embed(
        self, texts: list[str], task_type: TaskType = TaskType.DOCUMENT
    ) -> list[list[float]]:
        if not texts:
            return []

        input_texts = texts
        if self.use_prefix:
            input_texts = [f"{task_type.value}: {t}" for t in texts]

        resp = await self._client.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": input_texts},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"]

    async def close(self) -> None:
        await self._client.aclose()
