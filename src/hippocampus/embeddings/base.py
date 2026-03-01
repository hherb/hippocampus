from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum


class TaskType(str, Enum):
    """Task-type prefixes for embedding models that support them (e.g. nomic)."""

    DOCUMENT = "search_document"
    QUERY = "search_query"
    CLUSTERING = "clustering"
    CLASSIFICATION = "classification"


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(
        self, texts: list[str], task_type: TaskType = TaskType.DOCUMENT
    ) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input text."""
        ...

    async def embed_one(
        self, text: str, task_type: TaskType = TaskType.DOCUMENT
    ) -> list[float]:
        results = await self.embed([text], task_type)
        return results[0]
