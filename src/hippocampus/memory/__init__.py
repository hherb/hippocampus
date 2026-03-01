from __future__ import annotations

import asyncpg

from hippocampus.embeddings.base import EmbeddingProvider
from hippocampus.memory.episodic import EpisodicMemory
from hippocampus.memory.semantic import SemanticMemory
from hippocampus.memory.reflection import ReflectionMemory


class MemoryManager:
    """Facade that holds all three memory subsystems."""

    def __init__(self, pool: asyncpg.Pool, embedder: EmbeddingProvider) -> None:
        self.pool = pool
        self.embedder = embedder
        self.episodic = EpisodicMemory(pool, embedder)
        self.semantic = SemanticMemory(pool, embedder)
        self.reflection = ReflectionMemory(pool, embedder)
