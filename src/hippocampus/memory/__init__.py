"""Memory subsystem facade."""

from __future__ import annotations

import asyncpg

from hippocampus.embeddings.base import EmbeddingProvider
from hippocampus.memory.episodic import EpisodicMemory
from hippocampus.memory.reflection import ReflectionMemory
from hippocampus.memory.semantic import SemanticMemory


class MemoryManager:
    """Facade that holds all three memory subsystems, scoped to an owner."""

    def __init__(
        self, pool: asyncpg.Pool, embedder: EmbeddingProvider, owner_id: str
    ) -> None:
        self.pool = pool
        self.embedder = embedder
        self.owner_id = owner_id
        self.episodic = EpisodicMemory(pool, embedder, owner_id)
        self.semantic = SemanticMemory(pool, embedder, owner_id)
        self.reflection = ReflectionMemory(pool, embedder, owner_id)
