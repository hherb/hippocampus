"""Episodic memory: timestamped, vector-searchable records of experiences."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg
import numpy as np

from hippocampus.db.models import Episode
from hippocampus.embeddings.base import EmbeddingProvider, TaskType


class EpisodicMemory:
    """Store and recall episodic memories via semantic and temporal queries."""

    def __init__(
        self, pool: asyncpg.Pool, embedder: EmbeddingProvider, owner_id: str
    ) -> None:
        self.pool = pool
        self.embedder = embedder
        self.owner_id = owner_id

    async def store(
        self,
        content: str,
        source: str = "conversation",
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        """Persist a new episodic memory with its embedding."""
        embedding = await self.embedder.embed_one(content, TaskType.DOCUMENT)
        row = await self.pool.fetchrow(
            """INSERT INTO episodes
                   (owner_id, source, content, embedding, session_id, metadata)
               VALUES ($1, $2, $3, $4, $5, $6)
               RETURNING *""",
            self.owner_id,
            source,
            content,
            np.array(embedding, dtype=np.float32),
            session_id,
            metadata or {},
        )
        return Episode.from_row(row)

    async def recall(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.0,
        source: str | None = None,
        session_id: str | None = None,
    ) -> list[tuple[Episode, float]]:
        """Search episodes by semantic similarity, with optional filters."""
        embedding = await self.embedder.embed_one(query, TaskType.QUERY)
        vec = np.array(embedding, dtype=np.float32)

        conditions = ["embedding IS NOT NULL", "owner_id = $2"]
        params: list[Any] = [vec, self.owner_id]
        idx = 3

        if min_similarity > 0.0:
            conditions.append(f"1 - (embedding <=> $1) >= ${idx}")
            params.append(min_similarity)
            idx += 1
        if source is not None:
            conditions.append(f"source = ${idx}")
            params.append(source)
            idx += 1
        if session_id is not None:
            conditions.append(f"session_id = ${idx}")
            params.append(session_id)
            idx += 1

        where = " AND ".join(conditions)
        params.append(limit)

        rows = await self.pool.fetch(
            f"""SELECT *, 1 - (embedding <=> $1) AS similarity
                FROM episodes
                WHERE {where}
                ORDER BY embedding <=> $1
                LIMIT ${idx}""",
            *params,
        )
        return [
            (Episode.from_row(row), float(row["similarity"]))
            for row in rows
        ]

    async def recall_recent(
        self,
        limit: int = 10,
        session_id: str | None = None,
    ) -> list[Episode]:
        """Retrieve the most recent episodes, optionally filtered by session."""
        if session_id:
            rows = await self.pool.fetch(
                """SELECT * FROM episodes
                   WHERE owner_id = $1 AND session_id = $2
                   ORDER BY created_at DESC LIMIT $3""",
                self.owner_id,
                session_id,
                limit,
            )
        else:
            rows = await self.pool.fetch(
                """SELECT * FROM episodes
                   WHERE owner_id = $1
                   ORDER BY created_at DESC LIMIT $2""",
                self.owner_id,
                limit,
            )
        return [Episode.from_row(row) for row in rows]

    async def get(self, episode_id: UUID) -> Episode | None:
        """Fetch a single episode by ID."""
        row = await self.pool.fetchrow(
            "SELECT * FROM episodes WHERE id = $1 AND owner_id = $2",
            episode_id,
            self.owner_id,
        )
        return Episode.from_row(row) if row else None

    async def delete(self, episode_id: UUID) -> bool:
        """Delete an episode. Returns *True* if a row was removed."""
        result = await self.pool.execute(
            "DELETE FROM episodes WHERE id = $1 AND owner_id = $2",
            episode_id,
            self.owner_id,
        )
        return result == "DELETE 1"
