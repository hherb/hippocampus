from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg
import numpy as np

from hippocampus.db.models import Episode
from hippocampus.embeddings.base import EmbeddingProvider, TaskType


class EpisodicMemory:
    def __init__(self, pool: asyncpg.Pool, embedder: EmbeddingProvider) -> None:
        self.pool = pool
        self.embedder = embedder

    async def store(
        self,
        content: str,
        source: str = "conversation",
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        embedding = await self.embedder.embed_one(content, TaskType.DOCUMENT)
        row = await self.pool.fetchrow(
            """INSERT INTO episodes (source, content, embedding, session_id, metadata)
               VALUES ($1, $2, $3, $4, $5)
               RETURNING *""",
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
        embedding = await self.embedder.embed_one(query, TaskType.QUERY)
        vec = np.array(embedding, dtype=np.float32)

        conditions = ["embedding IS NOT NULL"]
        params: list[Any] = [vec]
        idx = 2

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
            if float(row["similarity"]) >= min_similarity
        ]

    async def recall_recent(
        self,
        limit: int = 10,
        session_id: str | None = None,
    ) -> list[Episode]:
        if session_id:
            rows = await self.pool.fetch(
                """SELECT * FROM episodes
                   WHERE session_id = $1
                   ORDER BY created_at DESC LIMIT $2""",
                session_id,
                limit,
            )
        else:
            rows = await self.pool.fetch(
                "SELECT * FROM episodes ORDER BY created_at DESC LIMIT $1",
                limit,
            )
        return [Episode.from_row(row) for row in rows]

    async def get(self, episode_id: UUID) -> Episode | None:
        row = await self.pool.fetchrow(
            "SELECT * FROM episodes WHERE id = $1", episode_id
        )
        return Episode.from_row(row) if row else None

    async def delete(self, episode_id: UUID) -> bool:
        result = await self.pool.execute(
            "DELETE FROM episodes WHERE id = $1", episode_id
        )
        return result == "DELETE 1"
