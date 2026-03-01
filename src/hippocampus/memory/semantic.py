from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg
import numpy as np

from hippocampus.db.models import Entity, Relation
from hippocampus.embeddings.base import EmbeddingProvider, TaskType


class SemanticMemory:
    """Knowledge graph: entities and their relations, scoped to an owner."""

    def __init__(
        self, pool: asyncpg.Pool, embedder: EmbeddingProvider, owner_id: str
    ) -> None:
        self.pool = pool
        self.embedder = embedder
        self.owner_id = owner_id

    # ── Entities ────────────────────────────────────────────────────────

    async def add_entity(
        self,
        name: str,
        entity_type: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        confidence: float = 1.0,
        source_episode_id: UUID | None = None,
    ) -> Entity:
        text = f"{name}: {description}" if description else name
        embedding = await self.embedder.embed_one(text, TaskType.DOCUMENT)

        row = await self.pool.fetchrow(
            """INSERT INTO entities
                   (owner_id, name, entity_type, description, embedding,
                    metadata, confidence, source_episode_id)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
               ON CONFLICT (owner_id, name, entity_type) DO UPDATE SET
                   description = COALESCE(EXCLUDED.description, entities.description),
                   embedding = EXCLUDED.embedding,
                   metadata = entities.metadata || EXCLUDED.metadata,
                   confidence = GREATEST(entities.confidence, EXCLUDED.confidence),
                   updated_at = NOW()
               RETURNING *""",
            self.owner_id,
            name,
            entity_type,
            description,
            np.array(embedding, dtype=np.float32),
            metadata or {},
            confidence,
            source_episode_id,
        )
        return Entity.from_row(row)

    async def find_entities(
        self,
        query: str,
        entity_type: str | None = None,
        limit: int = 10,
        semantic: bool = True,
    ) -> list[tuple[Entity, float]]:
        if semantic:
            embedding = await self.embedder.embed_one(query, TaskType.QUERY)
            vec = np.array(embedding, dtype=np.float32)

            if entity_type:
                rows = await self.pool.fetch(
                    """SELECT *, 1 - (embedding <=> $1) AS similarity
                       FROM entities
                       WHERE embedding IS NOT NULL
                         AND owner_id = $3 AND entity_type = $4
                       ORDER BY embedding <=> $1
                       LIMIT $2""",
                    vec,
                    limit,
                    self.owner_id,
                    entity_type,
                )
            else:
                rows = await self.pool.fetch(
                    """SELECT *, 1 - (embedding <=> $1) AS similarity
                       FROM entities
                       WHERE embedding IS NOT NULL AND owner_id = $3
                       ORDER BY embedding <=> $1
                       LIMIT $2""",
                    vec,
                    limit,
                    self.owner_id,
                )
            return [(Entity.from_row(r), float(r["similarity"])) for r in rows]

        # Text search fallback
        pattern = f"%{query}%"
        if entity_type:
            rows = await self.pool.fetch(
                """SELECT *, 1.0 AS similarity FROM entities
                   WHERE owner_id = $3
                     AND (name ILIKE $1 OR description ILIKE $1)
                     AND entity_type = $4
                   LIMIT $2""",
                pattern,
                limit,
                self.owner_id,
                entity_type,
            )
        else:
            rows = await self.pool.fetch(
                """SELECT *, 1.0 AS similarity FROM entities
                   WHERE owner_id = $3
                     AND (name ILIKE $1 OR description ILIKE $1)
                   LIMIT $2""",
                pattern,
                limit,
                self.owner_id,
            )
        return [(Entity.from_row(r), float(r["similarity"])) for r in rows]

    async def get_entity(self, entity_id: UUID) -> Entity | None:
        row = await self.pool.fetchrow(
            "SELECT * FROM entities WHERE id = $1 AND owner_id = $2",
            entity_id,
            self.owner_id,
        )
        return Entity.from_row(row) if row else None

    async def get_entity_by_name(
        self, name: str, entity_type: str
    ) -> Entity | None:
        row = await self.pool.fetchrow(
            """SELECT * FROM entities
               WHERE owner_id = $1 AND name = $2 AND entity_type = $3""",
            self.owner_id,
            name,
            entity_type,
        )
        return Entity.from_row(row) if row else None

    async def delete_entity(self, entity_id: UUID) -> bool:
        result = await self.pool.execute(
            "DELETE FROM entities WHERE id = $1 AND owner_id = $2",
            entity_id,
            self.owner_id,
        )
        return result == "DELETE 1"

    # ── Relations ───────────────────────────────────────────────────────

    async def add_relation(
        self,
        subject_id: UUID,
        predicate: str,
        object_id: UUID,
        metadata: dict[str, Any] | None = None,
        confidence: float = 1.0,
        source_episode_id: UUID | None = None,
    ) -> Relation:
        row = await self.pool.fetchrow(
            """INSERT INTO relations
                   (owner_id, subject_id, predicate, object_id, metadata,
                    confidence, source_episode_id)
               VALUES ($1, $2, $3, $4, $5, $6, $7)
               ON CONFLICT (subject_id, predicate, object_id) DO UPDATE SET
                   metadata = relations.metadata || EXCLUDED.metadata,
                   confidence = GREATEST(relations.confidence, EXCLUDED.confidence),
                   updated_at = NOW()
               RETURNING *""",
            self.owner_id,
            subject_id,
            predicate,
            object_id,
            metadata or {},
            confidence,
            source_episode_id,
        )
        return Relation.from_row(row)

    async def get_relations(
        self,
        entity_id: UUID | None = None,
        predicate: str | None = None,
        as_subject: bool = True,
        as_object: bool = True,
        limit: int = 50,
    ) -> list[Relation]:
        conditions: list[str] = [f"r.owner_id = $1"]
        params: list[Any] = [self.owner_id]
        idx = 2

        if entity_id is not None:
            parts = []
            if as_subject:
                parts.append(f"r.subject_id = ${idx}")
            if as_object:
                parts.append(f"r.object_id = ${idx}")
            if parts:
                conditions.append(f"({' OR '.join(parts)})")
                params.append(entity_id)
                idx += 1

        if predicate is not None:
            conditions.append(f"r.predicate = ${idx}")
            params.append(predicate)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}"
        params.append(limit)

        rows = await self.pool.fetch(
            f"""SELECT r.*,
                       s.name AS subject_name,
                       o.name AS object_name
                FROM relations r
                JOIN entities s ON r.subject_id = s.id
                JOIN entities o ON r.object_id = o.id
                {where}
                ORDER BY r.created_at DESC
                LIMIT ${idx}""",
            *params,
        )
        return [Relation.from_row(r) for r in rows]

    async def delete_relation(self, relation_id: UUID) -> bool:
        result = await self.pool.execute(
            "DELETE FROM relations WHERE id = $1 AND owner_id = $2",
            relation_id,
            self.owner_id,
        )
        return result == "DELETE 1"

    # ── Graph traversal ─────────────────────────────────────────────────

    async def traverse(
        self,
        entity_id: UUID,
        max_depth: int = 2,
    ) -> list[dict[str, Any]]:
        """BFS traversal from an entity, returning connected subgraph."""
        rows = await self.pool.fetch(
            """WITH RECURSIVE graph AS (
                   SELECT r.id, r.subject_id, r.predicate, r.object_id,
                          r.confidence, 1 AS depth
                   FROM relations r
                   WHERE r.owner_id = $3
                     AND (r.subject_id = $1 OR r.object_id = $1)

                   UNION

                   SELECT r.id, r.subject_id, r.predicate, r.object_id,
                          r.confidence, g.depth + 1
                   FROM relations r
                   JOIN graph g ON (r.subject_id = g.object_id
                                    OR r.subject_id = g.subject_id
                                    OR r.object_id = g.subject_id
                                    OR r.object_id = g.object_id)
                   WHERE r.owner_id = $3
                     AND g.depth < $2
                     AND r.id != g.id
               )
               SELECT DISTINCT g.*,
                      s.name AS subject_name, s.entity_type AS subject_type,
                      o.name AS object_name, o.entity_type AS object_type
               FROM graph g
               JOIN entities s ON g.subject_id = s.id
               JOIN entities o ON g.object_id = o.id
               ORDER BY g.depth, g.confidence DESC""",
            entity_id,
            max_depth,
            self.owner_id,
        )
        return [
            {
                "subject": {"id": str(r["subject_id"]), "name": r["subject_name"], "type": r["subject_type"]},
                "predicate": r["predicate"],
                "object": {"id": str(r["object_id"]), "name": r["object_name"], "type": r["object_type"]},
                "confidence": r["confidence"],
                "depth": r["depth"],
            }
            for r in rows
        ]
