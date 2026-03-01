from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg
import numpy as np

from hippocampus.db.models import Reflection, RevisionProposal
from hippocampus.embeddings.base import EmbeddingProvider, TaskType


class ReflectionMemory:
    """Meta-memory: reflections, summaries, and human-governed revision proposals."""

    def __init__(
        self, pool: asyncpg.Pool, embedder: EmbeddingProvider, owner_id: str
    ) -> None:
        self.pool = pool
        self.embedder = embedder
        self.owner_id = owner_id

    # ── Reflections ─────────────────────────────────────────────────────

    async def create(
        self,
        content: str,
        reflection_type: str = "summary",
        source_episode_ids: list[UUID] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Reflection:
        embedding = await self.embedder.embed_one(content, TaskType.DOCUMENT)

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """INSERT INTO reflections
                           (owner_id, content, reflection_type, embedding, metadata)
                       VALUES ($1, $2, $3, $4, $5)
                       RETURNING *""",
                    self.owner_id,
                    content,
                    reflection_type,
                    np.array(embedding, dtype=np.float32),
                    metadata or {},
                )
                reflection = Reflection.from_row(row)

                if source_episode_ids:
                    await conn.executemany(
                        """INSERT INTO reflection_episodes (reflection_id, episode_id)
                           VALUES ($1, $2)
                           ON CONFLICT DO NOTHING""",
                        [(reflection.id, eid) for eid in source_episode_ids],
                    )
                    reflection.source_episode_ids = list(source_episode_ids)

        return reflection

    async def search(
        self,
        query: str,
        reflection_type: str | None = None,
        limit: int = 10,
    ) -> list[tuple[Reflection, float]]:
        embedding = await self.embedder.embed_one(query, TaskType.QUERY)
        vec = np.array(embedding, dtype=np.float32)

        if reflection_type:
            rows = await self.pool.fetch(
                """SELECT *, 1 - (embedding <=> $1) AS similarity
                   FROM reflections
                   WHERE embedding IS NOT NULL
                     AND owner_id = $3 AND reflection_type = $4
                   ORDER BY embedding <=> $1
                   LIMIT $2""",
                vec,
                limit,
                self.owner_id,
                reflection_type,
            )
        else:
            rows = await self.pool.fetch(
                """SELECT *, 1 - (embedding <=> $1) AS similarity
                   FROM reflections
                   WHERE embedding IS NOT NULL AND owner_id = $3
                   ORDER BY embedding <=> $1
                   LIMIT $2""",
                vec,
                limit,
                self.owner_id,
            )
        return [(Reflection.from_row(r), float(r["similarity"])) for r in rows]

    async def list_recent(
        self,
        reflection_type: str | None = None,
        limit: int = 10,
    ) -> list[Reflection]:
        if reflection_type:
            rows = await self.pool.fetch(
                """SELECT * FROM reflections
                   WHERE owner_id = $1 AND reflection_type = $2
                   ORDER BY created_at DESC LIMIT $3""",
                self.owner_id,
                reflection_type,
                limit,
            )
        else:
            rows = await self.pool.fetch(
                """SELECT * FROM reflections
                   WHERE owner_id = $1
                   ORDER BY created_at DESC LIMIT $2""",
                self.owner_id,
                limit,
            )
        return [Reflection.from_row(r) for r in rows]

    # ── Revision Proposals ──────────────────────────────────────────────

    async def propose_revision(
        self,
        target_type: str,
        target_id: UUID,
        action: str,
        proposed_changes: dict[str, Any],
        reason: str,
    ) -> RevisionProposal:
        row = await self.pool.fetchrow(
            """INSERT INTO revision_proposals
                   (owner_id, target_type, target_id, action,
                    proposed_changes, reason)
               VALUES ($1, $2, $3, $4, $5, $6)
               RETURNING *""",
            self.owner_id,
            target_type,
            target_id,
            action,
            proposed_changes,
            reason,
        )
        return RevisionProposal.from_row(row)

    async def list_revisions(
        self,
        status: str = "pending",
        limit: int = 50,
    ) -> list[RevisionProposal]:
        rows = await self.pool.fetch(
            """SELECT * FROM revision_proposals
               WHERE owner_id = $1 AND status = $2
               ORDER BY created_at DESC LIMIT $3""",
            self.owner_id,
            status,
            limit,
        )
        return [RevisionProposal.from_row(r) for r in rows]

    async def approve_revision(
        self, revision_id: UUID, review_notes: str | None = None
    ) -> RevisionProposal | None:
        """Approve and apply a revision proposal."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """UPDATE revision_proposals
                       SET status = 'approved', reviewed_at = NOW(),
                           review_notes = $3
                       WHERE id = $1 AND owner_id = $2 AND status = 'pending'
                       RETURNING *""",
                    revision_id,
                    self.owner_id,
                    review_notes,
                )
                if row is None:
                    return None

                proposal = RevisionProposal.from_row(row)
                await self._apply_revision(conn, proposal)
                return proposal

    async def reject_revision(
        self, revision_id: UUID, review_notes: str | None = None
    ) -> RevisionProposal | None:
        row = await self.pool.fetchrow(
            """UPDATE revision_proposals
               SET status = 'rejected', reviewed_at = NOW(),
                   review_notes = $3
               WHERE id = $1 AND owner_id = $2 AND status = 'pending'
               RETURNING *""",
            revision_id,
            self.owner_id,
            review_notes,
        )
        return RevisionProposal.from_row(row) if row else None

    async def _apply_revision(
        self, conn: asyncpg.Connection, proposal: RevisionProposal
    ) -> None:
        changes = proposal.proposed_changes

        if proposal.action == "delete":
            table = "entities" if proposal.target_type == "entity" else "relations"
            await conn.execute(
                f"DELETE FROM {table} WHERE id = $1 AND owner_id = $2",
                proposal.target_id,
                self.owner_id,
            )

        elif proposal.action == "update" and proposal.target_type == "entity":
            sets = []
            params: list[Any] = [proposal.target_id, self.owner_id]
            idx = 3
            for field in ("name", "description", "entity_type", "confidence"):
                if field in changes:
                    sets.append(f"{field} = ${idx}")
                    params.append(changes[field])
                    idx += 1
            if sets:
                sets.append("updated_at = NOW()")
                await conn.execute(
                    f"UPDATE entities SET {', '.join(sets)} WHERE id = $1 AND owner_id = $2",
                    *params,
                )
                # Re-embed if name or description changed
                if "name" in changes or "description" in changes:
                    entity = await conn.fetchrow(
                        "SELECT name, description FROM entities WHERE id = $1",
                        proposal.target_id,
                    )
                    if entity:
                        text = f"{entity['name']}: {entity['description']}" if entity["description"] else entity["name"]
                        emb = await self.embedder.embed_one(text, TaskType.DOCUMENT)
                        await conn.execute(
                            "UPDATE entities SET embedding = $2 WHERE id = $1",
                            proposal.target_id,
                            np.array(emb, dtype=np.float32),
                        )

        elif proposal.action == "update" and proposal.target_type == "relation":
            sets = []
            params_r: list[Any] = [proposal.target_id, self.owner_id]
            idx = 3
            for field in ("predicate", "confidence"):
                if field in changes:
                    sets.append(f"{field} = ${idx}")
                    params_r.append(changes[field])
                    idx += 1
            if sets:
                sets.append("updated_at = NOW()")
                await conn.execute(
                    f"UPDATE relations SET {', '.join(sets)} WHERE id = $1 AND owner_id = $2",
                    *params_r,
                )

        elif proposal.action == "merge" and proposal.target_type == "entity":
            merge_into = changes.get("merge_into_id")
            if merge_into:
                from uuid import UUID as _UUID

                target = _UUID(merge_into) if isinstance(merge_into, str) else merge_into
                # Re-point all relations from the old entity to the merge target
                await conn.execute(
                    """UPDATE relations SET subject_id = $2, updated_at = NOW()
                       WHERE subject_id = $1 AND owner_id = $3""",
                    proposal.target_id,
                    target,
                    self.owner_id,
                )
                await conn.execute(
                    """UPDATE relations SET object_id = $2, updated_at = NOW()
                       WHERE object_id = $1 AND owner_id = $3""",
                    proposal.target_id,
                    target,
                    self.owner_id,
                )
                await conn.execute(
                    "DELETE FROM entities WHERE id = $1 AND owner_id = $2",
                    proposal.target_id,
                    self.owner_id,
                )
