"""FastAPI REST server exposing Hippocampus memory operations."""

from __future__ import annotations

import asyncpg
from contextlib import asynccontextmanager
from typing import Any, Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from hippocampus.config import SIMILARITY_PRECISION, settings
from hippocampus.db.pool import create_pool
from hippocampus.db.schema import ensure_schema
from hippocampus.embeddings.ollama import OllamaEmbedding
from hippocampus.memory import MemoryManager

_pool: asyncpg.Pool | None = None
_embedder: OllamaEmbedding | None = None


def _get_manager(owner_id: str) -> MemoryManager:
    """Create a MemoryManager scoped to the given owner."""
    return MemoryManager(_pool, _embedder, owner_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the database pool, schema, and embedding provider."""
    global _pool, _embedder
    _pool = await create_pool(settings)
    try:
        await ensure_schema(_pool, settings)
        _embedder = OllamaEmbedding(settings)
    except BaseException:
        await _pool.close()
        _pool = None
        raise
    try:
        yield
    finally:
        await _pool.close()
        await _embedder.close()


app = FastAPI(
    title="Hippocampus",
    description="Persistent memory retrieval system for LLMs",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Request Models ─────────────────────────────────────────────────────


class StoreEpisodeRequest(BaseModel):
    """Body for storing a new episodic memory."""

    content: str
    source: str = "conversation"
    session_id: str | None = None
    metadata: dict[str, Any] | None = None


class SearchRequest(BaseModel):
    """Body for semantic episode search."""

    query: str
    limit: int = Field(10, ge=1, le=100)
    min_similarity: float = Field(0.0, ge=0.0, le=1.0)
    source: str | None = None
    session_id: str | None = None


class EntityRequest(BaseModel):
    """Body for creating or upserting an entity."""

    name: str
    entity_type: str
    description: str | None = None
    metadata: dict[str, Any] | None = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    source_episode_id: UUID | None = None


class RelationRequest(BaseModel):
    """Body for creating a relation between entities."""

    subject_id: UUID
    predicate: str
    object_id: UUID
    metadata: dict[str, Any] | None = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    source_episode_id: UUID | None = None


class ReflectionRequest(BaseModel):
    """Body for creating a reflection."""

    content: str
    reflection_type: str = "summary"
    source_episode_ids: list[UUID] | None = None
    metadata: dict[str, Any] | None = None


class RevisionRequest(BaseModel):
    """Body for proposing a revision."""

    target_type: Literal["entity", "relation"]
    target_id: UUID
    action: Literal["update", "delete", "merge"]
    proposed_changes: dict[str, Any]
    reason: str


class ReviewRequest(BaseModel):
    """Body for approving or rejecting a revision."""

    review_notes: str | None = None


# ── Episodes ───────────────────────────────────────────────────────────


@app.post("/api/v1/episodes")
async def store_episode(
    req: StoreEpisodeRequest,
    owner_id: str = Query(..., description="Owner/tenant identifier"),
):
    mgr = _get_manager(owner_id)
    episode = await mgr.episodic.store(
        req.content, req.source, req.session_id, req.metadata
    )
    return episode.to_dict()


@app.post("/api/v1/episodes/search")
async def search_episodes(
    req: SearchRequest,
    owner_id: str = Query(..., description="Owner/tenant identifier"),
):
    mgr = _get_manager(owner_id)
    results = await mgr.episodic.recall(
        req.query, req.limit, req.min_similarity, req.source, req.session_id
    )
    return {
        "results": [
            {"episode": ep.to_dict(), "similarity": round(sim, SIMILARITY_PRECISION)}
            for ep, sim in results
        ]
    }


@app.get("/api/v1/episodes/recent")
async def recent_episodes(
    owner_id: str = Query(..., description="Owner/tenant identifier"),
    limit: int = Query(10, ge=1, le=100),
    session_id: str | None = Query(None),
):
    mgr = _get_manager(owner_id)
    episodes = await mgr.episodic.recall_recent(limit, session_id)
    return {"episodes": [ep.to_dict() for ep in episodes]}


@app.get("/api/v1/episodes/{episode_id}")
async def get_episode(
    episode_id: UUID,
    owner_id: str = Query(..., description="Owner/tenant identifier"),
):
    mgr = _get_manager(owner_id)
    episode = await mgr.episodic.get(episode_id)
    if episode is None:
        raise HTTPException(404, "Episode not found")
    return episode.to_dict()


@app.delete("/api/v1/episodes/{episode_id}")
async def delete_episode(
    episode_id: UUID,
    owner_id: str = Query(..., description="Owner/tenant identifier"),
):
    mgr = _get_manager(owner_id)
    deleted = await mgr.episodic.delete(episode_id)
    if not deleted:
        raise HTTPException(404, "Episode not found")
    return {"deleted": True}


# ── Entities ───────────────────────────────────────────────────────────


@app.post("/api/v1/entities")
async def create_entity(
    req: EntityRequest,
    owner_id: str = Query(..., description="Owner/tenant identifier"),
):
    mgr = _get_manager(owner_id)
    entity = await mgr.semantic.add_entity(
        req.name,
        req.entity_type,
        req.description,
        req.metadata,
        req.confidence,
        req.source_episode_id,
    )
    return entity.to_dict()


@app.get("/api/v1/entities/search")
async def search_entities(
    owner_id: str = Query(..., description="Owner/tenant identifier"),
    query: str = Query(...),
    entity_type: str | None = Query(None),
    limit: int = Query(10, ge=1, le=100),
    semantic: bool = Query(True),
):
    mgr = _get_manager(owner_id)
    results = await mgr.semantic.find_entities(
        query, entity_type, limit, semantic
    )
    return {
        "entities": [
            {"entity": ent.to_dict(), "similarity": round(sim, SIMILARITY_PRECISION)}
            for ent, sim in results
        ]
    }


@app.get("/api/v1/entities/{entity_id}")
async def get_entity(
    entity_id: UUID,
    owner_id: str = Query(..., description="Owner/tenant identifier"),
):
    mgr = _get_manager(owner_id)
    entity = await mgr.semantic.get_entity(entity_id)
    if entity is None:
        raise HTTPException(404, "Entity not found")
    return entity.to_dict()


@app.delete("/api/v1/entities/{entity_id}")
async def delete_entity(
    entity_id: UUID,
    owner_id: str = Query(..., description="Owner/tenant identifier"),
):
    mgr = _get_manager(owner_id)
    deleted = await mgr.semantic.delete_entity(entity_id)
    if not deleted:
        raise HTTPException(404, "Entity not found")
    return {"deleted": True}


# ── Relations ──────────────────────────────────────────────────────────


@app.post("/api/v1/relations")
async def create_relation(
    req: RelationRequest,
    owner_id: str = Query(..., description="Owner/tenant identifier"),
):
    mgr = _get_manager(owner_id)
    relation = await mgr.semantic.add_relation(
        req.subject_id,
        req.predicate,
        req.object_id,
        req.metadata,
        req.confidence,
        req.source_episode_id,
    )
    return relation.to_dict()


@app.get("/api/v1/relations")
async def list_relations(
    owner_id: str = Query(..., description="Owner/tenant identifier"),
    entity_id: UUID | None = Query(None),
    predicate: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    mgr = _get_manager(owner_id)
    relations = await mgr.semantic.get_relations(
        entity_id=entity_id, predicate=predicate, limit=limit
    )
    return {"relations": [r.to_dict() for r in relations]}


@app.delete("/api/v1/relations/{relation_id}")
async def delete_relation(
    relation_id: UUID,
    owner_id: str = Query(..., description="Owner/tenant identifier"),
):
    mgr = _get_manager(owner_id)
    deleted = await mgr.semantic.delete_relation(relation_id)
    if not deleted:
        raise HTTPException(404, "Relation not found")
    return {"deleted": True}


@app.get("/api/v1/entities/{entity_id}/graph")
async def entity_graph(
    entity_id: UUID,
    owner_id: str = Query(..., description="Owner/tenant identifier"),
    max_depth: int = Query(2, ge=1, le=3),
):
    mgr = _get_manager(owner_id)
    entity = await mgr.semantic.get_entity(entity_id)
    if entity is None:
        raise HTTPException(404, "Entity not found")
    subgraph = await mgr.semantic.traverse(entity_id, max_depth)
    return {"entity": entity.to_dict(), "connections": subgraph}


# ── Reflections ────────────────────────────────────────────────────────


@app.post("/api/v1/reflections")
async def create_reflection(
    req: ReflectionRequest,
    owner_id: str = Query(..., description="Owner/tenant identifier"),
):
    mgr = _get_manager(owner_id)
    reflection = await mgr.reflection.create(
        req.content, req.reflection_type, req.source_episode_ids, req.metadata
    )
    return reflection.to_dict()


@app.get("/api/v1/reflections/search")
async def search_reflections(
    owner_id: str = Query(..., description="Owner/tenant identifier"),
    query: str = Query(...),
    reflection_type: str | None = Query(None),
    limit: int = Query(10, ge=1, le=100),
):
    mgr = _get_manager(owner_id)
    results = await mgr.reflection.search(query, reflection_type, limit)
    return {
        "reflections": [
            {"reflection": ref.to_dict(), "similarity": round(sim, SIMILARITY_PRECISION)}
            for ref, sim in results
        ]
    }


@app.get("/api/v1/reflections")
async def list_reflections(
    owner_id: str = Query(..., description="Owner/tenant identifier"),
    reflection_type: str | None = Query(None),
    limit: int = Query(10, ge=1, le=100),
):
    mgr = _get_manager(owner_id)
    reflections = await mgr.reflection.list_recent(reflection_type, limit)
    return {"reflections": [r.to_dict() for r in reflections]}


@app.get("/api/v1/reflections/{reflection_id}")
async def get_reflection(
    reflection_id: UUID,
    owner_id: str = Query(..., description="Owner/tenant identifier"),
):
    mgr = _get_manager(owner_id)
    reflection = await mgr.reflection.get(reflection_id)
    if reflection is None:
        raise HTTPException(404, "Reflection not found")
    return reflection.to_dict()


# ── Revision Proposals ─────────────────────────────────────────────────


@app.post("/api/v1/revisions")
async def create_revision(
    req: RevisionRequest,
    owner_id: str = Query(..., description="Owner/tenant identifier"),
):
    mgr = _get_manager(owner_id)
    proposal = await mgr.reflection.propose_revision(
        req.target_type,
        req.target_id,
        req.action,
        req.proposed_changes,
        req.reason,
    )
    return proposal.to_dict()


@app.get("/api/v1/revisions")
async def list_revisions(
    owner_id: str = Query(..., description="Owner/tenant identifier"),
    status: str = Query("pending"),
    limit: int = Query(50, ge=1, le=200),
):
    mgr = _get_manager(owner_id)
    proposals = await mgr.reflection.list_revisions(status, limit)
    return {"proposals": [p.to_dict() for p in proposals]}


@app.post("/api/v1/revisions/{revision_id}/approve")
async def approve_revision(
    revision_id: UUID,
    owner_id: str = Query(..., description="Owner/tenant identifier"),
    req: ReviewRequest | None = None,
):
    mgr = _get_manager(owner_id)
    notes = req.review_notes if req else None
    proposal = await mgr.reflection.approve_revision(revision_id, notes)
    if proposal is None:
        raise HTTPException(404, "Pending revision not found")
    return proposal.to_dict()


@app.post("/api/v1/revisions/{revision_id}/reject")
async def reject_revision(
    revision_id: UUID,
    owner_id: str = Query(..., description="Owner/tenant identifier"),
    req: ReviewRequest | None = None,
):
    mgr = _get_manager(owner_id)
    notes = req.review_notes if req else None
    proposal = await mgr.reflection.reject_revision(revision_id, notes)
    if proposal is None:
        raise HTTPException(404, "Pending revision not found")
    return proposal.to_dict()


# ── System ─────────────────────────────────────────────────────────────


@app.get("/api/v1/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/v1/stats")
async def stats(
    owner_id: str = Query(..., description="Owner/tenant identifier"),
):
    """Return aggregate counts for all memory types."""
    mgr = _get_manager(owner_id)
    async with mgr.pool.acquire() as conn:
        counts = await conn.fetchrow(
            """SELECT
                   (SELECT count(*) FROM episodes WHERE owner_id = $1) AS episodes,
                   (SELECT count(*) FROM entities WHERE owner_id = $1) AS entities,
                   (SELECT count(*) FROM relations WHERE owner_id = $1) AS relations,
                   (SELECT count(*) FROM reflections WHERE owner_id = $1) AS reflections,
                   (SELECT count(*) FROM revision_proposals
                    WHERE owner_id = $1 AND status = 'pending') AS pending_revisions""",
            owner_id,
        )
    return dict(counts)


def run_api() -> None:
    """Launch the API server via uvicorn."""
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
