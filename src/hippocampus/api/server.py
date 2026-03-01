from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from hippocampus.config import settings
from hippocampus.db.pool import create_pool
from hippocampus.db.schema import ensure_schema
from hippocampus.embeddings.ollama import OllamaEmbedding
from hippocampus.memory import MemoryManager

_manager: MemoryManager | None = None
_embedder: OllamaEmbedding | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _manager, _embedder
    pool = await create_pool(settings)
    await ensure_schema(pool, settings)
    _embedder = OllamaEmbedding(settings)
    _manager = MemoryManager(pool, _embedder)
    yield
    await pool.close()
    await _embedder.close()


app = FastAPI(
    title="Hippocampus",
    description="Persistent memory retrieval system for LLMs",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Request Models ─────────────────────────────────────────────────────


class StoreEpisodeRequest(BaseModel):
    content: str
    source: str = "conversation"
    session_id: str | None = None
    metadata: dict[str, Any] | None = None


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    min_similarity: float = 0.0
    source: str | None = None
    session_id: str | None = None


class EntityRequest(BaseModel):
    name: str
    entity_type: str
    description: str | None = None
    metadata: dict[str, Any] | None = None
    confidence: float = 1.0
    source_episode_id: UUID | None = None


class RelationRequest(BaseModel):
    subject_id: UUID
    predicate: str
    object_id: UUID
    metadata: dict[str, Any] | None = None
    confidence: float = 1.0
    source_episode_id: UUID | None = None


class ReflectionRequest(BaseModel):
    content: str
    reflection_type: str = "summary"
    source_episode_ids: list[UUID] | None = None
    metadata: dict[str, Any] | None = None


class RevisionRequest(BaseModel):
    target_type: str
    target_id: UUID
    action: str
    proposed_changes: dict[str, Any]
    reason: str


class ReviewRequest(BaseModel):
    review_notes: str | None = None


# ── Episodes ───────────────────────────────────────────────────────────


@app.post("/api/v1/episodes")
async def store_episode(req: StoreEpisodeRequest):
    episode = await _manager.episodic.store(
        req.content, req.source, req.session_id, req.metadata
    )
    return episode.to_dict()


@app.post("/api/v1/episodes/search")
async def search_episodes(req: SearchRequest):
    results = await _manager.episodic.recall(
        req.query, req.limit, req.min_similarity, req.source, req.session_id
    )
    return {
        "results": [
            {"episode": ep.to_dict(), "similarity": round(sim, 4)}
            for ep, sim in results
        ]
    }


@app.get("/api/v1/episodes/recent")
async def recent_episodes(
    limit: int = Query(10, ge=1, le=100),
    session_id: str | None = Query(None),
):
    episodes = await _manager.episodic.recall_recent(limit, session_id)
    return {"episodes": [ep.to_dict() for ep in episodes]}


@app.get("/api/v1/episodes/{episode_id}")
async def get_episode(episode_id: UUID):
    episode = await _manager.episodic.get(episode_id)
    if episode is None:
        raise HTTPException(404, "Episode not found")
    return episode.to_dict()


@app.delete("/api/v1/episodes/{episode_id}")
async def delete_episode(episode_id: UUID):
    deleted = await _manager.episodic.delete(episode_id)
    if not deleted:
        raise HTTPException(404, "Episode not found")
    return {"deleted": True}


# ── Entities ───────────────────────────────────────────────────────────


@app.post("/api/v1/entities")
async def create_entity(req: EntityRequest):
    entity = await _manager.semantic.add_entity(
        req.name,
        req.entity_type,
        req.description,
        req.metadata,
        req.confidence,
        req.source_episode_id,
    )
    return entity.to_dict()


@app.post("/api/v1/entities/search")
async def search_entities(
    query: str,
    entity_type: str | None = Query(None),
    limit: int = Query(10, ge=1, le=100),
    semantic: bool = Query(True),
):
    results = await _manager.semantic.find_entities(
        query, entity_type, limit, semantic
    )
    return {
        "entities": [
            {"entity": ent.to_dict(), "similarity": round(sim, 4)}
            for ent, sim in results
        ]
    }


@app.get("/api/v1/entities/{entity_id}")
async def get_entity(entity_id: UUID):
    entity = await _manager.semantic.get_entity(entity_id)
    if entity is None:
        raise HTTPException(404, "Entity not found")
    return entity.to_dict()


@app.delete("/api/v1/entities/{entity_id}")
async def delete_entity(entity_id: UUID):
    deleted = await _manager.semantic.delete_entity(entity_id)
    if not deleted:
        raise HTTPException(404, "Entity not found")
    return {"deleted": True}


# ── Relations ──────────────────────────────────────────────────────────


@app.post("/api/v1/relations")
async def create_relation(req: RelationRequest):
    relation = await _manager.semantic.add_relation(
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
    entity_id: UUID | None = Query(None),
    predicate: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    relations = await _manager.semantic.get_relations(
        entity_id=entity_id, predicate=predicate, limit=limit
    )
    return {"relations": [r.to_dict() for r in relations]}


@app.delete("/api/v1/relations/{relation_id}")
async def delete_relation(relation_id: UUID):
    deleted = await _manager.semantic.delete_relation(relation_id)
    if not deleted:
        raise HTTPException(404, "Relation not found")
    return {"deleted": True}


@app.get("/api/v1/entities/{entity_id}/graph")
async def entity_graph(entity_id: UUID, max_depth: int = Query(2, ge=1, le=3)):
    entity = await _manager.semantic.get_entity(entity_id)
    if entity is None:
        raise HTTPException(404, "Entity not found")
    subgraph = await _manager.semantic.traverse(entity_id, max_depth)
    return {"entity": entity.to_dict(), "connections": subgraph}


# ── Reflections ────────────────────────────────────────────────────────


@app.post("/api/v1/reflections")
async def create_reflection(req: ReflectionRequest):
    reflection = await _manager.reflection.create(
        req.content, req.reflection_type, req.source_episode_ids, req.metadata
    )
    return reflection.to_dict()


@app.post("/api/v1/reflections/search")
async def search_reflections(
    query: str,
    reflection_type: str | None = Query(None),
    limit: int = Query(10, ge=1, le=100),
):
    results = await _manager.reflection.search(query, reflection_type, limit)
    return {
        "reflections": [
            {"reflection": ref.to_dict(), "similarity": round(sim, 4)}
            for ref, sim in results
        ]
    }


@app.get("/api/v1/reflections")
async def list_reflections(
    reflection_type: str | None = Query(None),
    limit: int = Query(10, ge=1, le=100),
):
    reflections = await _manager.reflection.list_recent(reflection_type, limit)
    return {"reflections": [r.to_dict() for r in reflections]}


# ── Revision Proposals ─────────────────────────────────────────────────


@app.post("/api/v1/revisions")
async def create_revision(req: RevisionRequest):
    proposal = await _manager.reflection.propose_revision(
        req.target_type,
        req.target_id,
        req.action,
        req.proposed_changes,
        req.reason,
    )
    return proposal.to_dict()


@app.get("/api/v1/revisions")
async def list_revisions(
    status: str = Query("pending"),
    limit: int = Query(50, ge=1, le=200),
):
    proposals = await _manager.reflection.list_revisions(status, limit)
    return {"proposals": [p.to_dict() for p in proposals]}


@app.put("/api/v1/revisions/{revision_id}/approve")
async def approve_revision(revision_id: UUID, req: ReviewRequest | None = None):
    notes = req.review_notes if req else None
    proposal = await _manager.reflection.approve_revision(revision_id, notes)
    if proposal is None:
        raise HTTPException(404, "Pending revision not found")
    return proposal.to_dict()


@app.put("/api/v1/revisions/{revision_id}/reject")
async def reject_revision(revision_id: UUID, req: ReviewRequest | None = None):
    notes = req.review_notes if req else None
    proposal = await _manager.reflection.reject_revision(revision_id, notes)
    if proposal is None:
        raise HTTPException(404, "Pending revision not found")
    return proposal.to_dict()


# ── System ─────────────────────────────────────────────────────────────


@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}


@app.get("/api/v1/stats")
async def stats():
    async with _manager.pool.acquire() as conn:
        counts = await conn.fetchrow(
            """SELECT
                   (SELECT count(*) FROM episodes) AS episodes,
                   (SELECT count(*) FROM entities) AS entities,
                   (SELECT count(*) FROM relations) AS relations,
                   (SELECT count(*) FROM reflections) AS reflections,
                   (SELECT count(*) FROM revision_proposals
                    WHERE status = 'pending') AS pending_revisions"""
        )
    return dict(counts)


def run_api() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
