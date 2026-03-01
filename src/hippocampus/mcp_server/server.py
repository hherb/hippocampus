from __future__ import annotations

import json
from contextlib import asynccontextmanager
from uuid import UUID

from mcp.server.fastmcp import FastMCP

from hippocampus.config import settings
from hippocampus.db.pool import create_pool
from hippocampus.db.schema import ensure_schema
from hippocampus.embeddings.ollama import OllamaEmbedding
from hippocampus.memory import MemoryManager

_manager: MemoryManager | None = None
_embedder: OllamaEmbedding | None = None


@asynccontextmanager
async def lifespan(server: FastMCP):
    global _manager, _embedder
    pool = await create_pool(settings)
    await ensure_schema(pool, settings)
    _embedder = OllamaEmbedding(settings)
    _manager = MemoryManager(pool, _embedder)
    try:
        yield
    finally:
        await pool.close()
        await _embedder.close()
        _manager = None
        _embedder = None


mcp = FastMCP(
    "hippocampus",
    instructions=(
        "Hippocampus is your persistent memory system. Use it to remember "
        "important information across conversations, build a knowledge graph "
        "of entities and relationships, and create reflections that distill "
        "understanding over time."
    ),
    lifespan=lifespan,
)


def _json(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


# ── Episodic Memory ────────────────────────────────────────────────────


@mcp.tool()
async def remember(
    content: str,
    source: str = "conversation",
    session_id: str | None = None,
    metadata: str | None = None,
) -> str:
    """Store a new episodic memory. Use this when information is worth
    remembering across conversations — facts, preferences, decisions,
    observations, or any context the user may want recalled later.

    Args:
        content: The information to remember.
        source: Origin type (conversation, observation, api, document).
        session_id: Optional session identifier for grouping.
        metadata: Optional JSON string of additional key-value pairs.
    """
    meta = json.loads(metadata) if metadata else None
    episode = await _manager.episodic.store(content, source, session_id, meta)
    return _json({"stored": episode.to_dict()})


@mcp.tool()
async def recall(
    query: str,
    limit: int = 5,
    min_similarity: float = 0.3,
    source: str | None = None,
    session_id: str | None = None,
) -> str:
    """Search episodic memories by semantic similarity. Use this to find
    relevant memories related to a topic or question.

    Args:
        query: The search query (natural language).
        limit: Maximum number of results.
        min_similarity: Minimum cosine similarity threshold (0.0-1.0).
        source: Filter by source type.
        session_id: Filter by session.
    """
    results = await _manager.episodic.recall(
        query, limit, min_similarity, source, session_id
    )
    return _json({
        "results": [
            {"episode": ep.to_dict(), "similarity": round(sim, 4)}
            for ep, sim in results
        ]
    })


@mcp.tool()
async def recall_recent(
    limit: int = 10,
    session_id: str | None = None,
) -> str:
    """Retrieve the most recent episodic memories, optionally filtered
    by session.

    Args:
        limit: Maximum number of results.
        session_id: Filter by session.
    """
    episodes = await _manager.episodic.recall_recent(limit, session_id)
    return _json({"episodes": [ep.to_dict() for ep in episodes]})


# ── Knowledge Graph ────────────────────────────────────────────────────


@mcp.tool()
async def learn_entity(
    name: str,
    entity_type: str,
    description: str | None = None,
    metadata: str | None = None,
    confidence: float = 1.0,
) -> str:
    """Add or update an entity in the knowledge graph. Entities represent
    people, concepts, tools, projects, preferences, or any named thing
    worth tracking.

    If an entity with the same name and type already exists, it is updated
    (descriptions merged, confidence increased).

    Args:
        name: Entity name (e.g. "Python", "Alice", "dark mode preference").
        entity_type: Category (person, concept, tool, project, preference, etc.).
        description: What this entity is or means.
        metadata: Optional JSON string of additional properties.
        confidence: Confidence score 0.0-1.0.
    """
    meta = json.loads(metadata) if metadata else None
    entity = await _manager.semantic.add_entity(
        name, entity_type, description, meta, confidence
    )
    return _json({"entity": entity.to_dict()})


@mcp.tool()
async def learn_relation(
    subject: str,
    subject_type: str,
    predicate: str,
    object_: str,
    object_type: str,
    confidence: float = 1.0,
    metadata: str | None = None,
) -> str:
    """Add a relationship between two entities in the knowledge graph.
    Creates entities if they don't exist yet.

    Args:
        subject: Name of the subject entity.
        subject_type: Type of the subject entity.
        predicate: The relationship (e.g. "uses", "prefers", "knows", "is_a", "depends_on").
        object_: Name of the object entity.
        object_type: Type of the object entity.
        confidence: Confidence score 0.0-1.0.
        metadata: Optional JSON string of additional properties.
    """
    subj = await _manager.semantic.get_entity_by_name(subject, subject_type)
    if subj is None:
        subj = await _manager.semantic.add_entity(subject, subject_type)

    obj = await _manager.semantic.get_entity_by_name(object_, object_type)
    if obj is None:
        obj = await _manager.semantic.add_entity(object_, object_type)

    meta = json.loads(metadata) if metadata else None
    relation = await _manager.semantic.add_relation(
        subj.id, predicate, obj.id, meta, confidence
    )
    return _json({
        "relation": {
            **relation.to_dict(),
            "subject_name": subject,
            "object_name": object_,
        }
    })


@mcp.tool()
async def find_entities(
    query: str,
    entity_type: str | None = None,
    limit: int = 10,
    semantic: bool = True,
) -> str:
    """Search for entities in the knowledge graph.

    Args:
        query: Search query (semantic or text match).
        entity_type: Filter by entity type.
        limit: Maximum results.
        semantic: Use semantic similarity (True) or text matching (False).
    """
    results = await _manager.semantic.find_entities(
        query, entity_type, limit, semantic
    )
    return _json({
        "entities": [
            {"entity": ent.to_dict(), "similarity": round(sim, 4)}
            for ent, sim in results
        ]
    })


@mcp.tool()
async def query_relations(
    entity_name: str | None = None,
    entity_type: str | None = None,
    predicate: str | None = None,
    limit: int = 50,
) -> str:
    """Query relationships in the knowledge graph. At least one of
    entity_name or predicate must be provided.

    Args:
        entity_name: Filter by entity name (as subject or object).
        entity_type: Required if entity_name is provided.
        predicate: Filter by relationship type.
        limit: Maximum results.
    """
    entity_id = None
    if entity_name and entity_type:
        entity = await _manager.semantic.get_entity_by_name(entity_name, entity_type)
        if entity:
            entity_id = entity.id

    relations = await _manager.semantic.get_relations(
        entity_id=entity_id, predicate=predicate, limit=limit
    )
    return _json({"relations": [r.to_dict() for r in relations]})


@mcp.tool()
async def explore_connections(
    entity_name: str,
    entity_type: str,
    max_depth: int = 2,
) -> str:
    """Traverse the knowledge graph from a starting entity, showing all
    connected entities and relationships up to a given depth.

    Args:
        entity_name: Starting entity name.
        entity_type: Starting entity type.
        max_depth: How many hops to traverse (1-3).
    """
    entity = await _manager.semantic.get_entity_by_name(entity_name, entity_type)
    if entity is None:
        return _json({"error": f"Entity '{entity_name}' ({entity_type}) not found"})

    max_depth = min(max_depth, 3)
    subgraph = await _manager.semantic.traverse(entity.id, max_depth)
    return _json({"entity": entity.to_dict(), "connections": subgraph})


# ── Reflections ────────────────────────────────────────────────────────


@mcp.tool()
async def reflect(
    content: str,
    reflection_type: str = "summary",
    source_episode_ids: str | None = None,
    metadata: str | None = None,
) -> str:
    """Create a reflection — a distilled insight, summary, pattern, or
    noted contradiction derived from episodic memories. Reflections are
    the 'notebook revisions' that consolidate understanding over time.

    Args:
        content: The reflection text.
        reflection_type: Type (summary, insight, contradiction, pattern).
        source_episode_ids: JSON array of episode UUIDs this reflects on.
        metadata: Optional JSON string of additional properties.
    """
    ep_ids = None
    if source_episode_ids:
        ep_ids = [UUID(eid) for eid in json.loads(source_episode_ids)]
    meta = json.loads(metadata) if metadata else None

    reflection = await _manager.reflection.create(
        content, reflection_type, ep_ids, meta
    )
    return _json({"reflection": reflection.to_dict()})


@mcp.tool()
async def search_reflections(
    query: str,
    reflection_type: str | None = None,
    limit: int = 5,
) -> str:
    """Search reflections by semantic similarity.

    Args:
        query: The search query.
        reflection_type: Filter by type (summary, insight, contradiction, pattern).
        limit: Maximum results.
    """
    results = await _manager.reflection.search(query, reflection_type, limit)
    return _json({
        "reflections": [
            {"reflection": ref.to_dict(), "similarity": round(sim, 4)}
            for ref, sim in results
        ]
    })


# ── Revision Proposals (Human Oversight) ───────────────────────────────


@mcp.tool()
async def propose_revision(
    target_type: str,
    target_name: str,
    target_entity_type: str,
    action: str,
    proposed_changes: str,
    reason: str,
) -> str:
    """Propose a change to the knowledge graph that requires human approval.
    Use this for significant changes like merging entities, deleting
    established knowledge, or correcting contradictions.

    Args:
        target_type: "entity" or "relation".
        target_name: Name of the target entity.
        target_entity_type: Type of the target entity.
        action: "update", "delete", or "merge".
        proposed_changes: JSON string describing the changes.
        reason: Why this revision is proposed.
    """
    entity = await _manager.semantic.get_entity_by_name(
        target_name, target_entity_type
    )
    if entity is None:
        return _json({"error": f"Entity '{target_name}' ({target_entity_type}) not found"})

    changes = json.loads(proposed_changes)
    proposal = await _manager.reflection.propose_revision(
        target_type, entity.id, action, changes, reason
    )
    return _json({"proposal": proposal.to_dict()})


@mcp.tool()
async def pending_revisions(limit: int = 20) -> str:
    """List pending revision proposals awaiting human review.

    Args:
        limit: Maximum results.
    """
    proposals = await _manager.reflection.list_revisions("pending", limit)
    return _json({"proposals": [p.to_dict() for p in proposals]})


def run_stdio() -> None:
    mcp.run(transport="stdio")


def run_sse() -> None:
    mcp.run(
        transport="sse",
        sse_params={
            "host": settings.mcp_sse_host,
            "port": settings.mcp_sse_port,
        },
    )
