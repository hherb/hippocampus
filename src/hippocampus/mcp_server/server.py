"""MCP (Model Context Protocol) server exposing Hippocampus memory tools."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from mcp.server.fastmcp import FastMCP

from hippocampus.config import SIMILARITY_PRECISION, settings
from hippocampus.db.pool import create_pool
from hippocampus.db.schema import ensure_schema
from hippocampus.embeddings.ollama import OllamaEmbedding
from hippocampus.memory import MemoryManager

_manager: MemoryManager | None = None
_embedder: OllamaEmbedding | None = None

# Set at startup; overridable via CLI --owner or HIPPOCAMPUS_DEFAULT_OWNER
_owner_id: str = settings.default_owner


def _require_manager() -> MemoryManager:
    """Return the active MemoryManager, or raise if the server is not ready."""
    if _manager is None:
        raise RuntimeError("Hippocampus server is not initialised yet")
    return _manager


def _parse_json(raw: str | None, field_name: str = "value") -> Any:
    """Parse a JSON string from tool input, returning *None* for empty input."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in '{field_name}': {exc.args[0]}"
        ) from exc


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Initialise the database pool, schema, and embedding provider."""
    global _manager, _embedder
    pool = await create_pool(settings)
    try:
        await ensure_schema(pool, settings)
        _embedder = OllamaEmbedding(settings)
        _manager = MemoryManager(pool, _embedder, _owner_id)
    except BaseException:
        await pool.close()
        raise
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
    host=settings.mcp_sse_host,
    port=settings.mcp_sse_port,
)


def _json(obj: Any) -> str:
    """Serialise *obj* to a pretty-printed JSON string."""
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
    mgr = _require_manager()
    meta = _parse_json(metadata, "metadata")
    episode = await mgr.episodic.store(content, source, session_id, meta)
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
    mgr = _require_manager()
    results = await mgr.episodic.recall(
        query, limit, min_similarity, source, session_id
    )
    return _json({
        "results": [
            {"episode": ep.to_dict(), "similarity": round(sim, SIMILARITY_PRECISION)}
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
    mgr = _require_manager()
    episodes = await mgr.episodic.recall_recent(limit, session_id)
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
    mgr = _require_manager()
    meta = _parse_json(metadata, "metadata")
    entity = await mgr.semantic.add_entity(
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
    mgr = _require_manager()
    subj = await mgr.semantic.get_entity_by_name(subject, subject_type)
    if subj is None:
        subj = await mgr.semantic.add_entity(subject, subject_type)

    obj = await mgr.semantic.get_entity_by_name(object_, object_type)
    if obj is None:
        obj = await mgr.semantic.add_entity(object_, object_type)

    meta = _parse_json(metadata, "metadata")
    relation = await mgr.semantic.add_relation(
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
    mgr = _require_manager()
    results = await mgr.semantic.find_entities(
        query, entity_type, limit, semantic
    )
    return _json({
        "entities": [
            {"entity": ent.to_dict(), "similarity": round(sim, SIMILARITY_PRECISION)}
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
    mgr = _require_manager()

    if not entity_name and not predicate:
        return _json({
            "error": "Provide at least one of 'entity_name' or 'predicate'"
        })

    entity_id = None
    if entity_name:
        if not entity_type:
            return _json({
                "error": "'entity_type' is required when 'entity_name' is given"
            })
        entity = await mgr.semantic.get_entity_by_name(entity_name, entity_type)
        if entity is None:
            return _json({
                "error": f"Entity '{entity_name}' ({entity_type}) not found",
                "relations": [],
            })
        entity_id = entity.id

    relations = await mgr.semantic.get_relations(
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
    mgr = _require_manager()
    entity = await mgr.semantic.get_entity_by_name(entity_name, entity_type)
    if entity is None:
        return _json({"error": f"Entity '{entity_name}' ({entity_type}) not found"})

    max_depth = max(1, min(max_depth, 3))
    subgraph = await mgr.semantic.traverse(entity.id, max_depth)
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
    mgr = _require_manager()
    ep_ids = None
    if source_episode_ids:
        raw_ids = _parse_json(source_episode_ids, "source_episode_ids")
        ep_ids = [UUID(eid) for eid in raw_ids]
    meta = _parse_json(metadata, "metadata")

    reflection = await mgr.reflection.create(
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
    mgr = _require_manager()
    results = await mgr.reflection.search(query, reflection_type, limit)
    return _json({
        "reflections": [
            {"reflection": ref.to_dict(), "similarity": round(sim, SIMILARITY_PRECISION)}
            for ref, sim in results
        ]
    })


# ── Revision Proposals (Human Oversight) ───────────────────────────────


@mcp.tool()
async def propose_revision(
    target_type: str,
    action: str,
    proposed_changes: str,
    reason: str,
    target_name: str | None = None,
    target_entity_type: str | None = None,
    subject: str | None = None,
    subject_type: str | None = None,
    predicate: str | None = None,
    object_: str | None = None,
    object_type: str | None = None,
) -> str:
    """Propose a change to the knowledge graph that requires human approval.
    Use this for significant changes like merging entities, deleting
    established knowledge, or correcting contradictions.

    To target an entity, provide ``target_name`` and ``target_entity_type``.
    To target a relation, provide the full triple: ``subject`` / ``subject_type``
    / ``predicate`` / ``object_`` / ``object_type``.

    Args:
        target_type: "entity" or "relation".
        action: "update", "delete", or "merge".
        proposed_changes: JSON string describing the changes.
        reason: Why this revision is proposed.
        target_name: Name of the target entity (entity revisions).
        target_entity_type: Type of the target entity (entity revisions).
        subject: Subject entity name (relation revisions).
        subject_type: Subject entity type (relation revisions).
        predicate: Relation predicate (relation revisions).
        object_: Object entity name (relation revisions).
        object_type: Object entity type (relation revisions).
    """
    mgr = _require_manager()

    if target_type == "entity":
        if not (target_name and target_entity_type):
            return _json({
                "error": "target_name and target_entity_type are required "
                         "for entity revisions"
            })
        entity = await mgr.semantic.get_entity_by_name(
            target_name, target_entity_type
        )
        if entity is None:
            return _json({"error": f"Entity '{target_name}' ({target_entity_type}) not found"})
        target_id = entity.id

    elif target_type == "relation":
        if not all([subject, subject_type, predicate, object_, object_type]):
            return _json({
                "error": "subject, subject_type, predicate, object_, and "
                         "object_type are all required for relation revisions"
            })
        subj = await mgr.semantic.get_entity_by_name(subject, subject_type)
        obj = await mgr.semantic.get_entity_by_name(object_, object_type)
        if subj is None or obj is None:
            return _json({"error": "Subject or object entity not found"})
        relation = await mgr.semantic.get_relation_by_triple(
            subj.id, predicate, obj.id
        )
        if relation is None:
            return _json({
                "error": f"Relation '{subject} {predicate} {object_}' not found"
            })
        target_id = relation.id

    else:
        return _json({
            "error": f"Invalid target_type '{target_type}'; "
                     "must be 'entity' or 'relation'"
        })

    changes = _parse_json(proposed_changes, "proposed_changes")
    proposal = await mgr.reflection.propose_revision(
        target_type, target_id, action, changes, reason
    )
    return _json({"proposal": proposal.to_dict()})


@mcp.tool()
async def pending_revisions(limit: int = 20) -> str:
    """List pending revision proposals awaiting human review.

    Args:
        limit: Maximum results.
    """
    mgr = _require_manager()
    proposals = await mgr.reflection.list_revisions("pending", limit)
    return _json({"proposals": [p.to_dict() for p in proposals]})


def run_stdio(owner_id: str | None = None) -> None:
    """Start the MCP server over stdio transport."""
    global _owner_id
    if owner_id:
        _owner_id = owner_id
    mcp.run(transport="stdio")


def run_sse(owner_id: str | None = None) -> None:
    """Start the MCP server over SSE transport.

    Host and port are configured via the ``FastMCP`` constructor
    (sourced from ``settings.mcp_sse_host`` / ``settings.mcp_sse_port``).
    """
    global _owner_id
    if owner_id:
        _owner_id = owner_id
    mcp.run(transport="sse")
