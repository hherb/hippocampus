"""Data models for memory records, entities, relations, and reflections."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg


# ── Helpers ────────────────────────────────────────────────────────────


def _parse_metadata(raw: Any) -> dict[str, Any]:
    """Normalise a metadata value from a database row to a plain dict."""
    if isinstance(raw, dict):
        return raw
    return json.loads(raw or "{}")


def _parse_embedding(raw: Any) -> list[float] | None:
    """Convert a pgvector numpy array to a plain list, or return *None*."""
    if raw is None:
        return None
    return raw.tolist()


# ── Episode ────────────────────────────────────────────────────────────


@dataclass
class Episode:
    """A single episodic memory record."""

    id: UUID
    owner_id: str
    source: str
    content: str
    created_at: datetime
    updated_at: datetime
    session_id: str | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: asyncpg.Record) -> Episode:
        """Construct an :class:`Episode` from a database row."""
        return cls(
            id=row["id"],
            owner_id=row["owner_id"],
            source=row["source"],
            content=row["content"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            session_id=row["session_id"],
            embedding=_parse_embedding(row["embedding"]),
            metadata=_parse_metadata(row["metadata"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary (excludes embedding)."""
        return {
            "id": str(self.id),
            "owner_id": self.owner_id,
            "source": self.source,
            "content": self.content,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# ── Entity ─────────────────────────────────────────────────────────────


@dataclass
class Entity:
    """A named entity in the knowledge graph."""

    id: UUID
    owner_id: str
    name: str
    entity_type: str
    created_at: datetime
    updated_at: datetime
    description: str | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source_episode_id: UUID | None = None

    @classmethod
    def from_row(cls, row: asyncpg.Record) -> Entity:
        """Construct an :class:`Entity` from a database row."""
        return cls(
            id=row["id"],
            owner_id=row["owner_id"],
            name=row["name"],
            entity_type=row["entity_type"],
            description=row["description"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            embedding=_parse_embedding(row["embedding"]),
            metadata=_parse_metadata(row["metadata"]),
            confidence=row["confidence"],
            source_episode_id=row["source_episode_id"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary (excludes embedding)."""
        return {
            "id": str(self.id),
            "owner_id": self.owner_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "source_episode_id": str(self.source_episode_id) if self.source_episode_id else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# ── Relation ───────────────────────────────────────────────────────────


@dataclass
class Relation:
    """A directed relationship between two entities."""

    id: UUID
    owner_id: str
    subject_id: UUID
    predicate: str
    object_id: UUID
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source_episode_id: UUID | None = None
    # Populated by joins
    subject_name: str | None = None
    object_name: str | None = None

    @classmethod
    def from_row(cls, row: asyncpg.Record) -> Relation:
        """Construct a :class:`Relation` from a database row."""
        return cls(
            id=row["id"],
            owner_id=row["owner_id"],
            subject_id=row["subject_id"],
            predicate=row["predicate"],
            object_id=row["object_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_parse_metadata(row["metadata"]),
            confidence=row["confidence"],
            source_episode_id=row["source_episode_id"],
            subject_name=row.get("subject_name"),
            object_name=row.get("object_name"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        d: dict[str, Any] = {
            "id": str(self.id),
            "owner_id": self.owner_id,
            "subject_id": str(self.subject_id),
            "predicate": self.predicate,
            "object_id": str(self.object_id),
            "metadata": self.metadata,
            "confidence": self.confidence,
            "source_episode_id": str(self.source_episode_id) if self.source_episode_id else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if self.subject_name is not None:
            d["subject_name"] = self.subject_name
        if self.object_name is not None:
            d["object_name"] = self.object_name
        return d


# ── Reflection ─────────────────────────────────────────────────────────


@dataclass
class Reflection:
    """A distilled insight, summary, or pattern derived from episodic memories."""

    id: UUID
    owner_id: str
    content: str
    reflection_type: str
    created_at: datetime
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_episode_ids: list[UUID] = field(default_factory=list)

    @classmethod
    def from_row(cls, row: asyncpg.Record) -> Reflection:
        """Construct a :class:`Reflection` from a database row."""
        return cls(
            id=row["id"],
            owner_id=row["owner_id"],
            content=row["content"],
            reflection_type=row["reflection_type"],
            created_at=row["created_at"],
            embedding=_parse_embedding(row["embedding"]),
            metadata=_parse_metadata(row["metadata"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary (excludes embedding)."""
        return {
            "id": str(self.id),
            "owner_id": self.owner_id,
            "content": self.content,
            "reflection_type": self.reflection_type,
            "metadata": self.metadata,
            "source_episode_ids": [str(eid) for eid in self.source_episode_ids],
            "created_at": self.created_at.isoformat(),
        }


# ── RevisionProposal ──────────────────────────────────────────────────


@dataclass
class RevisionProposal:
    """A proposed change to the knowledge graph awaiting human review."""

    id: UUID
    owner_id: str
    target_type: str
    target_id: UUID
    action: str
    proposed_changes: dict[str, Any]
    reason: str
    status: str
    created_at: datetime
    reviewed_at: datetime | None = None
    review_notes: str | None = None

    @classmethod
    def from_row(cls, row: asyncpg.Record) -> RevisionProposal:
        """Construct a :class:`RevisionProposal` from a database row."""
        return cls(
            id=row["id"],
            owner_id=row["owner_id"],
            target_type=row["target_type"],
            target_id=row["target_id"],
            action=row["action"],
            proposed_changes=_parse_metadata(row["proposed_changes"]),
            reason=row["reason"],
            status=row["status"],
            created_at=row["created_at"],
            reviewed_at=row["reviewed_at"],
            review_notes=row["review_notes"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "id": str(self.id),
            "owner_id": self.owner_id,
            "target_type": self.target_type,
            "target_id": str(self.target_id),
            "action": self.action,
            "proposed_changes": self.proposed_changes,
            "reason": self.reason,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "review_notes": self.review_notes,
        }
