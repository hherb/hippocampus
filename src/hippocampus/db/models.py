from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass
class Episode:
    id: UUID
    source: str
    content: str
    created_at: datetime
    updated_at: datetime
    session_id: str | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row) -> Episode:
        return cls(
            id=row["id"],
            source=row["source"],
            content=row["content"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            session_id=row["session_id"],
            embedding=row["embedding"].tolist() if row["embedding"] is not None else None,
            metadata=row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "source": self.source,
            "content": self.content,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class Entity:
    id: UUID
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
    def from_row(cls, row) -> Entity:
        return cls(
            id=row["id"],
            name=row["name"],
            entity_type=row["entity_type"],
            description=row["description"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            embedding=row["embedding"].tolist() if row["embedding"] is not None else None,
            metadata=row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}"),
            confidence=row["confidence"],
            source_episode_id=row["source_episode_id"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "source_episode_id": str(self.source_episode_id) if self.source_episode_id else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class Relation:
    id: UUID
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
    def from_row(cls, row) -> Relation:
        return cls(
            id=row["id"],
            subject_id=row["subject_id"],
            predicate=row["predicate"],
            object_id=row["object_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}"),
            confidence=row["confidence"],
            source_episode_id=row["source_episode_id"],
            subject_name=row.get("subject_name"),
            object_name=row.get("object_name"),
        )

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": str(self.id),
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


@dataclass
class Reflection:
    id: UUID
    content: str
    reflection_type: str
    created_at: datetime
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_episode_ids: list[UUID] = field(default_factory=list)

    @classmethod
    def from_row(cls, row) -> Reflection:
        return cls(
            id=row["id"],
            content=row["content"],
            reflection_type=row["reflection_type"],
            created_at=row["created_at"],
            embedding=row["embedding"].tolist() if row["embedding"] is not None else None,
            metadata=row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "content": self.content,
            "reflection_type": self.reflection_type,
            "metadata": self.metadata,
            "source_episode_ids": [str(eid) for eid in self.source_episode_ids],
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class RevisionProposal:
    id: UUID
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
    def from_row(cls, row) -> RevisionProposal:
        return cls(
            id=row["id"],
            target_type=row["target_type"],
            target_id=row["target_id"],
            action=row["action"],
            proposed_changes=(
                row["proposed_changes"]
                if isinstance(row["proposed_changes"], dict)
                else json.loads(row["proposed_changes"] or "{}")
            ),
            reason=row["reason"],
            status=row["status"],
            created_at=row["created_at"],
            reviewed_at=row["reviewed_at"],
            review_notes=row["review_notes"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
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
