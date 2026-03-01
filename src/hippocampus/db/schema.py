"""Database schema DDL and migration helper."""

from __future__ import annotations

import asyncpg

from hippocampus.config import Settings


def get_schema_sql(dim: int) -> str:
    """Return the full DDL for all Hippocampus tables and indexes.

    Args:
        dim: Vector dimension used for embedding columns.
    """
    dim = int(dim)  # belt-and-suspenders against non-int input
    return f"""
    -- Extensions
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

    -- Trigger function for updated_at columns
    CREATE OR REPLACE FUNCTION update_updated_at()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = NOW();
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;

    -- Episodic Memory
    CREATE TABLE IF NOT EXISTS episodes (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        owner_id TEXT NOT NULL,
        session_id TEXT,
        source TEXT NOT NULL,
        content TEXT NOT NULL,
        embedding vector({dim}),
        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_episodes_embedding
        ON episodes USING hnsw (embedding vector_cosine_ops);
    CREATE INDEX IF NOT EXISTS idx_episodes_owner ON episodes (owner_id);
    CREATE INDEX IF NOT EXISTS idx_episodes_created_at ON episodes (created_at);
    CREATE INDEX IF NOT EXISTS idx_episodes_session_id ON episodes (session_id);
    CREATE INDEX IF NOT EXISTS idx_episodes_source ON episodes (source);

    DROP TRIGGER IF EXISTS trg_episodes_updated_at ON episodes;
    CREATE TRIGGER trg_episodes_updated_at BEFORE UPDATE ON episodes
        FOR EACH ROW EXECUTE FUNCTION update_updated_at();

    -- Knowledge Graph: Entities
    CREATE TABLE IF NOT EXISTS entities (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        owner_id TEXT NOT NULL,
        name TEXT NOT NULL,
        entity_type TEXT NOT NULL,
        description TEXT,
        embedding vector({dim}),
        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        confidence FLOAT NOT NULL DEFAULT 1.0
            CHECK (confidence >= 0 AND confidence <= 1),
        source_episode_id UUID REFERENCES episodes(id) ON DELETE SET NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_entities_embedding
        ON entities USING hnsw (embedding vector_cosine_ops);
    CREATE INDEX IF NOT EXISTS idx_entities_owner ON entities (owner_id);
    CREATE INDEX IF NOT EXISTS idx_entities_name ON entities (name);
    CREATE INDEX IF NOT EXISTS idx_entities_type ON entities (entity_type);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_owner_name_type
        ON entities (owner_id, name, entity_type);

    DROP TRIGGER IF EXISTS trg_entities_updated_at ON entities;
    CREATE TRIGGER trg_entities_updated_at BEFORE UPDATE ON entities
        FOR EACH ROW EXECUTE FUNCTION update_updated_at();

    -- Knowledge Graph: Relations
    CREATE TABLE IF NOT EXISTS relations (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        owner_id TEXT NOT NULL,
        subject_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
        predicate TEXT NOT NULL,
        object_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        confidence FLOAT NOT NULL DEFAULT 1.0
            CHECK (confidence >= 0 AND confidence <= 1),
        source_episode_id UUID REFERENCES episodes(id) ON DELETE SET NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_relations_owner ON relations (owner_id);
    CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations (subject_id);
    CREATE INDEX IF NOT EXISTS idx_relations_object ON relations (object_id);
    CREATE INDEX IF NOT EXISTS idx_relations_predicate ON relations (predicate);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_relations_triple
        ON relations (subject_id, predicate, object_id);

    DROP TRIGGER IF EXISTS trg_relations_updated_at ON relations;
    CREATE TRIGGER trg_relations_updated_at BEFORE UPDATE ON relations
        FOR EACH ROW EXECUTE FUNCTION update_updated_at();

    -- Reflections / Meta-Memory
    CREATE TABLE IF NOT EXISTS reflections (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        owner_id TEXT NOT NULL,
        content TEXT NOT NULL,
        reflection_type TEXT NOT NULL,
        embedding vector({dim}),
        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_reflections_embedding
        ON reflections USING hnsw (embedding vector_cosine_ops);
    CREATE INDEX IF NOT EXISTS idx_reflections_owner ON reflections (owner_id);
    CREATE INDEX IF NOT EXISTS idx_reflections_type
        ON reflections (reflection_type);

    -- Link reflections to source episodes
    CREATE TABLE IF NOT EXISTS reflection_episodes (
        reflection_id UUID NOT NULL REFERENCES reflections(id) ON DELETE CASCADE,
        episode_id UUID NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
        PRIMARY KEY (reflection_id, episode_id)
    );

    -- Revision Proposals (Human Oversight)
    CREATE TABLE IF NOT EXISTS revision_proposals (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        owner_id TEXT NOT NULL,
        target_type TEXT NOT NULL
            CHECK (target_type IN ('entity', 'relation')),
        target_id UUID NOT NULL,
        action TEXT NOT NULL
            CHECK (action IN ('update', 'delete', 'merge')),
        proposed_changes JSONB NOT NULL,
        reason TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending'
            CHECK (status IN ('pending', 'approved', 'rejected')),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        reviewed_at TIMESTAMPTZ,
        review_notes TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_revisions_owner
        ON revision_proposals (owner_id);
    CREATE INDEX IF NOT EXISTS idx_revisions_status
        ON revision_proposals (status);
    CREATE INDEX IF NOT EXISTS idx_revisions_target
        ON revision_proposals (target_type, target_id);
    """


async def ensure_schema(pool: asyncpg.Pool, settings: Settings) -> None:
    """Create all tables and indexes if they do not already exist."""
    sql = get_schema_sql(settings.embedding_dimensions)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(sql)
