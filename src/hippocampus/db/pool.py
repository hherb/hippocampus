"""asyncpg connection pool with pgvector and JSONB codec registration."""

from __future__ import annotations

import json

import asyncpg
from pgvector.asyncpg import register_vector

from hippocampus.config import Settings


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Register pgvector types and a JSONB codec on each new connection."""
    await register_vector(conn)
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )


async def create_pool(settings: Settings) -> asyncpg.Pool:
    """Create and return an asyncpg connection pool."""
    pool = await asyncpg.create_pool(
        settings.db_url,
        min_size=settings.db_min_connections,
        max_size=settings.db_max_connections,
        init=_init_connection,
    )
    if pool is None:
        raise RuntimeError("Failed to create database connection pool")
    return pool
