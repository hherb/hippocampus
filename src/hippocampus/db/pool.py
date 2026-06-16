"""asyncpg connection pool with pgvector and JSONB codec registration."""

from __future__ import annotations

import json
import logging

import asyncpg
from pgvector.asyncpg import register_vector

from hippocampus.config import Settings

log = logging.getLogger(__name__)

_VALID_ITERATIVE_SCAN = {"off", "relaxed_order", "strict_order"}


def _make_init(settings: Settings):
    """Build an asyncpg connection initialiser bound to *settings*."""

    async def _init_connection(conn: asyncpg.Connection) -> None:
        """Register pgvector types/codecs and apply HNSW search tuning."""
        await register_vector(conn)
        await conn.set_type_codec(
            "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
        )

        # Tune HNSW search so post-filtered queries (owner_id / type) still
        # return up to LIMIT rows. These GUCs require pgvector >= 0.8 for
        # iterative scan; if unsupported, skip rather than break the pool.
        scan = settings.hnsw_iterative_scan
        if scan not in _VALID_ITERATIVE_SCAN:
            log.warning(
                "Invalid hnsw_iterative_scan %r; expected one of %s. Skipping.",
                scan,
                sorted(_VALID_ITERATIVE_SCAN),
            )
            scan = None

        for stmt in (
            f"SET hnsw.iterative_scan = '{scan}'" if scan else None,
            f"SET hnsw.ef_search = {int(settings.hnsw_ef_search)}",
        ):
            if stmt is None:
                continue
            try:
                await conn.execute(stmt)
            except asyncpg.PostgresError as exc:
                log.warning(
                    "Could not apply '%s' (pgvector too old?): %s", stmt, exc
                )

    return _init_connection


async def create_pool(settings: Settings) -> asyncpg.Pool:
    """Create and return an asyncpg connection pool."""
    pool = await asyncpg.create_pool(
        settings.db_url,
        min_size=settings.db_min_connections,
        max_size=settings.db_max_connections,
        init=_make_init(settings),
    )
    if pool is None:
        raise RuntimeError("Failed to create database connection pool")
    return pool
