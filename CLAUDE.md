# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hippocampus is a persistent memory retrieval system for LLMs, accessible via MCP (Model Context Protocol) or REST API. It provides three tiers of long-term memory backed by PostgreSQL + pgvector:

- **Episodic memory** — timestamped interaction records with vector embeddings for semantic retrieval
- **Semantic memory / knowledge graph** — entities and relationships with vector search
- **Reflection / meta-memory** — summaries, insights, and human-governed revision proposals

## Development Commands

```bash
# Install from source (uses hatchling build system)
pip install -e .

# Initialize database schema (requires running PostgreSQL with pgvector)
hippocampus init-db

# Run MCP server (stdio transport, for Claude Desktop/Claude Code)
hippocampus mcp --owner <owner-id>

# Run MCP server (SSE transport, for remote clients)
hippocampus mcp --transport sse --owner <owner-id>

# Run REST API server (port 8420, Swagger UI at /docs)
hippocampus api
```

No test framework, linter, or formatter is configured yet. The `tests/` directory exists but is empty.

## Prerequisites

- Python 3.11+
- PostgreSQL 15+ with pgvector extension
- Ollama running locally with `nomic-embed-text-v2-moe` model (`ollama pull nomic-embed-text-v2-moe`)

## Architecture

```
MCP Client (stdio/SSE)  ──►  MCP Server (mcp_server/server.py)
REST Client (HTTP)      ──►  FastAPI    (api/server.py)
                                    │
                              MemoryManager (memory/__init__.py)
                              ├── EpisodicMemory  (memory/episodic.py)
                              ├── SemanticMemory  (memory/semantic.py)
                              └── ReflectionMemory(memory/reflection.py)
                                    │
                         ┌──────────┴──────────┐
                    PostgreSQL+pgvector     Ollama embeddings
```

### Key Modules (`src/hippocampus/`)

- **`__main__.py`** — CLI entry point with `mcp`, `api`, `init-db` subcommands
- **`config.py`** — Pydantic Settings loading from `HIPPOCAMPUS_*` env vars. Module-level `settings` singleton.
- **`db/pool.py`** — asyncpg connection pool with pgvector and JSONB codec registration
- **`db/schema.py`** — DDL for all tables and HNSW vector indexes (dynamic embedding dimension)
- **`db/models.py`** — Dataclasses (`Episode`, `Entity`, `Relation`, `Reflection`, `RevisionProposal`) with `from_row()`/`to_dict()` helpers
- **`embeddings/base.py`** — Abstract `EmbeddingProvider` with `TaskType` enum for document/query/clustering prefixes
- **`embeddings/ollama.py`** — Ollama implementation using httpx
- **`memory/__init__.py`** — `MemoryManager` facade scoped to an `owner_id` (multi-tenancy boundary)
- **`memory/episodic.py`** — Store/recall with pgvector cosine similarity search
- **`memory/semantic.py`** — Knowledge graph: entities, relations, BFS traversal via recursive SQL
- **`memory/reflection.py`** — Reflections + revision proposal system with approve/reject workflow
- **`mcp_server/server.py`** — FastMCP server exposing 12 tools (single owner per instance)
- **`api/server.py`** — FastAPI REST server with `/api/v1/` routes (multi-owner, owner_id as query param)

### Code Patterns

- **Fully async** — all DB operations use asyncpg with `await`
- **Multi-tenancy** — every table has `owner_id`; every query filters by it. MCP serves one owner per instance, REST accepts owner_id per request.
- **Embedding abstraction** — decoupled via `EmbeddingProvider` base class. TaskType prefixes (e.g. `search_document:`, `search_query:`) are required by nomic models.
- **Vector search** — pgvector HNSW indexes with cosine distance; similarity = `1 - cosine_distance`, rounded to 4 decimal places (`SIMILARITY_PRECISION` in config.py)
- **Upsert patterns** — `ON CONFLICT` clauses for entities (unique on owner_id + name + entity_type)
- **JSONB metadata** — flexible metadata columns parsed via codec in pool.py
- **Revision proposals** — human oversight system with status tracking (pending/approved/rejected) and typed actions (delete, update, merge)

## Configuration

All settings via `HIPPOCAMPUS_*` environment variables. See `.env.example` for defaults. Key settings:

| Variable | Default |
|----------|---------|
| `HIPPOCAMPUS_DB_URL` | `postgresql://localhost:5432/hippocampus` |
| `HIPPOCAMPUS_OLLAMA_URL` | `http://localhost:11434` |
| `HIPPOCAMPUS_EMBEDDING_MODEL` | `nomic-embed-text-v2-moe` |
| `HIPPOCAMPUS_EMBEDDING_DIMENSIONS` | `768` |
| `HIPPOCAMPUS_API_HOST` | `127.0.0.1` |
| `HIPPOCAMPUS_API_PORT` | `8420` |
| `HIPPOCAMPUS_MCP_SSE_HOST` | `127.0.0.1` |
| `HIPPOCAMPUS_MCP_SSE_PORT` | `8421` |
| `HIPPOCAMPUS_DEFAULT_OWNER` | `default` |
