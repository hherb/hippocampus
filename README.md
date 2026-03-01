# Hippocampus

A persistent memory retrieval system for LLMs, accessible via MCP or REST API.

Hippocampus provides LLMs with three tiers of long-term memory backed by
PostgreSQL + pgvector:

- **Episodic memory** — timestamped interaction records with vector embeddings
  for semantic retrieval (conversations, observations, events)
- **Semantic memory / knowledge graph** — entities and relationships extracted
  from episodes, stored relationally with vector search
- **Reflection / meta-memory** — periodic summaries, insights, contradiction
  tracking, and human-governed revision proposals

## Architecture

```
┌─────────────┐  ┌─────────────┐
│  MCP Client │  │  REST Client│
│ (Claude etc) │  │ (dashboard) │
└──────┬──────┘  └──────┬──────┘
       │ stdio/SSE      │ HTTP
       ▼                ▼
┌─────────────┐  ┌─────────────┐
│  MCP Server │  │  FastAPI    │
└──────┬──────┘  └──────┬──────┘
       └────────┬───────┘
                ▼
       ┌────────────────┐
       │ Memory Manager │
       ├────────────────┤
       │ Episodic       │
       │ Semantic (KG)  │
       │ Reflection     │
       └───────┬────────┘
          ┌────┴────┐
          ▼         ▼
   ┌──────────┐ ┌────────┐
   │PostgreSQL│ │ Ollama │
   │+ pgvector│ │embeddings│
   └──────────┘ └────────┘
```

## Prerequisites

- Python 3.11+
- PostgreSQL 15+ with [pgvector](https://github.com/pgvector/pgvector) extension
- [Ollama](https://ollama.com) running locally with an embedding model

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Pull the embedding model
ollama pull nomic-embed-text-v2-moe

# 3. Create the database
createdb hippocampus
# Ensure pgvector is installed: CREATE EXTENSION vector;

# 4. Initialize the schema
hippocampus init-db

# 5a. Run the MCP server (stdio — for Claude Desktop / Claude Code)
hippocampus mcp

# 5b. Run the MCP server for a specific owner/agent
hippocampus mcp --owner agent-alice

# 5c. Run the MCP server (SSE — for remote clients)
hippocampus mcp --transport sse

# 5d. Run the REST API (serves all owners; owner_id is a query parameter)
hippocampus api
```

## Configuration

All settings are via environment variables with `HIPPOCAMPUS_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `HIPPOCAMPUS_DB_URL` | `postgresql://localhost:5432/hippocampus` | PostgreSQL connection URL |
| `HIPPOCAMPUS_OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL |
| `HIPPOCAMPUS_EMBEDDING_MODEL` | `nomic-embed-text-v2-moe` | Ollama embedding model |
| `HIPPOCAMPUS_EMBEDDING_DIMENSIONS` | `768` | Embedding vector dimensions |
| `HIPPOCAMPUS_EMBEDDING_PREFIX` | `true` | Use task-type prefixes (nomic models) |
| `HIPPOCAMPUS_API_HOST` | `0.0.0.0` | REST API bind host |
| `HIPPOCAMPUS_API_PORT` | `8420` | REST API port |
| `HIPPOCAMPUS_MCP_SSE_HOST` | `0.0.0.0` | MCP SSE bind host |
| `HIPPOCAMPUS_MCP_SSE_PORT` | `8421` | MCP SSE port |
| `HIPPOCAMPUS_DEFAULT_OWNER` | `default` | Default owner/tenant for MCP server |

## MCP Tools

When connected via MCP, the LLM has access to these tools:

### Episodic Memory
- **`remember`** — store a new memory
- **`recall`** — semantic search over memories
- **`recall_recent`** — retrieve recent memories by time

### Knowledge Graph
- **`learn_entity`** — add/update an entity
- **`learn_relation`** — add a relationship between entities (auto-creates entities)
- **`find_entities`** — search entities semantically or by text
- **`query_relations`** — query relationships
- **`explore_connections`** — BFS graph traversal from an entity

### Reflections
- **`reflect`** — create a summary, insight, or pattern observation
- **`search_reflections`** — semantic search over reflections

### Human Oversight
- **`propose_revision`** — propose a knowledge graph change for human review
- **`pending_revisions`** — list proposals awaiting approval

## Multi-tenancy

Hippocampus supports multiple isolated memory spaces. Each owner (user, agent,
organization) has completely separate episodic memories, knowledge graphs,
reflections, and revision proposals.

- **MCP server**: Each instance serves one owner, set via `--owner` flag or
  `HIPPOCAMPUS_DEFAULT_OWNER` env var. Run multiple instances for multiple
  owners.
- **REST API**: A single instance serves all owners. Every endpoint requires an
  `owner_id` query parameter for tenant isolation.

## REST API

Full REST API available at `http://localhost:8420/docs` (Swagger UI) when
running `hippocampus api`. All endpoints require `?owner_id=<id>` for tenant
scoping.

## Claude Desktop Configuration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "hippocampus": {
      "command": "hippocampus",
      "args": ["mcp", "--owner", "my-agent"]
    }
  }
}
```

## License

GNU Affero General Public License v3 (AGPL-3.0)
