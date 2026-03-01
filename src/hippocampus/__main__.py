"""CLI entry point: hippocampus {mcp,api,init-db}"""
from __future__ import annotations

import argparse
import asyncio
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hippocampus",
        description="Persistent memory system for LLMs",
    )
    sub = parser.add_subparsers(dest="command")

    # hippocampus mcp [--transport stdio|sse] [--owner OWNER]
    mcp_cmd = sub.add_parser("mcp", help="Run the MCP server")
    mcp_cmd.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    mcp_cmd.add_argument(
        "--owner",
        default=None,
        help="Owner/tenant ID for this MCP instance (default: HIPPOCAMPUS_DEFAULT_OWNER or 'default')",
    )

    # hippocampus api
    sub.add_parser("api", help="Run the REST API server")

    # hippocampus init-db
    sub.add_parser("init-db", help="Initialize the database schema")

    args = parser.parse_args()

    if args.command == "mcp":
        from hippocampus.mcp_server.server import run_sse, run_stdio

        if args.transport == "sse":
            run_sse(owner_id=args.owner)
        else:
            run_stdio(owner_id=args.owner)

    elif args.command == "api":
        from hippocampus.api.server import run_api

        run_api()

    elif args.command == "init-db":
        asyncio.run(_init_db())

    else:
        parser.print_help()
        sys.exit(1)


async def _init_db() -> None:
    from hippocampus.config import settings
    from hippocampus.db.pool import create_pool
    from hippocampus.db.schema import ensure_schema

    print(f"Connecting to {settings.db_url} ...")
    pool = await create_pool(settings)
    print(f"Creating schema (vector dimension: {settings.embedding_dimensions}) ...")
    await ensure_schema(pool, settings)
    await pool.close()
    print("Done.")


if __name__ == "__main__":
    main()
