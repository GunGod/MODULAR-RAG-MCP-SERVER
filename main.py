"""
Modular RAG MCP Server - Main Entry Point

This is the main entry point for the Model Context Protocol (MCP) server.
It provides knowledge hub capabilities via JSON-RPC 2.0 over Stdio transport.

Author: Modular RAG MCP Server Project
License: MIT
"""

import sys
from mcp import Server
from mcp.server.stdio import stdio_server

from src.core.settings import load_settings
from src.observability.logger import get_logger

# Load configuration at startup (fail-fast if config is invalid)
try:
    settings = load_settings()
    logger = get_logger(__name__)
    logger.info("Configuration loaded successfully")
    logger.info(f"LLM provider: {settings.llm.provider}, model: {settings.llm.model}")
    logger.info(f"Embedding provider: {settings.embedding.provider}, model: {settings.embedding.model}")
    logger.info(f"Vector store backend: {settings.vector_store.backend}")
except Exception as e:
    print(f"Failed to load configuration: {e}", file=sys.stderr)
    sys.exit(1)


# Create MCP server instance
server = Server("modular-rag-mcp-server")


@server.list_resources()
async def list_resources() -> list[dict]:
    """List available resources (placeholder for future implementation)."""
    return []


@server.list_tools()
async def list_tools() -> list[dict]:
    """List available tools.

    Currently returns placeholder tools. Full implementation will include:
    - query_knowledge_hub: Main retrieval tool
    - list_collections: List document collections
    - get_document_summary: Get document metadata
    """
    return [
        {
            "name": "query_knowledge_hub",
            "description": "Query the knowledge hub using hybrid search (Dense + Sparse retrieval with RRF fusion)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query text"
                    },
                    "top_k": {
                        "type": "number",
                        "description": "Number of results to return (default: 10)",
                        "default": 10
                    },
                    "collection": {
                        "type": "string",
                        "description": "Collection name (default: 'default')",
                        "default": "default"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "list_collections",
            "description": "List all available document collections",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_document_summary",
            "description": "Get summary and metadata for a specific document",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document identifier"
                    }
                },
                "required": ["doc_id"]
            }
        }
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[dict]:
    """Handle tool calls (placeholder implementation)."""
    if name == "query_knowledge_hub":
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 10)
        collection = arguments.get("collection", "default")

        return [{
            "type": "text",
            "text": f"[PLACEHOLDER] Query '{query}' on collection '{collection}' with top_k={top_k}\n\n"
                  f"Full implementation will be available after completing the Core and Ingestion layers.\n"
                  f"See DEV_SPEC.md task schedule for implementation roadmap."
        }]

    elif name == "list_collections":
        return [{
            "type": "text",
            "text": "[PLACEHOLDER] Available collections:\n- default (placeholder)\n\n"
                  "Full implementation coming soon."
        }]

    elif name == "get_document_summary":
        doc_id = arguments.get("doc_id", "")
        return [{
            "type": "text",
            "text": f"[PLACEHOLDER] Document summary for '{doc_id}'\n\n"
                  "Full implementation coming soon."
        }]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    """Main entry point for the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    try:
        # Run the server
        import asyncio
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)
