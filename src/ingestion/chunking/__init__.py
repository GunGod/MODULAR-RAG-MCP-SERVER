"""
Ingestion Chunking Module.

This module provides the DocumentChunker adapter layer that bridges
libs.splitter and the Ingestion Pipeline.

Author: Modular RAG MCP Server Project
License: MIT
"""

from src.ingestion.chunking.document_chunker import DocumentChunker

__all__ = ["DocumentChunker"]
