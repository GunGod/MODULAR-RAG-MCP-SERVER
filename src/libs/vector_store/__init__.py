"""
VectorStore module for vector database storage and retrieval.

This module provides pluggable vector store implementations for storing
and querying embeddings with associated metadata.

Author: Modular RAG MCP Server Project
License: MIT
"""

from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult,
    UpsertResult,
)
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

__all__ = [
    "BaseVectorStore",
    "VectorRecord",
    "QueryResult",
    "UpsertResult",
    "VectorStoreFactory",
]
