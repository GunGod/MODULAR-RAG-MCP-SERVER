"""
Loader module for document ingestion.

This module provides document loading capabilities including file integrity
checking for incremental ingestion.

Author: Modular RAG MCP Server Project
License: MIT
"""

from src.libs.loader.file_integrity import (
    FileIntegrityChecker,
    IntegrityRecord,
    SQLiteIntegrityChecker,
)

__all__ = [
    "FileIntegrityChecker",
    "IntegrityRecord",
    "SQLiteIntegrityChecker",
]
