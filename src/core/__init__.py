"""
Core module for the Modular RAG MCP Server.

This module provides fundamental data structures and configurations
used throughout the entire RAG pipeline.

Author: Modular RAG MCP Server Project
License: MIT
"""

from src.core.settings import Settings, VisionLLMConfig, load_settings, validate_settings
from src.core.types import (
    Document,
    Chunk,
    ChunkRecord,
    ImageMetadata,
)

__all__ = [
    # Settings
    "Settings",
    "VisionLLMConfig",
    "load_settings",
    "validate_settings",
    # Core types
    "Document",
    "Chunk",
    "ChunkRecord",
    "ImageMetadata",
]
