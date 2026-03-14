"""
Splitter module for text chunking strategies.

This module provides pluggable splitter implementations for dividing
documents into smaller chunks for retrieval and processing.

Author: Modular RAG MCP Server Project
License: MIT
"""

from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import SplitterFactory

__all__ = ["BaseSplitter", "SplitterFactory"]
