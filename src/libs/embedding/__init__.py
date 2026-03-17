"""
Embedding provider implementations for Modular RAG MCP Server.

This package provides a pluggable embedding layer with support for multiple providers:
- OpenAI (text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002)
- Azure OpenAI (same models hosted on Azure)
- Ollama (local models like nomic-embed-text, mxbai-embed-large)
- Fake (for testing)

Author: Modular RAG MCP Server Project
License: MIT
"""

from src.libs.embedding.base_embedding import BaseEmbedding, EmbeddingResult
from src.libs.embedding.fake_embedding import FakeEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.openai_embedding import OpenAIEmbedding
from src.libs.embedding.azure_embedding import AzureEmbedding
from src.libs.embedding.ollama_embedding import OllamaEmbedding

__all__ = [
    # Base interfaces
    "BaseEmbedding",
    "EmbeddingResult",
    # Factory
    "EmbeddingFactory",
    # Providers
    "FakeEmbedding",
    "OpenAIEmbedding",
    "AzureEmbedding",
    "OllamaEmbedding",
]
