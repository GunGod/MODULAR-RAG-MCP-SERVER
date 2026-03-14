"""
Base Embedding abstract interface for the Modular RAG MCP Server.

This module defines the abstract interface that all embedding providers must implement.
Embeddings convert text into vector representations for semantic search.

Author: Modular RAG MCP Server Project
License: MIT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Any


@dataclass
class EmbeddingResult:
    """
    Result from an embedding provider.

    Attributes:
        embeddings: List of embedding vectors (each vector is a list of floats)
        model: Model name used for embedding
        provider: Provider name (e.g., "openai", "azure")
        dimension: Dimension of each embedding vector
        usage: Optional token usage information
    """
    embeddings: List[List[float]]
    model: str
    provider: str
    dimension: int
    usage: Optional[dict[str, int]] = None


class BaseEmbedding(ABC):
    """
    Abstract base class for embedding providers.

    All embedding implementations (OpenAI, Azure, Ollama, etc.) must inherit from
    this class and implement the embed() method.

    The interface focuses on batch processing for efficiency, converting multiple
    texts to vectors in a single API call when possible.
    """

    def __init__(
        self,
        model: str,
        dimension: int,
        api_key: Optional[str] = None,
        batch_size: int = 32,
        **kwargs
    ):
        """
        Initialize the embedding provider.

        Args:
            model: Model name (e.g., "text-embedding-3-small")
            dimension: Dimension of the embedding vectors
            api_key: API key for the provider (if required)
            batch_size: Maximum number of texts to process in one batch
            **kwargs: Additional provider-specific parameters
        """
        self.model = model
        self.dimension = dimension
        self.api_key = api_key
        self.batch_size = batch_size
        self._provider_config = kwargs

    @abstractmethod
    def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> EmbeddingResult:
        """
        Convert texts to embedding vectors.

        This method should handle batching internally if the number of texts
        exceeds batch_size. The returned vectors should be in the same order as
        the input texts.

        Args:
            texts: List of text strings to embed
            **kwargs: Additional provider-specific parameters

        Returns:
            EmbeddingResult containing:
                - embeddings: List of embedding vectors (one per input text)
                - model: Model name used
                - provider: Provider name
                - dimension: Vector dimension
                - usage: Token usage if available

        Raises:
            RuntimeError: If the API call fails
            ValueError: If texts list is empty or parameters are invalid

        Example:
            >>> embedding = EmbeddingFactory.create(settings)
            >>> result = embedding.embed(["Hello", "World"])
            >>> print(len(result.embeddings))  # 2
            >>> print(len(result.embeddings[0]))  # dimension (e.g., 1536)
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Get the provider name (e.g., "openai", "azure", "ollama").

        Returns:
            Provider name as a string
        """
        pass

    def __repr__(self) -> str:
        """String representation of the embedding instance."""
        return f"{self.__class__.__name__}(model='{self.model}', dimension={self.dimension}, provider='{self.provider_name}')"
