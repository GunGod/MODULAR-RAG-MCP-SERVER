"""
Fake Embedding implementation for testing purposes.

This module provides a mock embedding implementation that returns stable, predictable
vectors without making any API calls. It's used for testing the factory routing logic
and for development when actual embedding APIs are not available.

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import List

from src.libs.embedding.base_embedding import BaseEmbedding, EmbeddingResult


class FakeEmbedding(BaseEmbedding):
    """
    Fake embedding implementation for testing.

    This embedding does not make any actual API calls. Instead, it returns
    stable, predictable vectors based on the input texts. It's useful for:
    - Testing the EmbeddingFactory routing logic
    - Development without API credentials
    - Unit tests that don't require external dependencies

    Vectors are generated using a simple hash-based approach that ensures:
    - Same text always produces the same vector (stability)
    - Different texts produce different vectors (discriminability)
    - Vectors have the correct dimension
    """

    def __init__(
        self,
        model: str = "fake-embedding",
        dimension: int = 1536,
        api_key: str = None,
        batch_size: int = 32,
        **kwargs
    ):
        """
        Initialize the Fake Embedding.

        Args:
            model: Model name (default: "fake-embedding")
            dimension: Vector dimension (default: 1536, matches OpenAI)
            api_key: Not used, kept for interface compatibility
            batch_size: Not used, kept for interface compatibility
            **kwargs: Additional ignored parameters
        """
        super().__init__(model, dimension, api_key, batch_size, **kwargs)
        self._call_count = 0
        self._total_texts_processed = 0

    def embed(self, texts: List[str], **kwargs) -> EmbeddingResult:
        """
        Generate fake embedding vectors for texts.

        The vectors are generated using a simple deterministic algorithm:
        - Each text produces a vector of length `self.dimension`
        - Same text produces same vector (hash-based)
        - Different texts produce different vectors

        Args:
            texts: List of text strings to embed
            **kwargs: Additional ignored parameters

        Returns:
            EmbeddingResult with fake embeddings
        """
        if not texts:
            raise ValueError("texts list cannot be empty")

        self._call_count += 1
        self._total_texts_processed += len(texts)

        # Generate fake embeddings
        embeddings = []
        for text in texts:
            # Create a stable hash-based vector
            vector = self._generate_fake_vector(text, self.dimension)
            embeddings.append(vector)

        return EmbeddingResult(
            embeddings=embeddings,
            model=self.model,
            provider=self.provider_name,
            dimension=self.dimension,
            usage={
                "total_tokens": sum(len(t.split()) for t in texts),
                "prompt_tokens": sum(len(t.split()) for t in texts),
            },
        )

    def _generate_fake_vector(self, text: str, dimension: int) -> List[float]:
        """
        Generate a stable fake vector for a given text.

        Uses a simple hash-based approach that ensures:
        - Same text -> same vector
        - Different texts -> different vectors
        - Vector values are between -1 and 1 (normalized)

        Args:
            text: Input text
            dimension: Vector dimension

        Returns:
            List of floats representing the fake embedding
        """
        # Create a simple hash of the text
        text_hash = hash(text)

        # Generate vector from hash
        vector = []
        for i in range(dimension):
            # Use a simple formula that creates values between -1 and 1
            value = ((text_hash + i) % 2000) / 1000.0 - 1.0
            vector.append(value)

        return vector

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "fake"

    def get_call_count(self) -> int:
        """
        Get the number of times embed() has been called.

        Returns:
            Number of embed() calls
        """
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the call counter to zero."""
        self._call_count = 0

    def get_total_texts_processed(self) -> int:
        """
        Get total number of texts processed across all calls.

        Returns:
            Total number of texts
        """
        return self._total_texts_processed
