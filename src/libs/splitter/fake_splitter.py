"""
Fake Splitter implementation for testing purposes.

This module provides a mock splitter implementation that returns predictable
chunk boundaries without complex logic. It's used for testing the factory
routing logic and for development when actual splitters are not available.

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import List, Optional

from src.libs.splitter.base_splitter import BaseSplitter, TextChunk


class FakeSplitter(BaseSplitter):
    """
    Fake splitter implementation for testing.

    This splitter does not use any sophisticated splitting logic. Instead,
    it returns simple, predictable chunks based on a fixed-size approach.
    It's useful for:
    - Testing the SplitterFactory routing logic
    - Development without complex dependencies
    - Unit tests that don't require actual NLP processing

    The splitting algorithm:
    - Splits text at regular intervals (chunk_size)
    - No overlap between chunks (simplified for testing)
    - Empty text returns empty list
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the Fake Splitter.

        Args:
            chunk_size: Maximum size of each chunk (default: 1000)
            chunk_overlap: Overlap between chunks (ignored in fake implementation)
            separators: Separators list (ignored in fake implementation)
            **kwargs: Additional ignored parameters
        """
        super().__init__(chunk_size, chunk_overlap, separators, **kwargs)
        self._call_count = 0
        self._total_texts_processed = 0
        self._total_chunks_generated = 0

    def split_text(self, text: str, **kwargs) -> List[TextChunk]:
        """
        Split text into fixed-size chunks with precise positions.

        The splitting is performed at regular intervals of chunk_size characters.
        No semantic analysis or overlap is performed.

        Args:
            text: Input text to split
            **kwargs: Additional ignored parameters

        Returns:
            List of TextChunk objects with text and position information

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("text cannot be empty")

        self._call_count += 1
        self._total_texts_processed += 1

        # Simple fixed-size splitting with precise positions
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk_text = text[start:end]

            # Create TextChunk with precise positions
            chunk = TextChunk(
                text=chunk_text,
                start_offset=start,
                end_offset=end
            )
            chunks.append(chunk)

            start = end

        self._total_chunks_generated += len(chunks)
        return chunks

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "fake"

    def get_call_count(self) -> int:
        """
        Get the number of times split_text() has been called.

        Returns:
            Number of split_text() calls
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

    def get_total_chunks_generated(self) -> int:
        """
        Get total number of chunks generated across all calls.

        Returns:
            Total number of chunks
        """
        return self._total_chunks_generated
