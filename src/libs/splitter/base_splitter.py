"""
Base Splitter abstract interface for the Modular RAG MCP Server.

This module defines the abstract interface that all splitter providers must implement.
Splitters divide documents into smaller chunks for retrieval and processing.

Author: Modular RAG MCP Server Project
License: MIT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Any


@dataclass
class TextChunk:
    """
    A text chunk with precise position information.

    This dataclass wraps a text chunk along with its exact position
    in the original document text. This enables accurate mapping between
    chunks and their source locations.

    Attributes:
        text: The text content of the chunk
        start_offset: Starting character position in the original document (0-based)
        end_offset: Ending character position in the original document (exclusive)

    Example:
        >>> chunk = TextChunk("Hello world", 0, 11)
        >>> print(chunk.text)  # "Hello world"
        >>> print(chunk.start_offset)  # 0
        >>> print(chunk.end_offset)  # 11
    """

    text: str
    start_offset: int
    end_offset: int

    def __post_init__(self):
        """Validate TextChunk fields."""
        if not isinstance(self.text, str):
            raise ValueError("TextChunk.text must be a string")
        if self.start_offset < 0:
            raise ValueError("TextChunk.start_offset must be non-negative")
        if self.end_offset < self.start_offset:
            raise ValueError(
                f"TextChunk.end_offset ({self.end_offset}) must be >= start_offset ({self.start_offset})"
            )

    @property
    def length(self) -> int:
        """Get the length of the chunk text."""
        return len(self.text)

    def to_str(self) -> str:
        """
        Get the text content (for backward compatibility).

        Returns:
            The text content of the chunk
        """
        return self.text


class BaseSplitter(ABC):
    """
    Abstract base class for splitter providers.

    All splitter implementations (Recursive, Semantic, Fixed, etc.) must inherit from
    this class and implement the split_text() method.

    The interface focuses on splitting text into semantically meaningful chunks
    while preserving context and structure.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the splitter provider.

        Args:
            chunk_size: Maximum size of each chunk (in characters)
            chunk_overlap: Overlap between consecutive chunks
            separators: List of separator strings to use for splitting (optional)
            **kwargs: Additional provider-specific parameters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        self._provider_config = kwargs

    @abstractmethod
    def split_text(
        self,
        text: str,
        **kwargs
    ) -> List[TextChunk]:
        """
        Split text into smaller chunks with position information.

        This method should split the input text into semantically meaningful chunks
        while respecting the chunk_size and preserving context through overlap.
        Each chunk includes precise start/end offsets in the original document.

        Args:
            text: Input text to split
            **kwargs: Additional provider-specific parameters

        Returns:
            List of TextChunk objects with text and position information

        Raises:
            RuntimeError: If the splitting operation fails
            ValueError: If text is empty or parameters are invalid

        Example:
            >>> splitter = SplitterFactory.create(settings)
            >>> chunks = splitter.split_text("Long document text...")
            >>> print(len(chunks))  # Number of chunks
            >>> print(chunks[0].text)  # Text content
            >>> print(chunks[0].start_offset)  # Start position in document
            >>> print(chunks[0].end_offset)  # End position in document
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Get the provider name (e.g., "recursive", "semantic", "fixed").

        Returns:
            Provider name as a string
        """
        pass

    def __repr__(self) -> str:
        """String representation of the splitter instance."""
        return f"{self.__class__.__name__}(chunk_size={self.chunk_size}, overlap={self.chunk_overlap}, provider='{self.provider_name}')"
