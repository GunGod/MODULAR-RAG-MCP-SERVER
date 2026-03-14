"""
Base Splitter abstract interface for the Modular RAG MCP Server.

This module defines the abstract interface that all splitter providers must implement.
Splitters divide documents into smaller chunks for retrieval and processing.

Author: Modular RAG MCP Server Project
License: MIT
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any


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
    ) -> List[str]:
        """
        Split text into smaller chunks.

        This method should split the input text into semantically meaningful chunks
        while respecting the chunk_size and preserving context through overlap.

        Args:
            text: Input text to split
            **kwargs: Additional provider-specific parameters

        Returns:
            List of text chunks

        Raises:
            RuntimeError: If the splitting operation fails
            ValueError: If text is empty or parameters are invalid

        Example:
            >>> splitter = SplitterFactory.create(settings)
            >>> chunks = splitter.split_text("Long document text...")
            >>> print(len(chunks))  # Number of chunks
            >>> print(len(chunks[0]))  # Size of first chunk
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
