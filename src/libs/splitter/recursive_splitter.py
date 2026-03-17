"""
Recursive Splitter provider implementation.

This module implements the BaseSplitter interface using a recursive
splitting strategy. It splits text by trying different separators in order,
preserving document structure as much as possible.

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import List, Optional

from src.libs.splitter.base_splitter import BaseSplitter
from src.observability.logger import get_logger

logger = get_logger(__name__)


class RecursiveSplitter(BaseSplitter):
    """
    Recursive Splitter provider implementation.

    This class implements a recursive splitting strategy that divides text into chunks
    by trying different separators in a hierarchical manner. It preserves document
    structure by prioritizing larger separators (paragraphs, sentences) over smaller ones.

    Splitting hierarchy (from largest to smallest):
    1. Double newlines (paragraphs)
    2. Single newline (line breaks)
    3. Spaces (words)
    4. No separator (character level)

    This approach ensures that:
    - Paragraphs stay together when possible
    - Code blocks and structured sections are preserved
    - Chunks respect the chunk_size limit
    - Consecutive chunks have overlap for context preservation

    Example:
        >>> splitter = RecursiveSplitter(chunk_size=1000, chunk_overlap=200)
        >>> text = "Hello world.\\n\\nThis is a test."
        >>> chunks = splitter.split_text(text)
        >>> print(len(chunks))  # Number of chunks
        >>> print(len(chunks[0]))  # Size of first chunk
    """

    # Default separators in order of preference (largest to smallest)
    DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize Recursive Splitter.

        Args:
            chunk_size: Maximum size of each chunk (in characters)
            chunk_overlap: Number of characters to overlap between consecutive chunks
            separators: List of separator strings to use (defaults to ["\n\n", "\n", " ", ""])
            **kwargs: Additional provider-specific parameters
        """
        super().__init__(chunk_size, chunk_overlap, separators, **kwargs)

        # Use provided separators or default
        if separators:
            self.separators = separators
        else:
            self.separators = self.DEFAULT_SEPARATORS.copy()

        logger.debug(
            f"Initialized RecursiveSplitter: chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}, separators={len(self.separators)}"
        )

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "recursive"

    def split_text(
        self,
        text: str,
        **kwargs
    ) -> List[str]:
        """
        Split text into chunks using recursive strategy.

        This method recursively splits the text by trying different separators.
        It preserves document structure by prioritizing larger separators.

        Args:
            text: Input text to split
            **kwargs: Additional parameters (currently unused)

        Returns:
            List of text chunks

        Raises:
            ValueError: If text is empty
            RuntimeError: If the splitting operation fails

        Example:
            >>> splitter = RecursiveSplitter(chunk_size=100, chunk_overlap=20)
            >>> text = "Hello world.\\n\\nThis is a test document."
            >>> chunks = splitter.split_text(text)
            >>> print(len(chunks))  # Number of chunks
        """
        # Validate input
        if not text:
            raise ValueError(
                f"[{self.provider_name}] Cannot split empty text."
            )

        if not isinstance(text, str):
            raise ValueError(
                f"[{self.provider_name}] Text must be a string, got {type(text).__name__}."
            )

        # If text is shorter than chunk_size, return as single chunk
        if len(text) <= self.chunk_size:
            logger.debug(
                f"[{self.provider_name}] Text is shorter than chunk_size, "
                f"returning as single chunk."
            )
            return [text]

        # Split text recursively
        try:
            chunks = self._split_recursive(text, 0)
            logger.debug(
                f"[{self.provider_name}] Split text into {len(chunks)} chunks"
            )
            return chunks

        except Exception as e:
            error_msg = (
                f"[{self.provider_name}] Failed to split text: {str(e)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _split_recursive(
        self,
        text: str,
        separator_index: int
    ) -> List[str]:
        """
        Recursively split text using separators.

        This method tries to split the text using the current separator.
        If any resulting chunk is still too large, it recursively tries the next separator.

        Args:
            text: Text to split
            separator_index: Index in self.separators list to use

        Returns:
            List of text chunks

        Algorithm:
        1. Split text by current separator
        2. For each split, check if chunk size is acceptable
        3. If chunk is too large, recursively try next separator
        4. If we reach the last separator (character level), force split by chunk_size
        """
        # If we've exhausted all separators, split by character size
        if separator_index >= len(self.separators):
            logger.debug(
                f"[{self.provider_name}] Exhausted all separators, "
                f"splitting by character size."
            )
            return self._split_by_size(text)

        # Get current separator
        separator = self.separators[separator_index]

        # Empty separator means character-level splitting
        if separator == "":
            logger.debug(
                f"[{self.provider_name}] Empty separator encountered, "
                f"splitting by character size."
            )
            return self._split_by_size(text)

        # Split text by current separator
        splits = text.split(separator)

        # Process each split
        chunks = []
        current_chunk = ""

        for i, split in enumerate(splits):
            # Add separator back (except for last element)
            if i < len(splits) - 1:
                split_with_sep = split + separator
            else:
                split_with_sep = split

            # Check if adding this would exceed chunk_size
            if len(current_chunk) + len(split_with_sep) <= self.chunk_size:
                # Add to current chunk
                current_chunk += split_with_sep
            else:
                # Current chunk is full, save it and start new chunk
                if current_chunk:
                    chunks.append(current_chunk)

                # Start new chunk
                # Note: Overlap will be applied in post-processing (_apply_overlap)
                current_chunk = split_with_sep

                # Check if new chunk already exceeds chunk_size
                if len(current_chunk) > self.chunk_size:
                    # New chunk is too large, decide how to handle it
                    if self._should_split_further(current_chunk, separator_index):
                        # Intelligent splitting: try next separator
                        sub_chunks = self._split_recursive(current_chunk, separator_index + 1)

                        # Add all sub-chunks except the last one
                        if len(sub_chunks) > 1:
                            chunks.extend(sub_chunks[:-1])

                        # Last sub-chunk becomes current chunk for next iteration
                        current_chunk = sub_chunks[-1]
                    else:
                        # Splitting wouldn't help, force split by size
                        # This is the fallback when we can't split meaningfully
                        size_chunks = self._split_by_size(current_chunk)

                        # Add all but the last size-based chunk
                        if len(size_chunks) > 1:
                            chunks.extend(size_chunks[:-1])
                            current_chunk = size_chunks[-1]
                        else:
                            # Even size-based splitting produced only one chunk
                            # Accept it as-is (it will be slightly larger than chunk_size)
                            chunks.append(current_chunk)
                            current_chunk = ""
                # Note: if chunk fits, current_chunk is already set above (line 215)

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        # Post-process: handle overlap between chunks
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _split_by_size(self, text: str) -> List[str]:
        """
        Split text by character size when all separators are exhausted.

        This is the fallback method when text cannot be split by any separator
        without producing chunks larger than chunk_size.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        effective_chunk_size = self.chunk_size

        while start < len(text):
            # Calculate end position (with overlap consideration)
            if start > 0 and self.chunk_overlap > 0:
                start = start - self.chunk_overlap

            end = start + effective_chunk_size

            # Extract chunk
            chunk = text[start:end]
            chunks.append(chunk)

            # Move to next chunk (with overlap)
            start = end

            # Adjust for overlap in next iteration
            if start >= len(text):
                break

        logger.debug(
            f"[{self.provider_name}] Split by size into {len(chunks)} chunks"
        )

        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Apply overlap between consecutive chunks.

        This method ensures that consecutive chunks have the specified overlap
        for better context preservation in retrieval.

        Args:
            chunks: List of text chunks

        Returns:
            List of text chunks with overlap applied
        """
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk, keep as is
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self.chunk_overlap:]

                # Combine overlap with current chunk
                overlapped_chunk = overlap_text + chunk
                overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks

    def _should_split_further(self, text: str, separator_index: int) -> bool:
        """
        Determine if text should be split further using a smaller separator.

        This method implements a heuristic to decide whether splitting with the
        next separator would be beneficial. It avoids unnecessary splitting that
        would produce too-small or too-many chunks.

        Args:
            text: Text to check
            separator_index: Index of current separator

        Returns:
            True if text should be split further, False otherwise
        """
        # If we're at the last separator, cannot split further
        if separator_index >= len(self.separators) - 1:
            return False

        # If text fits in chunk_size, don't split further
        if len(text) <= self.chunk_size:
            return False

        # Get next separator
        next_separator = self.separators[separator_index + 1]

        # Empty separator means character-level splitting (always valid)
        if next_separator == "":
            return True

        # Try splitting with next separator
        test_splits = text.split(next_separator)

        # Check if split produces reasonably sized chunks
        # A split is useful if it produces chunks that are:
        # - Not too small (at least 20% of chunk_size, min 50 chars)
        # - Not too many (at most 10 chunks) to avoid fragmentation
        min_chunk_size = max(self.chunk_size // 5, 50)

        good_chunks = 0
        for split in test_splits:
            if len(split) >= min_chunk_size:
                good_chunks += 1

        # If we get at least 2 good chunks, it's worth splitting
        return good_chunks >= 2
