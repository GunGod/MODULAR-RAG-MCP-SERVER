"""
Recursive Splitter provider implementation.

This module implements the BaseSplitter interface using a recursive
splitting strategy. It splits text by trying different separators in order,
preserving document structure as much as possible.

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import List, Optional

from src.libs.splitter.base_splitter import BaseSplitter, TextChunk
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
        >>> print(chunks[0].text)  # Text content
        >>> print(chunks[0].start_offset)  # Start position
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
    ) -> List[TextChunk]:
        """
        Split text into chunks using recursive strategy.

        This method recursively splits the text by trying different separators.
        It preserves document structure by prioritizing larger separators.

        Args:
            text: Input text to split
            **kwargs: Additional parameters (currently unused)

        Returns:
            List of TextChunk objects with text and position information

        Raises:
            ValueError: If text is empty
            RuntimeError: If the splitting operation fails

        Example:
            >>> splitter = RecursiveSplitter(chunk_size=100, chunk_overlap=20)
            >>> text = "Hello world.\\n\\nThis is a test document."
            >>> chunks = splitter.split_text(text)
            >>> print(len(chunks))  # Number of chunks
            >>> print(chunks[0].start_offset)  # Start position in original text
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
            return [TextChunk(text=text, start_offset=0, end_offset=len(text))]

        # Split text recursively with position tracking
        try:
            chunks = self._split_recursive(text, 0, 0)
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
        separator_index: int,
        base_offset: int
    ) -> List[TextChunk]:
        """
        Recursively split text using separators with position tracking.

        This method tries to split the text using the current separator.
        If any resulting chunk is still too large, it recursively tries the next separator.

        Args:
            text: Text to split
            separator_index: Index in self.separators list to use
            base_offset: Starting position of text in the original document

        Returns:
            List of TextChunk objects

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
            return self._split_by_size(text, base_offset)

        # Get current separator
        separator = self.separators[separator_index]

        # Empty separator means character-level splitting
        if separator == "":
            logger.debug(
                f"[{self.provider_name}] Empty separator encountered, "
                f"splitting by character size."
            )
            return self._split_by_size(text, base_offset)

        # Split text by current separator
        splits = text.split(separator)

        # Process each split
        chunks = []
        current_chunk_text = ""
        current_start = base_offset

        for i, split in enumerate(splits):
            # Add separator back (except for last element)
            if i < len(splits) - 1:
                split_with_sep = split + separator
            else:
                split_with_sep = split

            # Check if adding this would exceed chunk_size
            if len(current_chunk_text) + len(split_with_sep) <= self.chunk_size:
                # Add to current chunk
                current_chunk_text += split_with_sep
            else:
                # Current chunk is full, save it and start new chunk
                if current_chunk_text:
                    chunk = TextChunk(
                        text=current_chunk_text,
                        start_offset=current_start,
                        end_offset=current_start + len(current_chunk_text)
                    )
                    chunks.append(chunk)
                    current_start += len(current_chunk_text)

                # Start new chunk
                current_chunk_text = split_with_sep

                # Check if new chunk already exceeds chunk_size
                if len(current_chunk_text) > self.chunk_size:
                    # New chunk is too large, decide how to handle it
                    if self._should_split_further(current_chunk_text, separator_index):
                        # Intelligent splitting: try next separator
                        sub_chunks = self._split_recursive(
                            current_chunk_text,
                            separator_index + 1,
                            current_start
                        )

                        # Add all sub-chunks except the last one
                        if len(sub_chunks) > 1:
                            chunks.extend(sub_chunks[:-1])

                        # Last sub-chunk becomes current chunk for next iteration
                        last_sub = sub_chunks[-1]
                        current_chunk_text = last_sub.text
                        current_start = last_sub.start_offset
                    else:
                        # Splitting wouldn't help, force split by size
                        size_chunks = self._split_by_size(
                            current_chunk_text,
                            current_start
                        )

                        # Add all but the last size-based chunk
                        if len(size_chunks) > 1:
                            chunks.extend(size_chunks[:-1])
                            last_size = size_chunks[-1]
                            current_chunk_text = last_size.text
                            current_start = last_size.start_offset
                        else:
                            # Even size-based splitting produced only one chunk
                            # Accept it as-is (it will be slightly larger than chunk_size)
                            chunk = TextChunk(
                                text=current_chunk_text,
                                start_offset=current_start,
                                end_offset=current_start + len(current_chunk_text)
                            )
                            chunks.append(chunk)
                            current_chunk_text = ""
                            current_start += len(current_chunk_text)

        # Add the last chunk
        if current_chunk_text:
            chunk = TextChunk(
                text=current_chunk_text,
                start_offset=current_start,
                end_offset=current_start + len(current_chunk_text)
            )
            chunks.append(chunk)

        # Post-process: handle overlap between chunks
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _split_by_size(self, text: str, base_offset: int) -> List[TextChunk]:
        """
        Split text by character size when all separators are exhausted.

        This is the fallback method when text cannot be split by any separator
        without producing chunks larger than chunk_size.

        Args:
            text: Text to split
            base_offset: Starting position in the original document

        Returns:
            List of TextChunk objects
        """
        chunks = []
        start = 0
        effective_chunk_size = self.chunk_size

        while start < len(text):
            # Calculate end position
            end = start + effective_chunk_size
            end = min(end, len(text))

            # Extract chunk
            chunk_text = text[start:end]
            chunk = TextChunk(
                text=chunk_text,
                start_offset=base_offset + start,
                end_offset=base_offset + end
            )
            chunks.append(chunk)

            # Move to next chunk
            start = end

        logger.debug(
            f"[{self.provider_name}] Split by size into {len(chunks)} chunks"
        )

        return chunks

    def _apply_overlap(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Apply overlap between consecutive chunks.

        This method ensures that consecutive chunks have the specified overlap
        for better context preservation in retrieval.

        Args:
            chunks: List of TextChunk objects

        Returns:
            List of TextChunk objects with overlap applied
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
                overlap_text = prev_chunk.text[-self.chunk_overlap:]

                # Combine overlap with current chunk
                overlapped_text = overlap_text + chunk.text

                # Adjust start offset (overlap comes from previous chunk)
                overlap_start_offset = prev_chunk.end_offset - self.chunk_overlap

                overlapped_chunk = TextChunk(
                    text=overlapped_text,
                    start_offset=overlap_start_offset,
                    end_offset=chunk.end_offset
                )
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
