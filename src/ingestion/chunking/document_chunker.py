"""
Document Chunker adapter between libs.splitter and Ingestion Pipeline.

This module implements the DocumentChunker class, which serves as an adapter
layer between the pure-text splitting provided by libs.splitter and the
business object transformation required by the Ingestion Pipeline.

Key responsibilities:
- Convert Document objects to List[Chunk] objects
- Generate unique and deterministic Chunk IDs
- Inherit metadata from parent Document
- Add chunk-specific fields (chunk_index)
- Establish source_ref for traceability
- Distribute image references to appropriate chunks

Author: Modular RAG MCP Server Project
License: MIT
"""

import re
from typing import List

from src.core.types import Chunk, Document
from src.libs.splitter.base_splitter import BaseSplitter, TextChunk
from src.observability.logger import get_logger

logger = get_logger(__name__)

# Image placeholder pattern: [IMAGE: {image_id}]
IMAGE_PLACEHOLDER_PATTERN = r"\[IMAGE:\s*([^\]]+)\]"


class DocumentChunker:
    """
    Adapter layer for document chunking.

    This class bridges the gap between libs.splitter (which now returns TextChunk
    with precise position information) and the Ingestion Pipeline (which needs
    Document → List[Chunk] transformation with metadata inheritance).

    Responsibilities beyond libs.splitter:
    1. Chunk ID generation: {doc_id}_{index:04d}_{hash_8chars}
    2. Metadata inheritance from Document
    3. Add chunk_index field
    4. Establish source_ref to parent Document.id
    5. Image reference distribution (scan [IMAGE: {id}] placeholders)
    6. Type conversion: List[TextChunk] → List[Chunk]

    Example:
        >>> from src.libs.splitter import SplitterFactory
        >>> from src.core.settings import load_settings
        >>>
        >>> settings = load_settings()
        >>> splitter = SplitterFactory.create(settings)
        >>> chunker = DocumentChunker(splitter)
        >>>
        >>> document = loader.load("/path/to/document.pdf")
        >>> chunks = chunker.chunk_document(document)
        >>> print(f"Created {len(chunks)} chunks")
    """

    def __init__(self, splitter: BaseSplitter):
        """
        Initialize the DocumentChunker.

        Args:
            splitter: Configured splitter instance from SplitterFactory
        """
        if not isinstance(splitter, BaseSplitter):
            raise TypeError(
                f"splitter must be an instance of BaseSplitter, got {type(splitter)}"
            )

        self.splitter = splitter
        self.logger = logger

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split a Document into a list of Chunks.

        This method performs the full adapter transformation:
        1. Calls splitter.split_text() to get TextChunk objects with positions
        2. Generates Chunk IDs for each chunk
        3. Inherits metadata from Document
        4. Adds chunk_index and source_ref
        5. Distributes image references

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects

        Raises:
            ValueError: If document is invalid or splitting fails
            RuntimeError: If splitting operation fails

        Example:
            >>> chunks = chunker.chunk_document(document)
            >>> for i, chunk in enumerate(chunks):
            ...     print(f"Chunk {i}: {chunk.id[:50]}...")
        """
        if not document or not document.text:
            raise ValueError("Document must have non-empty text")

        self.logger.info(
            f"Chunking document {document.id[:8]}... "
            f"({len(document.text)} chars)"
        )

        # Step 1: Split text into TextChunks with precise positions
        try:
            text_chunks = self.splitter.split_text(document.text)
        except Exception as e:
            self.logger.error(f"Splitting failed for document {document.id}: {e}")
            raise RuntimeError(f"Document splitting failed: {e}") from e

        if not text_chunks:
            raise RuntimeError(
                f"Splitter produced no chunks for document {document.id}"
            )

        self.logger.info(f"Split into {len(text_chunks)} text chunks")

        # Step 2-6: Convert TextChunks to Chunk objects
        chunks = []
        for chunk_index, text_chunk in enumerate(text_chunks):
            # Generate chunk ID
            chunk_id = self._generate_chunk_id(
                document.id, chunk_index, text_chunk.text
            )

            # Use precise positions from TextChunk
            start_offset = text_chunk.start_offset
            end_offset = text_chunk.end_offset

            # Inherit metadata and add chunk-specific fields
            chunk_metadata = self._inherit_metadata(
                document, chunk_index, text_chunk.text
            )

            # Create Chunk object
            chunk = Chunk(
                id=chunk_id,
                text=text_chunk.text,
                metadata=chunk_metadata,
                start_offset=start_offset,
                end_offset=end_offset,
                source_ref=document.id
            )

            chunks.append(chunk)

        self.logger.info(
            f"Created {len(chunks)} chunks from document {document.id[:8]}..."
        )

        return chunks

    def _generate_chunk_id(
        self,
        doc_id: str,
        chunk_index: int,
        chunk_text: str
    ) -> str:
        """
        Generate a unique and deterministic chunk ID.

        Format: {doc_id}_{index:04d}_{content_hash_8chars}

        The ID is deterministic: the same document text and chunk index
        will always produce the same chunk ID.

        Args:
            doc_id: Source document ID
            chunk_index: Chunk index (0-based)
            chunk_text: Text content of the chunk

        Returns:
            Chunk ID string
        """
        # Use Chunk.generate_id static method (it already implements the spec)
        return Chunk.generate_id(doc_id, chunk_index, chunk_text)

    def _inherit_metadata(
        self,
        document: Document,
        chunk_index: int,
        chunk_text: str
    ) -> dict:
        """
        Inherit metadata from Document and add chunk-specific fields.

        This method:
        1. Copies all Document.metadata fields
        2. Adds chunk_index field
        3. Scans for image placeholders and creates image_refs

        Args:
            document: Source document
            chunk_index: Position of this chunk in the document
            chunk_text: Text content of the chunk

        Returns:
            Metadata dictionary for the chunk
        """
        # Start with a copy of document metadata
        chunk_metadata = document.metadata.copy()

        # Add chunk-specific fields
        chunk_metadata["chunk_index"] = chunk_index
        chunk_metadata["chunk_size"] = len(chunk_text)

        # Scan for image placeholders
        image_refs = self._extract_image_refs(chunk_text)
        if image_refs:
            chunk_metadata["image_refs"] = image_refs
            chunk_metadata["has_images"] = True

            # Filter document images to only those referenced in this chunk
            if "images" in document.metadata:
                doc_images = document.metadata["images"]
                # Create a set of image IDs for fast lookup
                ref_ids = set(img["id"] if isinstance(img, dict) else img.id
                             for img in image_refs)

                # Filter images to only those in this chunk
                chunk_images = [
                    img for img in doc_images
                    if (img["id"] if isinstance(img, dict) else img.id) in ref_ids
                ]
                chunk_metadata["images"] = chunk_images
        else:
            chunk_metadata["has_images"] = False
            # Remove images field if no image references
            chunk_metadata.pop("images", None)

        return chunk_metadata

    def _extract_image_refs(self, chunk_text: str) -> List[dict]:
        """
        Extract image references from chunk text.

        Scans for [IMAGE: {image_id}] placeholders and returns
        structured image reference metadata.

        Args:
            chunk_text: Chunk text to scan

        Returns:
            List of image reference dicts with 'id' field
        """
        # Find all image placeholders
        matches = re.findall(IMAGE_PLACEHOLDER_PATTERN, chunk_text)

        if not matches:
            return []

        # Convert to structured image refs
        image_refs = []
        for image_id in matches:
            image_refs.append({"id": image_id})

        return image_refs
