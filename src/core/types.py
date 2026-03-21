"""
Core data types and contracts for the Modular RAG MCP Server.

This module defines the fundamental data structures used throughout the entire
RAG pipeline, from ingestion to retrieval to MCP tools. These types serve as
the central contract that all modules must agree on.

Key types:
- Document: Represents an ingested document with text and metadata
- Chunk: A piece of a document with position information
- ChunkRecord: A chunk with embedding vectors for storage/retrieval
- ImageMetadata: Metadata for images in multimodal documents

Author: Modular RAG MCP Server Project
License: MIT
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ImageMetadata:
    """
    Metadata for an image in a multimodal document.

    This structure provides comprehensive information about images embedded
    in documents, enabling precise localization and reference.

    Attributes:
        id: Globally unique image identifier (format: {doc_hash}_{page}_{seq})
        path: Storage path for the image file (data/images/{collection}/{image_id}.png)
        page: Page number where the image appears (optional, for PDFs etc.)
        text_offset: Starting character position of the image placeholder in Document.text
        text_length: Length of the image placeholder in characters
        position: Physical position information (PDF coordinates, pixel position, size, etc.)

    Example:
        >>> image_meta = ImageMetadata(
        ...     id="abc123_page1_0000",
        ...     path="data/images/default/abc123_page1_0000.png",
        ...     page=1,
        ...     text_offset=150,
        ...     text_length=20,
        ...     position={"x": 100, "y": 200, "width": 800, "height": 600}
        ... )
    """

    id: str
    path: str
    page: Optional[int] = None
    text_offset: int = 0
    text_length: int = 0
    position: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate image metadata."""
        if not self.id:
            raise ValueError("ImageMetadata.id cannot be empty")
        if not self.path:
            raise ValueError("ImageMetadata.path cannot be empty")
        if self.text_offset < 0:
            raise ValueError("ImageMetadata.text_offset must be non-negative")
        if self.text_length < 0:
            raise ValueError("ImageMetadata.text_length must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the image metadata
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageMetadata":
        """
        Create ImageMetadata from dictionary.

        Args:
            data: Dictionary containing image metadata

        Returns:
            ImageMetadata instance
        """
        return cls(**data)


@dataclass
class Document:
    """
    Represents an ingested document with text and metadata.

    This is the core data structure that flows through the ingestion pipeline.
    It contains the raw document text and structured metadata about the source.

    Attributes:
        id: Unique document identifier (typically SHA256 hash of content)
        text: Full text content of the document (may include image placeholders)
        metadata: Document metadata (must include source_path, may include images)
        created_at: Timestamp when the document was created/ingested
        updated_at: Timestamp when the document was last updated

    Metadata requirements:
        - source_path: Original file path or URL (required)
        - images: List of ImageMetadata for multimodal documents (optional)
        - title: Document title (optional)
        - author: Document author (optional)
        - created_date: Original document creation date (optional)

    Image placeholders in text:
        Images are represented as [IMAGE: {image_id}] in the text.
        The text_offset and text_length in ImageMetadata point to these placeholders.

    Example:
        >>> doc = Document(
        ...     id="abc123",
        ...     text="This is a document with [IMAGE: img_001] an image.",
        ...     metadata={
        ...         "source_path": "/path/to/document.pdf",
        ...         "title": "Sample Document",
        ...         "images": [
        ...             ImageMetadata(
        ...                 id="img_001",
        ...                 path="data/images/default/img_001.png",
        ...                 text_offset=24,
        ...                 text_length=17
        ...             )
        ...         ]
        ...     }
        ... )
    """

    id: str
    text: str
    metadata: Dict[str, Any]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Validate document fields."""
        if not self.id:
            raise ValueError("Document.id cannot be empty")

        # Ensure metadata is a dict
        if not isinstance(self.metadata, dict):
            raise ValueError("Document.metadata must be a dictionary")

        # Validate required metadata fields
        if "source_path" not in self.metadata:
            raise ValueError("Document.metadata must include 'source_path'")

        # Validate images field if present
        if "images" in self.metadata:
            images = self.metadata["images"]
            if not isinstance(images, list):
                raise ValueError("Document.metadata['images'] must be a list")

            # Validate each image metadata
            for i, img_data in enumerate(images):
                if isinstance(img_data, dict):
                    # Ensure all required fields are present
                    required_fields = ["id", "path", "text_offset", "text_length"]
                    for field in required_fields:
                        if field not in img_data:
                            raise ValueError(
                                f"Image at index {i} missing required field: {field}"
                            )
                elif not isinstance(img_data, ImageMetadata):
                    raise ValueError(
                        f"Image at index {i} must be ImageMetadata or dict"
                    )

        # Set timestamps if not provided
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Datetime objects are converted to ISO 8601 strings.

        Returns:
            Dictionary representation of the document
        """
        data = {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata.copy(),
        }

        # Convert datetime to ISO format
        if self.created_at:
            data["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            data["updated_at"] = self.updated_at.isoformat()

        return data

    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Convert to JSON string.

        Args:
            indent: JSON indentation level (None for compact)

        Returns:
            JSON string representation
        """
        # Convert datetime objects to ISO format
        data = self.to_dict()
        return json.dumps(data, indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """
        Create Document from dictionary.

        Args:
            data: Dictionary containing document data

        Returns:
            Document instance
        """
        # Parse datetime fields
        created_at = data.get("created_at")
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        updated_at = data.get("updated_at")
        if updated_at and isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data["metadata"],
            created_at=created_at,
            updated_at=updated_at,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Document":
        """
        Create Document from JSON string.

        Args:
            json_str: JSON string containing document data

        Returns:
            Document instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @staticmethod
    def generate_id(content: str) -> str:
        """
        Generate a document ID from content using SHA256.

        Args:
            content: Content to hash (typically file content or text)

        Returns:
            SHA256 hash as hexadecimal string
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get_images(self) -> List[ImageMetadata]:
        """
        Get list of images from metadata.

        Returns:
            List of ImageMetadata objects (empty list if no images)
        """
        images_data = self.metadata.get("images", [])
        if not images_data:
            return []

        # Convert dict to ImageMetadata if needed
        images = []
        for img_data in images_data:
            if isinstance(img_data, ImageMetadata):
                images.append(img_data)
            elif isinstance(img_data, dict):
                images.append(ImageMetadata.from_dict(img_data))

        return images


@dataclass
class Chunk:
    """
    A piece of a document with position information.

    Chunks are created by splitting documents into smaller, semantically
    meaningful pieces. Each chunk maintains a reference to its source document.

    Attributes:
        id: Unique chunk identifier (format: {doc_id}_{index:04d}_{hash})
        text: Text content of the chunk
        metadata: Chunk metadata (inherits from document, may be extended)
        start_offset: Starting character position in the original Document.text
        end_offset: Ending character position in the original Document.text
        source_ref: Reference to the source document (document ID)

    Metadata inheritance:
        Chunks inherit metadata from their parent Document, but can also
        have chunk-specific metadata such as:
        - chunk_index: Position of this chunk in the document (0-based)
        - chunk_size: Number of characters in this chunk
        - has_images: Whether this chunk contains image references

    Example:
        >>> chunk = Chunk(
        ...     id="abc123_0000_a1b2c3",
        ...     text="This is the first chunk of the document.",
        ...     metadata={"source_path": "/path/to/doc.pdf", "chunk_index": 0},
        ...     start_offset=0,
        ...     end_offset=50,
        ...     source_ref="abc123"
        ... )
    """

    id: str
    text: str
    metadata: Dict[str, Any]
    start_offset: int
    end_offset: int
    source_ref: Optional[str] = None

    def __post_init__(self):
        """Validate chunk fields."""
        if not self.id:
            raise ValueError("Chunk.id cannot be empty")
        if not self.text:
            raise ValueError("Chunk.text cannot be empty")

        # Ensure metadata is a dict
        if not isinstance(self.metadata, dict):
            raise ValueError("Chunk.metadata must be a dictionary")

        # Validate offsets
        if self.start_offset < 0:
            raise ValueError("Chunk.start_offset must be non-negative")
        if self.end_offset < self.start_offset:
            raise ValueError(
                f"Chunk.end_offset ({self.end_offset}) must be >= start_offset ({self.start_offset})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the chunk
        """
        return asdict(self)

    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Convert to JSON string.

        Args:
            indent: JSON indentation level (None for compact)

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """
        Create Chunk from dictionary.

        Args:
            data: Dictionary containing chunk data

        Returns:
            Chunk instance
        """
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "Chunk":
        """
        Create Chunk from JSON string.

        Args:
            json_str: JSON string containing chunk data

        Returns:
            Chunk instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @staticmethod
    def generate_id(doc_id: str, index: int, text: str) -> str:
        """
        Generate a chunk ID.

        Format: {doc_id}_{index:04d}_{content_hash}

        Args:
            doc_id: Source document ID
            index: Chunk index (0-based)
            text: Chunk text content for hashing

        Returns:
            Chunk ID string
        """
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{content_hash}"

    def get_size(self) -> int:
        """
        Get the size of the chunk in characters.

        Returns:
            Number of characters in the chunk
        """
        return len(self.text)


@dataclass
class ChunkRecord:
    """
    A chunk with embedding vectors for storage and retrieval.

    ChunkRecord is the data structure that gets stored in vector databases
    and used for similarity search. It extends Chunk with dense and sparse vectors.

    Attributes:
        id: Unique chunk identifier (same as Chunk.id)
        text: Text content of the chunk
        metadata: Chunk metadata (must include source_path for citation)
        dense_vector: Optional dense embedding vector (for semantic search)
        sparse_vector: Optional sparse embedding vector (for BM25/keyword search)
        start_offset: Starting character position in the original Document.text
        end_offset: Ending character position in the original Document.text
        source_ref: Reference to the source document (document ID)

    Usage:
        ChunkRecord is created during the ingestion pipeline after embedding
        computation and stored in ChromaDB (dense) and BM25 index (sparse).

    Example:
        >>> record = ChunkRecord(
        ...     id="abc123_0000_a1b2c3",
        ...     text="This is a searchable chunk.",
        ...     metadata={"source_path": "/path/to/doc.pdf", "chunk_index": 0},
        ...     dense_vector=[0.1, 0.2, 0.3, ...],  # 1536-dim vector
        ...     sparse_vector={"word1": 0.5, "word2": 0.3},  # BM25 weights
        ...     start_offset=0,
        ...     end_offset=30,
        ...     source_ref="abc123"
        ... )
    """

    id: str
    text: str
    metadata: Dict[str, Any]
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict[str, float]] = None
    start_offset: int = 0
    end_offset: int = 0
    source_ref: Optional[str] = None

    def __post_init__(self):
        """Validate chunk record fields."""
        if not self.id:
            raise ValueError("ChunkRecord.id cannot be empty")
        if not self.text:
            raise ValueError("ChunkRecord.text cannot be empty")

        # Ensure metadata is a dict
        if not isinstance(self.metadata, dict):
            raise ValueError("ChunkRecord.metadata must be a dictionary")

        # Validate required metadata fields
        if "source_path" not in self.metadata:
            raise ValueError("ChunkRecord.metadata must include 'source_path' for citation")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the chunk record
        """
        return asdict(self)

    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Convert to JSON string.

        Args:
            indent: JSON indentation level (None for compact)

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkRecord":
        """
        Create ChunkRecord from dictionary.

        Args:
            data: Dictionary containing chunk record data

        Returns:
            ChunkRecord instance
        """
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "ChunkRecord":
        """
        Create ChunkRecord from JSON string.

        Args:
            json_str: JSON string containing chunk record data

        Returns:
            ChunkRecord instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        dense_vector: Optional[List[float]] = None,
        sparse_vector: Optional[Dict[str, float]] = None,
    ) -> "ChunkRecord":
        """
        Create ChunkRecord from Chunk.

        Args:
            chunk: Source Chunk object
            dense_vector: Optional dense embedding vector
            sparse_vector: Optional sparse embedding vector

        Returns:
            ChunkRecord instance
        """
        return cls(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata.copy(),
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )

    def to_chunk(self) -> Chunk:
        """
        Convert to Chunk (dropping vectors).

        Returns:
            Chunk instance
        """
        return Chunk(
            id=self.id,
            text=self.text,
            metadata=self.metadata.copy(),
            start_offset=self.start_offset,
            end_offset=self.end_offset,
            source_ref=self.source_ref,
        )

    def has_dense_vector(self) -> bool:
        """Check if this record has a dense vector."""
        return self.dense_vector is not None and len(self.dense_vector) > 0

    def has_sparse_vector(self) -> bool:
        """Check if this record has a sparse vector."""
        return self.sparse_vector is not None and len(self.sparse_vector) > 0
