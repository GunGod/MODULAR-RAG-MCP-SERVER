"""
Unit tests for core data types.

These tests verify the Document, Chunk, ChunkRecord, and ImageMetadata classes,
ensuring they correctly serialize, validate, and maintain data integrity.

Author: Modular RAG MCP Server Project
License: MIT
"""

import json
import pytest
from datetime import datetime

from src.core.types import (
    Document,
    Chunk,
    ChunkRecord,
    ImageMetadata,
)


class TestImageMetadata:
    """Test ImageMetadata dataclass."""

    def test_create_image_metadata(self):
        """Test creating ImageMetadata with all fields."""
        image = ImageMetadata(
            id="abc123_page1_0000",
            path="data/images/default/abc123_page1_0000.png",
            page=1,
            text_offset=100,
            text_length=20,
            position={"x": 100, "y": 200, "width": 800, "height": 600}
        )

        assert image.id == "abc123_page1_0000"
        assert image.path == "data/images/default/abc123_page1_0000.png"
        assert image.page == 1
        assert image.text_offset == 100
        assert image.text_length == 20
        assert image.position["x"] == 100

    def test_create_image_metadata_minimal(self):
        """Test creating ImageMetadata with minimal fields."""
        image = ImageMetadata(
            id="img_001",
            path="data/images/default/img_001.png"
        )

        assert image.id == "img_001"
        assert image.path == "data/images/default/img_001.png"
        assert image.page is None
        assert image.text_offset == 0
        assert image.text_length == 0
        assert image.position is None

    def test_empty_id_raises_error(self):
        """Test that empty id raises ValueError."""
        with pytest.raises(ValueError, match="id cannot be empty"):
            ImageMetadata(id="", path="data/images/test.png")

    def test_empty_path_raises_error(self):
        """Test that empty path raises ValueError."""
        with pytest.raises(ValueError, match="path cannot be empty"):
            ImageMetadata(id="img_001", path="")

    def test_negative_text_offset_raises_error(self):
        """Test that negative text_offset raises ValueError."""
        with pytest.raises(ValueError, match="text_offset must be non-negative"):
            ImageMetadata(id="img_001", path="data/images/test.png", text_offset=-1)

    def test_negative_text_length_raises_error(self):
        """Test that negative text_length raises ValueError."""
        with pytest.raises(ValueError, match="text_length must be non-negative"):
            ImageMetadata(id="img_001", path="data/images/test.png", text_length=-1)

    def test_to_dict(self):
        """Test ImageMetadata to_dict method."""
        image = ImageMetadata(
            id="img_001",
            path="data/images/test.png",
            page=1,
            text_offset=50,
            text_length=15
        )

        result = image.to_dict()

        assert result["id"] == "img_001"
        assert result["path"] == "data/images/test.png"
        assert result["page"] == 1
        assert result["text_offset"] == 50
        assert result["text_length"] == 15

    def test_from_dict(self):
        """Test ImageMetadata from_dict class method."""
        data = {
            "id": "img_001",
            "path": "data/images/test.png",
            "page": 1,
            "text_offset": 50,
            "text_length": 15,
            "position": {"x": 100}
        }

        image = ImageMetadata.from_dict(data)

        assert image.id == "img_001"
        assert image.path == "data/images/test.png"
        assert image.page == 1
        assert image.position["x"] == 100


class TestDocument:
    """Test Document dataclass."""

    def test_create_document(self):
        """Test creating Document with all fields."""
        doc = Document(
            id="abc123",
            text="This is a test document.",
            metadata={"source_path": "/path/to/doc.pdf", "title": "Test"}
        )

        assert doc.id == "abc123"
        assert doc.text == "This is a test document."
        assert doc.metadata["source_path"] == "/path/to/doc.pdf"
        assert doc.metadata["title"] == "Test"
        assert isinstance(doc.created_at, datetime)
        assert isinstance(doc.updated_at, datetime)

    def test_empty_id_raises_error(self):
        """Test that empty id raises ValueError."""
        with pytest.raises(ValueError, match="id cannot be empty"):
            Document(
                id="",
                text="test",
                metadata={"source_path": "/path/to/doc.pdf"}
            )

    def test_missing_source_path_raises_error(self):
        """Test that missing source_path in metadata raises ValueError."""
        with pytest.raises(ValueError, match="must include 'source_path'"):
            Document(
                id="abc123",
                text="test",
                metadata={}
            )

    def test_non_dict_metadata_raises_error(self):
        """Test that non-dict metadata raises ValueError."""
        with pytest.raises(ValueError, match="metadata must be a dictionary"):
            Document(
                id="abc123",
                text="test",
                metadata="not a dict"  # type: ignore
            )

    def test_images_as_image_metadata_objects(self):
        """Test document with ImageMetadata objects in metadata."""
        images = [
            ImageMetadata(
                id="img_001",
                path="data/images/test/img_001.png",
                text_offset=10,
                text_length=17
            )
        ]

        doc = Document(
            id="abc123",
            text="Text with [IMAGE: img_001] placeholder.",
            metadata={
                "source_path": "/path/to/doc.pdf",
                "images": images
            }
        )

        assert len(doc.get_images()) == 1
        assert doc.get_images()[0].id == "img_001"

    def test_images_as_dicts(self):
        """Test document with image metadata as dicts."""
        images = [
            {
                "id": "img_001",
                "path": "data/images/test/img_001.png",
                "text_offset": 10,
                "text_length": 17
            }
        ]

        doc = Document(
            id="abc123",
            text="Text with [IMAGE: img_001] placeholder.",
            metadata={
                "source_path": "/path/to/doc.pdf",
                "images": images
            }
        )

        retrieved_images = doc.get_images()
        assert len(retrieved_images) == 1
        assert isinstance(retrieved_images[0], ImageMetadata)
        assert retrieved_images[0].id == "img_001"

    def test_invalid_image_list_raises_error(self):
        """Test that non-list images raises ValueError."""
        with pytest.raises(ValueError, match="must be a list"):
            Document(
                id="abc123",
                text="test",
                metadata={
                    "source_path": "/path/to/doc.pdf",
                    "images": "not a list"  # type: ignore
                }
            )

    def test_invalid_image_dict_missing_fields(self):
        """Test that image dict missing required fields raises ValueError."""
        with pytest.raises(ValueError, match="missing required field"):
            Document(
                id="abc123",
                text="test",
                metadata={
                    "source_path": "/path/to/doc.pdf",
                    "images": [{"id": "img_001"}]  # Missing 'path'
                }
            )

    def test_to_dict(self):
        """Test Document to_dict method."""
        doc = Document(
            id="abc123",
            text="Test content",
            metadata={"source_path": "/path/to/doc.pdf"}
        )

        result = doc.to_dict()

        assert result["id"] == "abc123"
        assert result["text"] == "Test content"
        assert result["metadata"]["source_path"] == "/path/to/doc.pdf"
        assert "created_at" in result
        assert "updated_at" in result

    def test_to_json(self):
        """Test Document to_json method."""
        doc = Document(
            id="abc123",
            text="Test content",
            metadata={"source_path": "/path/to/doc.pdf"}
        )

        json_str = doc.to_json(indent=2)
        data = json.loads(json_str)

        assert data["id"] == "abc123"
        assert data["text"] == "Test content"
        assert data["metadata"]["source_path"] == "/path/to/doc.pdf"

    def test_from_dict(self):
        """Test Document from_dict class method."""
        data = {
            "id": "abc123",
            "text": "Test content",
            "metadata": {"source_path": "/path/to/doc.pdf"},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T01:00:00"
        }

        doc = Document.from_dict(data)

        assert doc.id == "abc123"
        assert doc.text == "Test content"
        assert doc.metadata["source_path"] == "/path/to/doc.pdf"
        assert isinstance(doc.created_at, datetime)
        assert isinstance(doc.updated_at, datetime)

    def test_from_json(self):
        """Test Document from_json class method."""
        json_str = '{"id": "abc123", "text": "Test", "metadata": {"source_path": "/path/to/doc.pdf"}}'

        doc = Document.from_json(json_str)

        assert doc.id == "abc123"
        assert doc.text == "Test"
        assert doc.metadata["source_path"] == "/path/to/doc.pdf"

    def test_generate_id(self):
        """Test Document.generate_id static method."""
        content = "test content"
        doc_id = Document.generate_id(content)

        assert len(doc_id) == 64  # SHA256 hash length
        assert isinstance(doc_id, str)

        # Same content should generate same ID
        doc_id2 = Document.generate_id(content)
        assert doc_id == doc_id2

        # Different content should generate different ID
        doc_id3 = Document.generate_id("different content")
        assert doc_id != doc_id3

    def test_get_images_empty(self):
        """Test get_images returns empty list when no images."""
        doc = Document(
            id="abc123",
            text="Test content",
            metadata={"source_path": "/path/to/doc.pdf"}
        )

        images = doc.get_images()
        assert images == []

    def test_get_images_with_images(self):
        """Test get_images returns ImageMetadata list."""
        images_data = [
            {
                "id": "img_001",
                "path": "data/images/test/img_001.png",
                "text_offset": 10,
                "text_length": 17
            },
            {
                "id": "img_002",
                "path": "data/images/test/img_002.png",
                "text_offset": 50,
                "text_length": 17
            }
        ]

        doc = Document(
            id="abc123",
            text="Test with two images",
            metadata={
                "source_path": "/path/to/doc.pdf",
                "images": images_data
            }
        )

        images = doc.get_images()
        assert len(images) == 2
        assert all(isinstance(img, ImageMetadata) for img in images)
        assert images[0].id == "img_001"
        assert images[1].id == "img_002"


class TestChunk:
    """Test Chunk dataclass."""

    def test_create_chunk(self):
        """Test creating Chunk with all fields."""
        chunk = Chunk(
            id="abc123_0000_a1b2c3",
            text="This is a chunk.",
            metadata={"source_path": "/path/to/doc.pdf", "chunk_index": 0},
            start_offset=0,
            end_offset=17,
            source_ref="abc123"
        )

        assert chunk.id == "abc123_0000_a1b2c3"
        assert chunk.text == "This is a chunk."
        assert chunk.metadata["chunk_index"] == 0
        assert chunk.start_offset == 0
        assert chunk.end_offset == 17
        assert chunk.source_ref == "abc123"

    def test_empty_id_raises_error(self):
        """Test that empty id raises ValueError."""
        with pytest.raises(ValueError, match="id cannot be empty"):
            Chunk(
                id="",
                text="test",
                metadata={},
                start_offset=0,
                end_offset=4
            )

    def test_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="text cannot be empty"):
            Chunk(
                id="abc123_0000_a1b2c3",
                text="",
                metadata={},
                start_offset=0,
                end_offset=0
            )

    def test_negative_start_offset_raises_error(self):
        """Test that negative start_offset raises ValueError."""
        with pytest.raises(ValueError, match="start_offset must be non-negative"):
            Chunk(
                id="abc123_0000_a1b2c3",
                text="test",
                metadata={},
                start_offset=-1,
                end_offset=4
            )

    def test_end_offset_less_than_start_raises_error(self):
        """Test that end_offset < start_offset raises ValueError."""
        with pytest.raises(ValueError, match="end_offset.*must be >= start_offset"):
            Chunk(
                id="abc123_0000_a1b2c3",
                text="test",
                metadata={},
                start_offset=10,
                end_offset=5
            )

    def test_to_dict(self):
        """Test Chunk to_dict method."""
        chunk = Chunk(
            id="abc123_0000_a1b2c3",
            text="Test",
            metadata={},
            start_offset=0,
            end_offset=4
        )

        result = chunk.to_dict()

        assert result["id"] == "abc123_0000_a1b2c3"
        assert result["text"] == "Test"
        assert result["start_offset"] == 0
        assert result["end_offset"] == 4

    def test_to_json(self):
        """Test Chunk to_json method."""
        chunk = Chunk(
            id="abc123_0000_a1b2c3",
            text="Test",
            metadata={},
            start_offset=0,
            end_offset=4
        )

        json_str = chunk.to_json()
        data = json.loads(json_str)

        assert data["id"] == "abc123_0000_a1b2c3"
        assert data["text"] == "Test"

    def test_from_dict(self):
        """Test Chunk from_dict class method."""
        data = {
            "id": "abc123_0000_a1b2c3",
            "text": "Test",
            "metadata": {},
            "start_offset": 0,
            "end_offset": 4,
            "source_ref": "abc123"
        }

        chunk = Chunk.from_dict(data)

        assert chunk.id == "abc123_0000_a1b2c3"
        assert chunk.text == "Test"
        assert chunk.source_ref == "abc123"

    def test_from_json(self):
        """Test Chunk from_json class method."""
        json_str = '{"id": "abc123_0000_a1b2c3", "text": "Test", "metadata": {}, "start_offset": 0, "end_offset": 4}'

        chunk = Chunk.from_json(json_str)

        assert chunk.id == "abc123_0000_a1b2c3"
        assert chunk.text == "Test"

    def test_generate_id(self):
        """Test Chunk.generate_id static method."""
        doc_id = "abc123"
        index = 0
        text = "test content"

        chunk_id = Chunk.generate_id(doc_id, index, text)

        assert chunk_id.startswith(f"{doc_id}_0000_")
        assert len(chunk_id.split("_")[-1]) == 8  # 8-char hash

    def test_generate_id_multiple_chunks(self):
        """Test generate_id for multiple chunks."""
        doc_id = "abc123"

        id1 = Chunk.generate_id(doc_id, 0, "first chunk")
        id2 = Chunk.generate_id(doc_id, 1, "second chunk")
        id3 = Chunk.generate_id(doc_id, 2, "third chunk")

        # Check that IDs have correct format and unique hashes
        assert id1.startswith(f"{doc_id}_0000_")
        assert id2.startswith(f"{doc_id}_0001_")
        assert id3.startswith(f"{doc_id}_0002_")

        # Hashes should be different
        hash1 = id1.split("_")[-1]
        hash2 = id2.split("_")[-1]
        hash3 = id3.split("_")[-1]
        assert hash1 != hash2 != hash3

    def test_get_size(self):
        """Test get_size method."""
        chunk = Chunk(
            id="abc123_0000_a1b2c3",
            text="This is a test chunk.",
            metadata={},
            start_offset=0,
            end_offset=21  # Actual length of "This is a test chunk."
        )

        assert chunk.get_size() == 21


class TestChunkRecord:
    """Test ChunkRecord dataclass."""

    def test_create_chunk_record(self):
        """Test creating ChunkRecord with all fields."""
        dense_vec = [0.1, 0.2, 0.3]
        sparse_vec = {"word": 0.5}

        record = ChunkRecord(
            id="abc123_0000_a1b2c3",
            text="This is a test chunk.",
            metadata={"source_path": "/path/to/doc.pdf"},
            dense_vector=dense_vec,
            sparse_vector=sparse_vec,
            start_offset=0,
            end_offset=22,
            source_ref="abc123"
        )

        assert record.id == "abc123_0000_a1b2c3"
        assert record.text == "This is a test chunk."
        assert record.dense_vector == dense_vec
        assert record.sparse_vector == sparse_vec

    def test_chunk_record_requires_source_path(self):
        """Test that ChunkRecord requires source_path in metadata."""
        with pytest.raises(ValueError, match="must include 'source_path'"):
            ChunkRecord(
                id="abc123_0000_a1b2c3",
                text="test",
                metadata={},
                start_offset=0,
                end_offset=4
            )

    def test_from_chunk(self):
        """Test creating ChunkRecord from Chunk."""
        chunk = Chunk(
            id="abc123_0000_a1b2c3",
            text="Test chunk",
            metadata={"source_path": "/path/to/doc.pdf", "chunk_index": 0},
            start_offset=0,
            end_offset=10,
            source_ref="abc123"
        )

        dense_vec = [0.1, 0.2]
        sparse_vec = {"word": 0.5}

        record = ChunkRecord.from_chunk(
            chunk,
            dense_vector=dense_vec,
            sparse_vector=sparse_vec
        )

        assert record.id == chunk.id
        assert record.text == chunk.text
        assert record.metadata["chunk_index"] == 0
        assert record.dense_vector == dense_vec
        assert record.sparse_vector == sparse_vec

    def test_from_chunk_without_vectors(self):
        """Test creating ChunkRecord from Chunk without vectors."""
        chunk = Chunk(
            id="abc123_0000_a1b2c3",
            text="Test chunk",
            metadata={"source_path": "/path/to/doc.pdf"},
            start_offset=0,
            end_offset=10,
            source_ref="abc123"
        )

        record = ChunkRecord.from_chunk(chunk)

        assert record.id == chunk.id
        assert record.dense_vector is None
        assert record.sparse_vector is None

    def test_to_chunk(self):
        """Test converting ChunkRecord to Chunk."""
        record = ChunkRecord(
            id="abc123_0000_a1b2c3",
            text="Test",
            metadata={"source_path": "/path/to/doc.pdf"},
            dense_vector=[0.1, 0.2],
            start_offset=0,
            end_offset=4,
            source_ref="abc123"
        )

        chunk = record.to_chunk()

        assert chunk.id == record.id
        assert chunk.text == record.text
        assert chunk.metadata["source_path"] == "/path/to/doc.pdf"
        # Chunk doesn't have vectors
        assert not hasattr(chunk, "dense_vector")

    def test_has_dense_vector(self):
        """Test has_dense_vector method."""
        record_with = ChunkRecord(
            id="abc123_0000_a1b2c3",
            text="Test",
            metadata={"source_path": "/path/to/doc.pdf"},
            dense_vector=[0.1, 0.2]
        )

        record_without = ChunkRecord(
            id="abc123_0000_a1b2c3",
            text="Test",
            metadata={"source_path": "/path/to/doc.pdf"}
        )

        assert record_with.has_dense_vector() is True
        assert record_without.has_dense_vector() is False

    def test_has_sparse_vector(self):
        """Test has_sparse_vector method."""
        record_with = ChunkRecord(
            id="abc123_0000_a1b2c3",
            text="Test",
            metadata={"source_path": "/path/to/doc.pdf"},
            sparse_vector={"word": 0.5}
        )

        record_without = ChunkRecord(
            id="abc123_0000_a1b2c3",
            text="Test",
            metadata={"source_path": "/path/to/doc.pdf"}
        )

        assert record_with.has_sparse_vector() is True
        assert record_without.has_sparse_vector() is False

    def test_to_dict(self):
        """Test ChunkRecord to_dict method."""
        record = ChunkRecord(
            id="abc123_0000_a1b2c3",
            text="Test",
            metadata={"source_path": "/path/to/doc.pdf"},
            dense_vector=[0.1, 0.2],
            start_offset=0,
            end_offset=4
        )

        result = record.to_dict()

        assert result["id"] == "abc123_0000_a1b2c3"
        assert result["text"] == "Test"
        assert result["dense_vector"] == [0.1, 0.2]

    def test_to_json(self):
        """Test ChunkRecord to_json method."""
        record = ChunkRecord(
            id="abc123_0000_a1b2c3",
            text="Test",
            metadata={"source_path": "/path/to/doc.pdf"},
            start_offset=0,
            end_offset=4
        )

        json_str = record.to_json()
        data = json.loads(json_str)

        assert data["id"] == "abc123_0000_a1b2c3"
        assert data["text"] == "Test"

    def test_from_dict(self):
        """Test ChunkRecord from_dict class method."""
        data = {
            "id": "abc123_0000_a1b2c3",
            "text": "Test",
            "metadata": {"source_path": "/path/to/doc.pdf"},
            "dense_vector": [0.1, 0.2],
            "start_offset": 0,
            "end_offset": 4,
            "source_ref": "abc123"
        }

        record = ChunkRecord.from_dict(data)

        assert record.id == "abc123_0000_a1b2c3"
        assert record.dense_vector == [0.1, 0.2]

    def test_from_json(self):
        """Test ChunkRecord from_json class method."""
        json_str = '{"id": "abc123_0000_a1b2c3", "text": "Test", "metadata": {"source_path": "/path/to/doc.pdf"}, "start_offset": 0, "end_offset": 4}'

        record = ChunkRecord.from_json(json_str)

        assert record.id == "abc123_0000_a1b2c3"
        assert record.text == "Test"
