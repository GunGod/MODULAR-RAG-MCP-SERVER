"""
Unit tests for DocumentChunker with TextChunk support.

These tests verify the adapter layer behavior that now uses TextChunk
with precise position information from libs.splitter.

Test Coverage:
- Factory creation and initialization
- Document chunking with FakeSplitter
- Chunk ID generation (format and determinism)
- Metadata inheritance from Document
- chunk_index field addition
- source_ref establishment
- Image reference distribution
- Precise offset tracking
- Error handling

Author: Modular RAG MCP Server Project
License: MIT
"""

import pytest
from unittest.mock import Mock

from src.core.types import Chunk, Document, ImageMetadata
from src.ingestion.chunking import DocumentChunker
from src.libs.splitter.fake_splitter import FakeSplitter
from src.libs.splitter.base_splitter import BaseSplitter, TextChunk


class TestDocumentChunkerInitialization:
    """Tests for DocumentChunker initialization and factory creation."""

    def test_init_with_valid_splitter(self):
        """Test initialization with a valid splitter instance."""
        splitter = FakeSplitter(chunk_size=500, chunk_overlap=50)
        chunker = DocumentChunker(splitter)

        assert chunker.splitter == splitter
        assert hasattr(chunker, 'logger')

    def test_init_with_invalid_splitter(self):
        """Test that initialization fails with non-Splitter object."""
        not_a_splitter = {"not": "a splitter"}

        with pytest.raises(TypeError) as exc_info:
            DocumentChunker(not_a_splitter)

        assert "BaseSplitter" in str(exc_info.value)

    def test_init_with_none_splitter(self):
        """Test that initialization fails with None."""
        with pytest.raises(TypeError):
            DocumentChunker(None)


class TestDocumentChunkerChunking:
    """Tests for the core chunk_document() functionality."""

    def test_chunk_simple_document(self):
        """Test chunking a simple document without images."""
        # Setup
        splitter = FakeSplitter(chunk_size=500, chunk_overlap=0)
        chunker = DocumentChunker(splitter)

        document = Document(
            id="test_doc_001",
            text="This is a test document. " * 100,  # ~2400 chars
            metadata={"source_path": "/path/to/doc.txt", "title": "Test Doc"}
        )

        # Execute
        chunks = chunker.chunk_document(document)

        # Verify
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_document_with_images(self):
        """Test chunking a document with image placeholders."""
        # Setup
        splitter = FakeSplitter(chunk_size=300, chunk_overlap=0)
        chunker = DocumentChunker(splitter)

        # Create document with image placeholders
        text = "Section 1\n\n[IMAGE: img_001]\n\nSection 2\n\n[IMAGE: img_002]"
        document = Document(
            id="test_doc_with_images",
            text=text,
            metadata={
                "source_path": "/path/to/doc.pdf",
                "images": [
                    ImageMetadata(
                        id="img_001",
                        path="data/images/test/img_001.png",
                        text_offset=12,
                        text_length=17
                    ),
                    ImageMetadata(
                        id="img_002",
                        path="data/images/test/img_002.png",
                        text_offset=50,
                        text_length=17
                    )
                ]
            }
        )

        # Execute
        chunks = chunker.chunk_document(document)

        # Verify
        assert len(chunks) > 0
        # At least some chunks should have image refs
        chunks_with_images = [c for c in chunks if c.metadata.get("has_images")]
        assert len(chunks_with_images) > 0

    def test_chunk_empty_document(self):
        """Test that chunking fails for empty documents."""
        splitter = FakeSplitter()
        chunker = DocumentChunker(splitter)

        document = Document(
            id="empty_doc",
            text="",
            metadata={"source_path": "/path/to/empty.txt"}
        )

        with pytest.raises(ValueError) as exc_info:
            chunker.chunk_document(document)

        assert "non-empty text" in str(exc_info.value)

    def test_chunk_document_returns_correct_count(self):
        """Test that chunking produces expected number of chunks."""
        # Setup: 2500 chars, chunk_size 1000 -> expect ~3 chunks
        splitter = FakeSplitter(chunk_size=1000, chunk_overlap=0)
        chunker = DocumentChunker(splitter)

        document = Document(
            id="count_test",
            text="A" * 2500,
            metadata={"source_path": "/path/to/test.txt"}
        )

        # Execute
        chunks = chunker.chunk_document(document)

        # Verify: should get 3 chunks (1000 + 1000 + 500)
        assert len(chunks) == 3


class TestChunkIdGeneration:
    """Tests for Chunk ID generation."""

    def test_chunk_id_format(self):
        """Test that chunk IDs follow the correct format."""
        splitter = FakeSplitter(chunk_size=500)
        chunker = DocumentChunker(splitter)

        # Use a document ID without underscores to avoid split confusion
        document = Document(
            id="testdoc123",
            text="Test content for ID generation",
            metadata={"source_path": "/path/to/doc.txt"}
        )

        chunks = chunker.chunk_document(document)

        # Check ID format: {doc_id}_{index:04d}_{hash}
        for i, chunk in enumerate(chunks):
            assert chunk.id.startswith(f"testdoc123_{i:04d}_")
            # Hash should be 8 characters
            parts = chunk.id.split('_')
            assert len(parts) == 3
            assert len(parts[2]) == 8

    def test_chunk_id_determinism(self):
        """Test that chunking the same document twice produces identical IDs."""
        splitter = FakeSplitter(chunk_size=500)
        chunker = DocumentChunker(splitter)

        document = Document(
            id="determinism_test",
            text="Same content should produce same chunk IDs",
            metadata={"source_path": "/path/to/doc.txt"}
        )

        # Chunk twice
        chunks1 = chunker.chunk_document(document)
        chunks2 = chunker.chunk_document(document)

        # IDs should be identical
        ids1 = [c.id for c in chunks1]
        ids2 = [c.id for c in chunks2]

        assert ids1 == ids2


class TestMetadataInheritance:
    """Tests for metadata inheritance from Document to Chunk."""

    def test_metadata_inheritance_basic(self):
        """Test that chunks inherit document metadata."""
        splitter = FakeSplitter(chunk_size=500)
        chunker = DocumentChunker(splitter)

        document = Document(
            id="metadata_test",
            text="Content here",
            metadata={
                "source_path": "/path/to/doc.pdf",
                "title": "Test Title",
                "author": "Test Author",
                "doc_type": "pdf"
            }
        )

        chunks = chunker.chunk_document(document)

        # All chunks should inherit document metadata
        for chunk in chunks:
            assert chunk.metadata["source_path"] == "/path/to/doc.pdf"
            assert chunk.metadata["title"] == "Test Title"
            assert chunk.metadata["author"] == "Test Author"
            assert chunk.metadata["doc_type"] == "pdf"

    def test_chunk_index_field_added(self):
        """Test that chunk_index is added to metadata."""
        splitter = FakeSplitter(chunk_size=500)
        chunker = DocumentChunker(splitter)

        document = Document(
            id="index_test",
            text="Content " * 200,
            metadata={"source_path": "/path/to/doc.txt"}
        )

        chunks = chunker.chunk_document(document)

        # Verify chunk_index values
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i

    def test_chunk_size_field_added(self):
        """Test that chunk_size is added to metadata."""
        splitter = FakeSplitter(chunk_size=500)
        chunker = DocumentChunker(splitter)

        document = Document(
            id="size_test",
            text="Content " * 200,
            metadata={"source_path": "/path/to/doc.txt"}
        )

        chunks = chunker.chunk_document(document)

        # Verify chunk_size equals actual text length
        for chunk in chunks:
            assert chunk.metadata["chunk_size"] == len(chunk.text)

    def test_has_images_field_correctness(self):
        """Test that has_images field reflects image presence."""
        splitter = FakeSplitter(chunk_size=300)
        chunker = DocumentChunker(splitter)

        # Document with images
        text_with_images = "Text [IMAGE: img_001] more text"
        document_with = Document(
            id="has_images_test",
            text=text_with_images,
            metadata={
                "source_path": "/path/to/doc.pdf",
                "images": [
                    ImageMetadata(
                        id="img_001",
                        path="data/images/test/img_001.png",
                        text_offset=5,
                        text_length=17
                    )
                ]
            }
        )

        chunks_with = chunker.chunk_document(document_with)

        # Chunks with image refs should have has_images=True
        chunks_with_img = [c for c in chunks_with if c.metadata.get("has_images")]
        assert len(chunks_with_img) > 0


class TestImageReferenceDistribution:
    """Tests for image reference distribution across chunks."""

    def test_image_refs_extraction(self):
        """Test that image placeholders are correctly extracted."""
        splitter = FakeSplitter(chunk_size=400)
        chunker = DocumentChunker(splitter)

        text = "Before [IMAGE: img_001] middle [IMAGE: img_002] after"
        document = Document(
            id="image_refs_test",
            text=text,
            metadata={
                "source_path": "/path/to/doc.pdf",
                "images": [
                    ImageMetadata(id="img_001", path="/path/img1.png", text_offset=6, text_length=17),
                    ImageMetadata(id="img_002", path="/path/img2.png", text_offset=35, text_length=17),
                ]
            }
        )

        chunks = chunker.chunk_document(document)

        # Check image_refs in chunks
        for chunk in chunks:
            if chunk.metadata.get("has_images"):
                image_refs = chunk.metadata.get("image_refs", [])
                assert isinstance(image_refs, list)
                # Each ref should have 'id' field
                for ref in image_refs:
                    assert "id" in ref
                    assert ref["id"] in ["img_001", "img_002"]

    def test_images_field_filtered_by_refs(self):
        """Test that images metadata is filtered to only referenced images."""
        splitter = FakeSplitter(chunk_size=300)
        chunker = DocumentChunker(splitter)

        # Document with multiple images
        text = "Section with [IMAGE: img_001] only"
        document = Document(
            id="filter_test",
            text=text,
            metadata={
                "source_path": "/path/to/doc.pdf",
                "images": [
                    ImageMetadata(id="img_001", path="/path/img1.png", text_offset=15, text_length=17),
                    ImageMetadata(id="img_002", path="/path/img2.png", text_offset=100, text_length=17),
                    ImageMetadata(id="img_003", path="/path/img3.png", text_offset=200, text_length=17),
                ]
            }
        )

        chunks = chunker.chunk_document(document)

        # Chunks with images should only have referenced images
        for chunk in chunks:
            if chunk.metadata.get("has_images"):
                chunk_images = chunk.metadata.get("images", [])
                image_ids = [img["id"] if isinstance(img, dict) else img.id
                            for img in chunk_images]
                # Should only contain img_001 which is in the text
                assert "img_001" in image_ids
                assert "img_002" not in image_ids
                assert "img_003" not in image_ids

    def test_chunks_without_images_have_no_images_field(self):
        """Test that chunks without image placeholders don't have images field."""
        splitter = FakeSplitter(chunk_size=200)
        chunker = DocumentChunker(splitter)

        document = Document(
            id="no_images_test",
            text="Plain text without any image placeholders",
            metadata={"source_path": "/path/to/doc.txt"}
        )

        chunks = chunker.chunk_document(document)

        # Chunks without images should not have images field
        for chunk in chunks:
            if not chunk.metadata.get("has_images"):
                assert "images" not in chunk.metadata


class TestSourceReference:
    """Tests for source_ref establishment."""

    def test_source_ref_points_to_document_id(self):
        """Test that all chunks have source_ref pointing to parent document."""
        splitter = FakeSplitter(chunk_size=500)
        chunker = DocumentChunker(splitter)

        document = Document(
            id="parent_doc_12345",
            text="Content for testing source references",
            metadata={"source_path": "/path/to/doc.txt"}
        )

        chunks = chunker.chunk_document(document)

        # All chunks should point to parent document
        for chunk in chunks:
            assert chunk.source_ref == "parent_doc_12345"

    def test_source_ref_is_optional(self):
        """Test that source_ref is an optional field."""
        splitter = FakeSplitter(chunk_size=500)
        chunker = DocumentChunker(splitter)

        document = Document(
            id="optional_ref_test",
            text="Test content",
            metadata={"source_path": "/path/to/doc.txt"}
        )

        chunks = chunker.chunk_document(document)

        # source_ref should be set, but Chunk allows it to be optional
        for chunk in chunks:
            assert chunk.source_ref is not None


class TestPreciseOffsets:
    """Tests for precise offset tracking using TextChunk."""

    def test_offsets_are_precise(self):
        """Test that offsets match exact positions in original document."""
        splitter = FakeSplitter(chunk_size=1000, chunk_overlap=0)
        chunker = DocumentChunker(splitter)

        # Create document with known structure (3000 chars)
        document = Document(
            id="offset_test",
            text="A" * 1000 + "B" * 1000 + "C" * 1000,
            metadata={"source_path": "/path/to/test.txt"}
        )

        chunks = chunker.chunk_document(document)

        # Verify we get 3 chunks with precise offsets
        assert len(chunks) == 3
        assert chunks[0].start_offset == 0
        assert chunks[0].end_offset == 1000
        assert chunks[1].start_offset == 1000
        assert chunks[1].end_offset == 2000
        assert chunks[2].start_offset == 2000
        assert chunks[2].end_offset == 3000

    def test_offsets_match_content(self):
        """Test that offsets correctly point to the content in original text."""
        splitter = FakeSplitter(chunk_size=500, chunk_overlap=0)
        chunker = DocumentChunker(splitter)

        document_text = "START" + "MIDDLE" * 100 + "END"
        document = Document(
            id="content_test",
            text=document_text,
            metadata={"source_path": "/path/to/test.txt"}
        )

        chunks = chunker.chunk_document(document)

        # Verify each chunk's text matches the original document at its offsets
        for chunk in chunks:
            expected_text = document_text[chunk.start_offset:chunk.end_offset]
            assert chunk.text == expected_text

    def test_offsets_increasing(self):
        """Test that offsets are monotonically increasing."""
        splitter = FakeSplitter(chunk_size=300, chunk_overlap=0)
        chunker = DocumentChunker(splitter)

        document = Document(
            id="increasing_test",
            text="X" * 1500,
            metadata={"source_path": "/path/to/test.txt"}
        )

        chunks = chunker.chunk_document(document)

        # Verify offsets are strictly increasing
        for i in range(len(chunks) - 1):
            assert chunks[i].end_offset == chunks[i + 1].start_offset
            assert chunks[i].start_offset < chunks[i + 1].start_offset


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_splitting_failure_propagates(self):
        """Test that splitter failures are properly propagated."""
        # Create a mock splitter that raises an error
        mock_splitter = Mock(spec=BaseSplitter)
        mock_splitter.split_text.side_effect = RuntimeError("Splitter failed")
        mock_splitter.provider_name = "mock"

        chunker = DocumentChunker(mock_splitter)

        document = Document(
            id="error_test",
            text="Some text",
            metadata={"source_path": "/path/to/doc.txt"}
        )

        with pytest.raises(RuntimeError) as exc_info:
            chunker.chunk_document(document)

        assert "splitting failed" in str(exc_info.value).lower()

    def test_splitter_returns_empty_list(self):
        """Test handling when splitter returns empty list."""
        mock_splitter = Mock(spec=BaseSplitter)
        mock_splitter.split_text.return_value = []
        mock_splitter.provider_name = "mock"

        chunker = DocumentChunker(mock_splitter)

        document = Document(
            id="empty_chunks_test",
            text="Some text",
            metadata={"source_path": "/path/to/doc.txt"}
        )

        with pytest.raises(RuntimeError) as exc_info:
            chunker.chunk_document(document)

        assert "no chunks" in str(exc_info.value).lower()


class TestTextChunkIntegration:
    """Tests for TextChunk integration with DocumentChunker."""

    def test_textchunk_conversion(self):
        """Test that TextChunk objects are properly converted to Chunk."""
        # Create a mock splitter that returns TextChunk objects
        mock_splitter = Mock(spec=BaseSplitter)
        mock_splitter.provider_name = "mock"

        # Create TextChunk objects with known positions
        text_chunks = [
            TextChunk(text="First chunk", start_offset=0, end_offset=11),
            TextChunk(text="Second chunk", start_offset=11, end_offset=23),
            TextChunk(text="Third chunk", start_offset=23, end_offset=34),
        ]
        mock_splitter.split_text.return_value = text_chunks

        chunker = DocumentChunker(mock_splitter)

        document = Document(
            id="textchunk_test",
            text="First chunkSecond chunkThird chunk",
            metadata={"source_path": "/path/to/test.txt"}
        )

        chunks = chunker.chunk_document(document)

        # Verify TextChunk positions are preserved
        assert len(chunks) == 3
        assert chunks[0].start_offset == 0
        assert chunks[0].end_offset == 11
        assert chunks[1].start_offset == 11
        assert chunks[1].end_offset == 23
        assert chunks[2].start_offset == 23
        assert chunks[2].end_offset == 34
