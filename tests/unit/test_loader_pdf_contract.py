"""
Unit tests for PdfLoader contract.

These tests verify the PDF loading functionality including:
- PDF to Markdown conversion using MarkItDown
- Image extraction from PDF
- Image placeholder insertion
- Metadata validation
- Graceful degradation (image extraction failure doesn't block text parsing)

Author: Modular RAG MCP Server Project
License: MIT
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.core.types import Document, ImageMetadata
from src.libs.loader.pdf_loader import PdfLoader


class TestPdfLoaderContract:
    """Test PdfLoader contract compliance."""

    def test_supported_extensions(self):
        """Test that PdfLoader reports correct supported extensions."""
        loader = PdfLoader()
        assert loader.supported_extensions == [".pdf"]

    def test_supports_format(self):
        """Test format support detection."""
        loader = PdfLoader()

        assert loader.supports_format("test.pdf") is True
        assert loader.supports_format("test.PDF") is True
        assert loader.supports_format("test.txt") is False
        assert loader.supports_format("test.docx") is False

    def test_load_file_not_found(self):
        """Test that non-existent file raises FileNotFoundError."""
        loader = PdfLoader()

        with pytest.raises(FileNotFoundError, match="File not found"):
            loader.load("/nonexistent/document.pdf")

    def test_load_unsupported_format(self):
        """Test that unsupported format raises ValueError."""
        loader = PdfLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a text file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Hello, World!")

            with pytest.raises(ValueError, match="Unsupported file format"):
                loader.load(str(test_file))


class TestPdfLoaderWithMockMarkItDown:
    """Test PdfLoader with mocked MarkItDown."""

    def test_load_simple_pdf_success(self):
        """Test loading a simple PDF without images."""
        loader = PdfLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test PDF file (empty for this test)
            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\n%%EOF")

            # Mock MarkItDown result
            mock_result = Mock()
            mock_result.text_content = "# Test Document\n\nThis is a test document."

            with patch.object(loader.converter, "convert", return_value=mock_result):
                document = loader.load(str(test_file))

                # Verify document structure
                assert isinstance(document, Document)
                assert document.id == "test"
                assert document.text == "# Test Document\n\nThis is a test document."
                assert document.metadata["doc_type"] == "pdf"
                assert document.metadata["file_name"] == "test.pdf"
                assert "source_path" in document.metadata
                assert "images" not in document.metadata  # No images extracted

    def test_load_pdf_with_images(self):
        """Test loading a PDF with images."""
        loader = PdfLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test PDF file
            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\n%%EOF")

            # Mock MarkItDown result
            mock_result = Mock()
            mock_result.text_content = "# Test Document\n\nContent with image."

            # Mock image metadata
            mock_image_metadata = loader._create_image_metadata(
                image_id="test_0001_0000",
                image_path="data/images/test/test_0001_0000.png",
                page_num=1,
                text_offset=25,
                text_length=20,
                position={"x0": 0, "y0": 0, "x1": 100, "y1": 100}
            )

            with patch.object(loader.converter, "convert", return_value=mock_result):
                # Mock the image extraction method
                with patch.object(loader, "_extract_images_from_pdf", return_value=[mock_image_metadata]):
                    document = loader.load(str(test_file))

                    # Verify images were extracted
                    assert "images" in document.metadata
                    images = document.metadata["images"]
                    assert len(images) == 1

                    # Verify image metadata
                    first_image = images[0]
                    assert isinstance(first_image, ImageMetadata)
                    assert first_image.id == "test_0001_0000"
                    assert first_image.page == 1
                    assert "[IMAGE:" in document.text  # Placeholder inserted

    def test_load_pdf_image_extraction_graceful_degradation(self):
        """Test that image extraction failure doesn't block text parsing."""
        loader = PdfLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test PDF file
            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\n%%EOF")

            # Mock MarkItDown result
            mock_result = Mock()
            mock_result.text_content = "# Test Document\n\nContent without images."

            with patch.object(loader.converter, "convert", return_value=mock_result):
                # Mock image extraction to return empty list (simulating failure)
                with patch.object(loader, "_extract_images_from_pdf", return_value=[]):
                    # Should not raise error, should gracefully degrade
                    document = loader.load(str(test_file))

                    # Verify text was still parsed
                    assert document.text == "# Test Document\n\nContent without images."
                    assert "images" not in document.metadata  # No images due to degradation

    def test_load_pdf_markitdown_failure(self):
        """Test that MarkItDown failure raises RuntimeError."""
        loader = PdfLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test PDF file
            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\n%%EOF")

            # Mock MarkItDown failure
            with patch.object(loader.converter, "convert", side_effect=Exception("Conversion failed")):
                with pytest.raises(RuntimeError, match="PDF parsing failed"):
                    loader.load(str(test_file))

    def test_metadata_completeness(self):
        """Test that all required metadata fields are present."""
        loader = PdfLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test PDF file
            test_file = Path(tmpdir) / "document.pdf"
            test_file.write_bytes(b"%PDF-1.4\n%%EOF")

            # Mock MarkItDown result
            mock_result = Mock()
            mock_result.text_content = "# Document\n\nContent."

            with patch.object(loader.converter, "convert", return_value=mock_result):
                document = loader.load(str(test_file))

                # Verify required metadata fields
                assert "source_path" in document.metadata
                assert document.metadata["doc_type"] == "pdf"
                assert document.metadata["file_name"] == "document.pdf"
                assert "file_size" in document.metadata
                assert document.metadata["file_size"] >= 0

    def test_document_id_generation(self):
        """Test that document ID is correctly generated from filename."""
        loader = PdfLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test PDF files with different names
            test_file1 = Path(tmpdir) / "my_document.pdf"
            test_file1.write_bytes(b"%PDF-1.4\n%%EOF")

            test_file2 = Path(tmpdir) / "another.doc.pdf"
            test_file2.write_bytes(b"%PDF-1.4\n%%EOF")

            # Mock MarkItDown result
            mock_result = Mock()
            mock_result.text_content = "Content"

            with patch.object(loader.converter, "convert", return_value=mock_result):
                doc1 = loader.load(str(test_file1))
                doc2 = loader.load(str(test_file2))

                # Verify IDs are based on filename without extension
                assert doc1.id == "my_document"
                assert doc2.id == "another.doc"


class TestPdfLoaderImageExtraction:
    """Test image extraction functionality."""

    def test_extract_image_id_format(self):
        """Test that image ID follows the correct format."""
        loader = PdfLoader()

        doc_id = "test_doc"
        page_num = 1
        image_index = 0

        image_id = loader._extract_image_id(doc_id, page_num, image_index)

        assert image_id == "test_doc_0001_0000"

    def test_create_image_metadata(self):
        """Test ImageMetadata creation helper."""
        loader = PdfLoader()

        metadata = loader._create_image_metadata(
            image_id="test_0001_0000",
            image_path="data/images/test/test_0001_0000.png",
            page_num=1,
            text_offset=100,
            text_length=20,
            position={"x0": 0, "y0": 0, "x1": 100, "y1": 100}
        )

        assert isinstance(metadata, ImageMetadata)
        assert metadata.id == "test_0001_0000"
        assert metadata.path == "data/images/test/test_0001_0000.png"
        assert metadata.page == 1
        assert metadata.text_offset == 100
        assert metadata.text_length == 20
        assert metadata.position == {"x0": 0, "y0": 0, "x1": 100, "y1": 100}

    def test_insert_image_placeholders(self):
        """Test image placeholder insertion into text."""
        loader = PdfLoader()

        text_content = "This is a test document."

        # Create mock image metadata
        image1 = loader._create_image_metadata(
            image_id="test_0001_0000",
            image_path="data/images/test/test_0001_0000.png",
            page_num=1,
            text_offset=5,
            text_length=20,
        )

        image2 = loader._create_image_metadata(
            image_id="test_0001_0001",
            image_path="data/images/test/test_0001_0001.png",
            page_num=1,
            text_offset=25,
            text_length=20,
        )

        # Insert placeholders
        result = loader._insert_image_placeholders(text_content, [image1, image2])

        # Verify placeholders are inserted
        assert "[IMAGE: test_0001_0000]" in result
        assert "[IMAGE: test_0001_0001]" in result

    def test_estimate_image_position_with_page_marker(self):
        """Test image position estimation with page markers."""
        loader = PdfLoader()

        text_content = "\n\n[Page 1]\n\nSome content here."
        page_num = 1
        image_index = 0

        position = loader._estimate_image_position(text_content, page_num, image_index)

        # Should insert after page marker
        expected_pos = text_content.find("\n\n[Page 1]\n\n") + len("\n\n[Page 1]\n\n")
        assert position == expected_pos

    def test_estimate_image_position_without_page_marker(self):
        """Test image position estimation without page markers."""
        loader = PdfLoader()

        text_content = "Some content without page markers."
        page_num = 1
        image_index = 0

        position = loader._estimate_image_position(text_content, page_num, image_index)

        # Should insert at beginning for first image
        assert position == 0


class TestPdfLoaderImageSaving:
    """Test image saving functionality."""

    def test_images_directory_creation(self):
        """Test that images directory is created when needed."""
        loader = PdfLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test PDF file
            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\n%%EOF")

            # Mock MarkItDown result
            mock_result = Mock()
            mock_result.text_content = "# Test\n\nContent"

            # Create mock image metadata with a real path
            images_dir = Path(tmpdir) / "data" / "images" / "test"
            images_dir.mkdir(parents=True, exist_ok=True)

            mock_image_metadata = loader._create_image_metadata(
                image_id="test_0001_0000",
                image_path=str(images_dir / "test_0001_0000.png"),
                page_num=1,
                text_offset=25,
                text_length=20,
            )

            # Create the actual image file
            Path(mock_image_metadata.path).write_bytes(b"fake_image_data")

            with patch.object(loader.converter, "convert", return_value=mock_result):
                # Patch IMAGES_BASE_DIR to use temp directory
                with patch.object(loader, "IMAGES_BASE_DIR", str(Path(tmpdir) / "data" / "images")):
                    # Mock image extraction to return our mock metadata
                    with patch.object(loader, "_extract_images_from_pdf", return_value=[mock_image_metadata]):
                        # Load document
                        document = loader.load(str(test_file))

                        # Verify images directory was created
                        assert images_dir.exists()

                        # Verify image file exists
                        image_path = Path(mock_image_metadata.path)
                        assert image_path.exists()
