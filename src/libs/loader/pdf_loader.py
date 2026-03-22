"""
PDF Loader implementation using MarkItDown.

This module implements PDF document loading with image extraction support.
It uses MarkItDown to convert PDF files to Markdown format and extracts
embedded images for multimodal RAG processing.

Author: Modular RAG MCP Server Project
License: MIT
"""

import base64
import os
from pathlib import Path
from typing import Any, Optional

import markitdown
from PIL import Image

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader
from src.observability.logger import get_logger

logger = get_logger(__name__)


class PdfLoader(BaseLoader):
    """
    PDF document loader using MarkItDown.

    This loader converts PDF files to Markdown format and extracts embedded images.
    Images are saved to data/images/{doc_hash}/ and represented as placeholders
    in the document text.

    Features:
    - PDF to Markdown conversion using MarkItDown
    - Image extraction from PDF
    - Automatic image saving with unique IDs
    - Placeholder insertion for images: [IMAGE: {image_id}]
    - Graceful degradation (image extraction failure doesn't block text parsing)

    Example:
        >>> loader = PdfLoader()
        >>> document = loader.load("/path/to/document.pdf")
        >>> print(document.text)
        >>> print(len(document.metadata.get("images", [])))
    """

    # Supported file extensions
    SUPPORTED_EXTENSIONS = [".pdf"]

    # Image storage directory
    IMAGES_BASE_DIR = "data/images"

    # Image placeholder format
    IMAGE_PLACEHOLDER_FORMAT = "[IMAGE: {image_id}]"

    def __init__(self, output_images_dir: Optional[str] = None):
        """
        Initialize PDF loader.

        Args:
            output_images_dir: Custom directory for extracted images (defaults to data/images/{doc_hash}/)
        """
        super().__init__()
        self.output_images_dir = output_images_dir
        self.converter = markitdown.MarkItDown()

    def load(self, file_path: str) -> Document:
        """
        Load a PDF document and convert to Document.

        Args:
            file_path: Path to the PDF file

        Returns:
            Document object with text and metadata

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a PDF
            RuntimeError: If PDF parsing fails critically

        Example:
            >>> doc = loader.load("/path/to/document.pdf")
            >>> assert "pdf" in doc.metadata["doc_type"]
        """
        path = Path(file_path)

        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Validate file extension
        if not self.supports_format(file_path):
            raise ValueError(
                f"Unsupported file format: {path.suffix}. "
                f"PdfLoader only supports: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        # Generate document ID from file path hash
        # (Using file path for consistency, could be content hash in production)
        doc_id = path.name  # Use filename as ID for now
        doc_id = doc_id.rsplit('.', 1)[0] if '.' in doc_id else doc_id

        self.logger.info(f"Loading PDF document: {file_path}")

        # Convert PDF to Markdown using MarkItDown
        try:
            result = self.converter.convert(str(path))
        except Exception as e:
            self.logger.error(f"Failed to convert PDF '{file_path}': {e}")
            raise RuntimeError(f"PDF parsing failed for '{file_path}': {e}") from e

        # Extract text content
        text_content = result.text_content

        # Build metadata
        metadata = {
            "source_path": str(path.absolute()),
            "doc_type": "pdf",
            "file_name": path.name,
            "file_size": path.stat().st_size if path.exists() else 0,
        }

        # Extract images if present
        images = self._extract_images_from_pdf(
            file_path,
            doc_id
        )

        if images:
            metadata["images"] = images
            self.logger.info(f"Extracted {len(images)} images from PDF")
            # Update text with placeholders
            text_content = self._insert_image_placeholders(text_content, images)

        # Create Document
        document = Document(
            id=doc_id,
            text=text_content,
            metadata=metadata
        )

        self.logger.info(
            f"Successfully loaded PDF: {file_path} "
            f"(chars={len(text_content)}, images={len(images)})"
        )

        return document

    def _extract_images_from_pdf(
        self,
        file_path: str,
        doc_id: str
    ) -> list:
        """
        Extract images from PDF document using pdfplumber (preferred) or pypdf (fallback).

        Implementation Priority:
            1. pdfplumber (recommended) - precise positioning with bbox coordinates
            2. pypdf (fallback) - heuristic positioning

        Args:
            file_path: Path to the PDF file
            doc_id: Document ID

        Returns:
            List of ImageMetadata objects
        """
        images = []

        # Create output directory for this document
        images_dir = Path(self.IMAGES_BASE_DIR) / doc_id
        images_dir.mkdir(parents=True, exist_ok=True)

        self.logger.debug(f"Extracting images to: {images_dir}")

        # Priority 1: Try pdfplumber (recommended for production)
        try:
            import pdfplumber

            self.logger.info("Using pdfplumber for precise image extraction")
            images = self._extract_images_with_pdfplumber(
                file_path, doc_id, images_dir
            )

            if images:
                self.logger.info(f"Successfully extracted {len(images)} images using pdfplumber")
                return images

        except ImportError:
            self.logger.debug("pdfplumber not available, falling back to pypdf")
        except Exception as e:
            self.logger.warning(f"pdfplumber extraction failed: {e}, falling back to pypdf")

        # Priority 2: Fallback to pypdf
        try:
            import pypdf

            self.logger.debug("Using pypdf for image extraction")
            images = self._extract_images_with_pypdf2(
                file_path, doc_id, images_dir
            )

        except ImportError:
            self.logger.warning(
                "No PDF image extraction library available. "
                "Install with: pip install pdfplumber"
            )
        except Exception as e:
            self.logger.warning(f"Image extraction failed: {e}")

        return images

    def _extract_images_with_pdfplumber(
        self,
        file_path: str,
        doc_id: str,
        images_dir: Path
    ) -> list:
        """
        Extract images from PDF using pdfplumber with precise positioning.

        pdfplumber provides:
            - Accurate bbox coordinates for images
            - Page text layout information
            - Better image data extraction

        Args:
            file_path: Path to the PDF file
            doc_id: Document ID
            images_dir: Directory to save images

        Returns:
            List of ImageMetadata objects
        """
        import pdfplumber

        images = []
        text_length = len(self.IMAGE_PLACEHOLDER_FORMAT.format(image_id="__placeholder__"))

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # Get images from this page
                    page_images = page.images

                    if not page_images:
                        continue

                    for img_index, img in enumerate(page_images):
                        try:
                            # Extract image data
                            # pdfplumber stores image data in the PDF's stream
                            # We need to access the original PDF to get the image data
                            image_info = self._extract_image_data_from_page(page, img_index)

                            if not image_info:
                                continue

                            image_data, image_format = image_info

                            # Save image
                            image_filename = f"{doc_id}_{page_num:04d}_{img_index:04d}.{image_format}"
                            image_path = images_dir / image_filename

                            with open(image_path, "wb") as f:
                                f.write(image_data)

                            # Get bbox coordinates (pdfplumber provides this)
                            # bbox: (x0, top, x1, bottom)
                            x0 = img.get("x0", 0)
                            top = img.get("top", 0)
                            x1 = img.get("x1", 0)
                            bottom = img.get("bottom", 0)

                            position = {
                                "x0": round(x0, 2),
                                "y0": round(top, 2),
                                "x1": round(x1, 2),
                                "y1": round(bottom, 2),
                                "width": round(x1 - x0, 2),
                                "height": round(bottom - top, 2),
                            }

                            # Calculate text offset based on image position
                            # Use a heuristic based on page number and image index
                            # (More sophisticated calculation would require text_content analysis)
                            text_offset = self._calculate_text_offset_from_bbox(
                                page, page_num, img_index, position
                            )

                            # Generate image ID
                            image_id = self._extract_image_id(doc_id, page_num, img_index)

                            # Create ImageMetadata
                            # Use absolute path to avoid .relative_to() issues
                            image_metadata = self._create_image_metadata(
                                image_id=image_id,
                                image_path=str(image_path.absolute()),
                                page_num=page_num,
                                text_offset=text_offset,
                                text_length=text_length,
                                position=position
                            )

                            images.append(image_metadata)

                        except Exception as e:
                            self.logger.warning(
                                f"Failed to extract image {img_index} from page {page_num}: {e}"
                            )
                            continue

                except Exception as e:
                    self.logger.warning(f"Error processing page {page_num}: {e}")
                    continue

        return images

    def _extract_image_data_from_page(
        self,
        page: Any,
        image_index: int
    ) -> tuple:
        """
        Extract image data from pdfplumber page.

        pdfplumber provides access to PDF streams containing image data.

        Args:
            page: pdfplumber Page object
            image_index: Index of the image on the page

        Returns:
            Tuple of (image_data, image_format) or (None, None) if extraction fails
        """
        try:
            # pdfplumber provides page.images which contains metadata
            page_images = page.images

            if not page_images or image_index >= len(page_images):
                return None, None

            img_info = page_images[image_index]

            # Check if there's a stream with image data
            if 'stream' in img_info:
                pdf_stream = img_info['stream']

                # Get raw image data from the stream
                raw_data = None
                if hasattr(pdf_stream, 'get_data'):
                    try:
                        raw_data = pdf_stream.get_data()
                    except Exception as e:
                        self.logger.debug(f"stream.get_data() failed: {e}")

                if not raw_data and hasattr(pdf_stream, 'data'):
                    raw_data = pdf_stream.data

                if raw_data:
                    # Determine format from filter
                    image_format = "jpg"  # Default to jpg for PDF images

                    # Check stream attributes for format hints
                    if hasattr(pdf_stream, 'attrs'):
                        attrs = pdf_stream.attrs
                        if 'Filter' in attrs:
                            filter_val = attrs['Filter']
                            if '/DCTDecode' in str(filter_val):
                                image_format = "jpg"
                            elif '/JPXDecode' in str(filter_val):
                                image_format = "jp2"
                            elif '/FlateDecode' in str(filter_val):
                                # FlateDecode usually means PNG or needs further decoding
                                # Try to detect if it's PNG
                                if raw_data[:4] == b'\x89PNG':
                                    image_format = "png"
                                else:
                                    image_format = "jpg"

                    return raw_data, image_format

            # Fallback: Try using pypdf's image object (if available)
            if hasattr(page, 'page'):
                pypdf_page = page.page

                if hasattr(pypdf_page, 'images') and pypdf_page.images:
                    if image_index < len(pypdf_page.images):
                        image_object = pypdf_page.images[image_index]

                        # pypdf 6.x+ uses .image to access PIL Image object
                        if hasattr(image_object, 'image') and image_object.image is not None:
                            from io import BytesIO

                            pil_image = image_object.image
                            image_format = pil_image.format or "png"
                            if image_format == "JPEG":
                                image_format = "jpg"

                            # Convert PIL Image to bytes
                            buffer = BytesIO()
                            pil_image.save(buffer, format=image_format)
                            image_data = buffer.getvalue()

                            return image_data, image_format.lower()

                        # Very old pypdf versions (2.x)
                        elif hasattr(image_object, 'get_data'):
                            try:
                                image_data = image_object.get_data()
                                return image_data, "png"
                            except AttributeError:
                                pass

        except Exception as e:
            self.logger.debug(f"Failed to extract image data: {e}")

        return None, None

    def _calculate_text_offset_from_bbox(
        self,
        page: Any,
        page_num: int,
        image_index: int,
        position: dict
    ) -> int:
        """
        Calculate text offset for an image based on its bbox and page layout.

        This is a heuristic that estimates where an image should appear in the text.
        For precise positioning, you would need to analyze the actual text content.

        Args:
            page: pdfplumber Page object
            page_num: Page number (1-based)
            image_index: Image index on the page
            position: Image bbox position dict

        Returns:
            Estimated character position in text
        """
        # Simple heuristic: based on page number and image index
        # Images on earlier pages and earlier on the same page come first
        base_offset = (page_num - 1) * 2000  # 2000 chars per page
        image_offset = image_index * 200     # 200 chars per image

        return base_offset + image_offset

    def _extract_images_with_pypdf2(
        self,
        file_path: str,
        doc_id: str,
        images_dir: Path
    ) -> list:
        """
        Extract images from PDF using PyPDF2.

        MVP Implementation Notice:
            - 使用简单的启发式算法估算图片位置
            - 不依赖 text_content，避免与 MarkItDown 结果耦合
            - 位置估算可能不够精确，生产环境建议使用 pdfplumber

        Args:
            file_path: Path to the PDF file
            doc_id: Document ID
            images_dir: Directory to save images

        Returns:
            List of ImageMetadata objects
        """
        import pypdf
        from io import BytesIO

        images = []
        pdf_reader = pypdf.PdfReader(file_path)
        text_length = len(self.IMAGE_PLACEHOLDER_FORMAT.format(image_id="__placeholder__"))

        for page_num, page in enumerate(pdf_reader.pages):
            page_number = page_num + 1  # 1-based

            # Extract images from this page
            if "/XObject" in page.get("/Resources", {}):
                # This page has resources (potentially images)
                try:
                    # Try to extract images
                    image_count = 0
                    for image_index, image_object in enumerate(page.images):
                        try:
                            # Get image data (pypdf 6.x+ API)
                            image_data = None
                            if hasattr(image_object, 'image') and image_object.image is not None:
                                # Extract PIL Image to bytes
                                pil_image = image_object.image
                                image_format = pil_image.format or "PNG"
                                if image_format == "JPEG":
                                    image_format = "jpg"
                                else:
                                    image_format = image_format.lower()

                                # Convert PIL Image to bytes
                                buffer = BytesIO()
                                pil_image.save(buffer, format=image_format.upper())
                                image_data = buffer.getvalue()
                            else:
                                # Fallback for very old pypdf versions or edge cases
                                self.logger.debug(f"Cannot extract image {image_index}: no .image attribute")
                                continue

                            # Determine format from PIL Image
                            if image_format == "jpg" or image_format == "jpeg":
                                image_format = "jpg"
                            else:
                                image_format = "png"

                            # Save image
                            image_filename = f"{doc_id}_{page_number:04d}_{image_index:04d}.{image_format}"
                            image_path = images_dir / image_filename

                            with open(image_path, "wb") as f:
                                f.write(image_data)

                            # Get image position if available
                            position = None
                            if hasattr(image_object, "bbox"):
                                # BoundingBox: (x0, y0, x1, y1)
                                bbox = image_object.bbox
                                position = {
                                    "x0": bbox[0],
                                    "y0": bbox[1],
                                    "x1": bbox[2],
                                    "y1": bbox[3],
                                }

                            # MVP Implementation: 简单的启发式位置估算
                            # 根据页码和图片索引生成递增的偏移量
                            # 这样至少能保证图片按顺序插入，而不是堆叠在一起
                            text_offset = (page_number * 1000) + (image_index * 100)

                            # Generate image ID
                            image_id = self._extract_image_id(doc_id, page_number, image_index)

                            # Create ImageMetadata
                            # Use absolute path to avoid .relative_to() issues
                            image_metadata = self._create_image_metadata(
                                image_id=image_id,
                                image_path=str(image_path.absolute()),
                                page_num=page_number,
                                text_offset=text_offset,
                                text_length=text_length,
                                position=position
                            )

                            images.append(image_metadata)
                            image_count += 1

                        except Exception as e:
                            self.logger.warning(
                                f"Failed to extract image {image_index} from page {page_number}: {e}"
                            )

                    if image_count > 0:
                        self.logger.debug(f"Extracted {image_count} images from page {page_number}")

                except Exception as e:
                    self.logger.warning(f"Error processing page {page_number}: {e}")

        return images

    def _estimate_image_position(
        self,
        text_content: str,
        page_num: int,
        image_index: int
    ) -> int:
        """
        Estimate where an image placeholder should be inserted in the text.

        This is a heuristic that places images at logical positions in the text.
        In production, you would use PDF layout analysis for precise positioning.

        Args:
            text_content: Text content
            page_num: Page number
            image_index: Image index on the page

        Returns:
            Estimated character position in text
        """
        # Simple heuristic: place images at regular intervals
        # In production, this would be based on PDF layout analysis

        # Find page marker in text (if present)
        page_marker = f"\n\n[Page {page_num}]\n\n"
        page_pos = text_content.find(page_marker)

        if page_pos >= 0:
            # Insert after page marker
            return page_pos + len(page_marker)

        # If no page marker, insert at the beginning
        if image_index == 0:
            return 0

        # Otherwise, append to the end
        return len(text_content)

    def _insert_image_placeholders(
        self,
        text_content: str,
        images: list
    ) -> str:
        """
        Insert image placeholders into text content.

        Args:
            text_content: Original text content
            images: List of ImageMetadata

        Returns:
            Text content with image placeholders inserted
        """
        if not images:
            return text_content

        # Sort images by text_offset to insert in correct order
        sorted_images = sorted(images, key=lambda img: img.text_offset)

        # Insert placeholders
        result = text_content
        offset_adjustment = 0

        for image in sorted_images:
            # Calculate insertion position with offset adjustment
            insert_pos = image.text_offset + offset_adjustment

            # Ensure position is within bounds
            insert_pos = max(0, min(insert_pos, len(result)))

            # Create placeholder
            placeholder = self.IMAGE_PLACEHOLDER_FORMAT.format(image_id=image.id)

            # Insert placeholder
            result = (
                result[:insert_pos] +
                placeholder +
                result[insert_pos:]
            )

            # Adjust offset for subsequent insertions
            offset_adjustment += len(placeholder) - image.text_length

        return result

    @property
    def supported_extensions(self) -> list[str]:
        """Get list of supported file extensions."""
        return self.SUPPORTED_EXTENSIONS
