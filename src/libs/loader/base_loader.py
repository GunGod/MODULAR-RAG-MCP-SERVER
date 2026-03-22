"""
Base Loader abstract interface for document loading.

This module defines the abstract interface that all loader implementations
must follow. Loaders are responsible for parsing document files and
converting them into the standardized Document format.

Author: Modular RAG MCP Server Project
License: MIT
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.core.types import Document, ImageMetadata
from src.observability.logger import get_logger

logger = get_logger(__name__)


class BaseLoader(ABC):
    """
    Abstract base class for document loaders.

    Loaders are responsible for parsing document files and converting
    them into the standardized Document format defined in src/core/types.py.

    All loaders must:
    1. Parse document files into text content
    2. Extract metadata (at minimum: source_path, doc_type)
    3. Handle images if present (extract and save)
    4. Handle errors gracefully (image extraction failure should not block text parsing)

    Example:
        >>> loader = PdfLoader()
        >>> document = loader.load("/path/to/document.pdf")
        >>> print(document.text)
        >>> print(document.metadata["source_path"])
    """

    def __init__(self):
        """Initialize the loader."""
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def load(self, file_path: str) -> Document:
        """
        Load a document from a file.

        Args:
            file_path: Path to the document file

        Returns:
            Document object with text and metadata

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
            RuntimeError: If parsing fails critically

        Example:
            >>> doc = loader.load("/path/to/document.pdf")
            >>> assert doc.metadata["source_path"] == "/path/to/document.pdf"
            >>> assert doc.text is not None
        """
        pass

    def _extract_image_id(self, doc_hash: str, page_num: int, image_index: int) -> str:
        """
        Generate a unique image ID.

        Format: {doc_hash}_{page_num:04d}_{image_index:04d}

        Args:
            doc_hash: Document hash (from Document.id)
            page_num: Page number where image appears
            image_index: Image index on the page

        Returns:
            Unique image ID string
        """
        return f"{doc_hash}_{page_num:04d}_{image_index:04d}"

    def _create_image_metadata(
        self,
        image_id: str,
        image_path: str,
        page_num: int,
        text_offset: int,
        text_length: int,
        position: Optional[dict] = None
    ) -> ImageMetadata:
        """
        Create ImageMetadata object for an extracted image.

        Args:
            image_id: Unique image identifier
            image_path: Storage path for the image
            page_num: Page number where image appears
            text_offset: Character position of placeholder in Document.text
            text_length: Length of the placeholder text
            position: Optional position information

        Returns:
            ImageMetadata object
        """
        return ImageMetadata(
            id=image_id,
            path=image_path,
            page=page_num,
            text_offset=text_offset,
            text_length=text_length,
            position=position
        )

    def supports_format(self, file_path: str) -> bool:
        """
        Check if this loader supports the given file format.

        Args:
            file_path: Path to the file

        Returns:
            True if supported, False otherwise
        """
        # Default implementation checks file extension
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """
        Get list of supported file extensions.

        Returns:
            List of supported file extensions (e.g., [".pdf", ".txt"])
        """
        pass
