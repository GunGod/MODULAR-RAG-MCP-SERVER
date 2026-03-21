"""
Base Vision LLM abstract interface for the Modular RAG MCP Server.

This module defines the abstract interface for Vision LLM providers that
support both text and image inputs. Vision LLMs are used in the Ingestion
Pipeline for image captioning and multimodal content understanding.

Key differences from BaseLLM:
- VisionMessage supports both text and image content
- Images can be provided as URLs or base64-encoded strings
- Maintains compatibility with the existing LLM architecture

Author: Modular RAG MCP Server Project
License: MIT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Union
from enum import Enum


class ImageType(Enum):
    """
    Supported image input types for Vision LLM.

    Attributes:
        URL: Image provided as a URL (http/https)
        BASE64: Image provided as base64-encoded string
        PATH: Local file path (will be read and encoded by provider)
    """
    URL = "url"
    BASE64 = "base64"
    PATH = "path"


@dataclass
class ImageContent:
    """
    Image content for Vision LLM input.

    Attributes:
        data: Image data (URL, base64 string, or file path)
        image_type: Type of image data (url/base64/path)
        mime_type: Optional MIME type (e.g., "image/png", "image/jpeg")
        metadata: Optional additional metadata about the image
    """
    data: str
    image_type: ImageType
    mime_type: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        """Validate image content."""
        if not self.data:
            raise ValueError("Image data cannot be empty")

        # Auto-detect MIME type from image_type if not provided
        if self.mime_type is None and self.image_type == ImageType.BASE64:
            # Try to detect from base64 prefix (e.g., "data:image/png;base64,...")
            if self.data.startswith("data:"):
                try:
                    mime_part = self.data.split(";")[0]
                    self.mime_type = mime_part.replace("data:", "")
                except (IndexError, AttributeError):
                    self.mime_type = "image/png"
            else:
                self.mime_type = "image/png"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary format for API calls.

        Returns:
            Dictionary with image data in a format suitable for API calls
        """
        result = {
            "type": "image",
            "data": self.data,
        }

        if self.mime_type:
            result["mime_type"] = self.mime_type

        return result


@dataclass
class VisionMessage:
    """
    A message in a conversation with a Vision LLM.

    Unlike the regular Message class, VisionMessage supports both text
    and image content, enabling multimodal conversations.

    Attributes:
        role: The role of the message sender (system, user, assistant)
        content: Text content (can be empty string if only images are provided)
        images: Optional list of images associated with this message
    """
    role: str
    content: str
    images: list[ImageContent] = field(default_factory=list)

    def __post_init__(self):
        """Validate role and content."""
        valid_roles = {"system", "user", "assistant"}
        if self.role not in valid_roles:
            raise ValueError(
                f"Invalid role '{self.role}'. Must be one of {valid_roles}"
            )

        # Ensure at least text or images are provided
        if not self.content and not self.images:
            raise ValueError(
                "VisionMessage must have either text content or images"
            )

    def has_images(self) -> bool:
        """Check if this message contains images."""
        return len(self.images) > 0

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary format for API calls.

        Returns:
            Dictionary representation of the message
        """
        result = {
            "role": self.role,
            "content": self.content,
        }

        if self.images:
            result["images"] = [img.to_dict() for img in self.images]

        return result


@dataclass
class VisionResponse:
    """
    Response from a Vision LLM provider.

    Extends the standard LLMResponse with vision-specific metadata
    such as the number of images processed.

    Attributes:
        content: The generated text content
        model: The model name used for generation
        provider: The provider name (e.g., "azure_vision", "openai_vision")
        images_processed: Number of images processed in the request
        usage: Optional token usage information
        raw_response: Optional raw response from the provider
    """
    content: str
    model: str
    provider: str
    images_processed: int = 0
    usage: Optional[dict[str, Any]] = None
    raw_response: Optional[Any] = None


class BaseVisionLLM(ABC):
    """
    Abstract base class for Vision LLM providers.

    Vision LLMs support both text and image inputs, enabling multimodal
    AI capabilities such as image captioning, visual question answering,
    and document understanding.

    Key use cases in this project:
    - Image Captioning: Generate text descriptions of images in documents
    - Multimodal RAG: Enable search across visual content
    - Document Understanding: Extract information from charts/diagrams

    Example:
        >>> vision_llm = VisionLLMFactory.create(settings)
        >>> image = ImageContent(
        ...     data="path/to/image.png",
        ...     image_type=ImageType.PATH
        ... )
        >>> message = VisionMessage(
        ...     role="user",
        ...     content="Describe this image",
        ...     images=[image]
        ... )
        >>> response = vision_llm.chat([message])
        >>> print(response.content)
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ):
        """
        Initialize the Vision LLM provider.

        Args:
            model: Model name (e.g., "gpt-4o", "qwen-vl-max")
            api_key: API key for the provider (if required)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
        """
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._provider_config = kwargs

    @abstractmethod
    def chat(
        self,
        messages: list[VisionMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> VisionResponse:
        """
        Send a chat request with text and/or images to the Vision LLM.

        Args:
            messages: List of messages (can include images)
            temperature: Override the default temperature
            max_tokens: Override the default max_tokens
            **kwargs: Additional provider-specific parameters

        Returns:
            VisionResponse containing the generated text and metadata

        Raises:
            RuntimeError: If the API call fails
            ValueError: If parameters are invalid or images are malformed
        """
        pass

    def describe_image(
        self,
        image: ImageContent,
        prompt: str = "Describe this image in detail.",
        **kwargs
    ) -> VisionResponse:
        """
        Convenience method to generate a description for a single image.

        This is a helper method for the common use case of image captioning.

        Args:
            image: The image to describe
            prompt: The prompt to guide description generation
            **kwargs: Additional provider-specific parameters

        Returns:
            VisionResponse with the image description
        """
        message = VisionMessage(
            role="user",
            content=prompt,
            images=[image]
        )
        return self.chat([message], **kwargs)

    @abstractmethod
    def validate_image_support(self) -> bool:
        """
        Check if the provider supports vision capabilities.

        This method should verify that the configured model has vision
        capabilities and the API is accessible.

        Returns:
            True if vision is supported, False otherwise
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Get the provider name (e.g., "azure_vision", "openai_vision").

        Returns:
            Provider name as a string
        """
        pass

    def __repr__(self) -> str:
        """String representation of the Vision LLM instance."""
        return (
            f"{self.__class__.__name__}("
            f"model='{self.model}', "
            f"provider='{self.provider_name}')"
        )
