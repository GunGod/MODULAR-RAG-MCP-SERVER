"""
Azure OpenAI Vision LLM provider implementation.

This module implements the BaseVisionLLM interface for Azure OpenAI Service
with vision capabilities. It supports multimodal inputs (text + images) using
GPT-4o and other vision-enabled models.

Author: Modular RAG MCP Server Project
License: MIT
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Optional

import httpx
from PIL import Image

from src.libs.llm.base_vision_llm import (
    BaseVisionLLM,
    VisionMessage,
    VisionResponse,
    ImageContent,
    ImageType,
)
from src.observability.logger import get_logger

logger = get_logger(__name__)


class AzureVisionLLM(BaseVisionLLM):
    """
    Azure OpenAI Vision LLM provider implementation.

    This class implements multimodal chat completions using Azure OpenAI Service
    with vision-enabled models like GPT-4o. It supports both text and image inputs.

    Azure OpenAI Vision API format:
        https://{resource_name}.openai.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}

    The API accepts images in two formats:
        1. URL: Public HTTP/HTTPS URL
        2. Base64: Base64-encoded image data with MIME type prefix

    Example:
        >>> vision_llm = AzureVisionLLM(
        ...     model="gpt-4o",
        ...     azure_endpoint="https://my-resource.openai.azure.com",
        ...     api_key="...",
        ...     api_version="2024-02-15-preview"
        ... )
        >>> image = ImageContent(
        ...     data="https://example.com/image.png",
        ...     image_type=ImageType.URL
        ... )
        >>> message = VisionMessage(
        ...     role="user",
        ...     content="Describe this image",
        ...     images=[image]
        ... )
        >>> response = vision_llm.chat([message])
        >>> print(response.content)
    """

    # Azure OpenAI API configuration
    DEFAULT_API_VERSION = "2024-02-15-preview"
    CHAT_COMPLETION_PATH = "/openai/deployments"
    MAX_IMAGE_SIZE = 2048  # Maximum image dimension (width or height)

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        max_image_size: int = MAX_IMAGE_SIZE,
        **kwargs
    ):
        """
        Initialize Azure OpenAI Vision LLM provider.

        Args:
            model: Deployment name (e.g., "gpt-4o", "gpt-4-vision-preview")
            api_key: Azure OpenAI API key
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            azure_endpoint: Azure OpenAI endpoint (e.g., "https://my-resource.openai.azure.com")
            api_version: API version (defaults to "2024-02-15-preview")
            max_image_size: Maximum dimension for image resizing (default 2048)
            **kwargs: Additional provider-specific parameters

        Raises:
            ValueError: If required Azure configuration is missing
        """
        super().__init__(model, api_key, temperature, max_tokens, **kwargs)

        # Azure-specific configuration
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version or self.DEFAULT_API_VERSION
        self.max_image_size = max_image_size

        # Validate Azure configuration
        if not self.azure_endpoint:
            raise ValueError(
                f"[{self.provider_name}] azure_endpoint is required. "
                f"Provide your Azure OpenAI endpoint (e.g., 'https://my-resource.openai.azure.com')."
            )

        if not self.api_key:
            raise ValueError(
                f"[{self.provider_name}] API key is required. "
                f"Provide it via api_key parameter or AZURE_OPENAI_API_KEY environment variable."
            )

        # Ensure endpoint doesn't have trailing slash
        self.azure_endpoint = self.azure_endpoint.rstrip('/')

        # Build the full chat completion URL
        # Format: https://{endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={version}
        self.chat_completion_url = (
            f"{self.azure_endpoint}{self.CHAT_COMPLETION_PATH}"
            f"/{self.model}/chat/completions"
            f"?api-version={self.api_version}"
        )

        # Initialize HTTP client with longer timeout for image processing
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "api-key": self.api_key,  # Azure also uses api-key header
                "Content-Type": "application/json",
            },
            timeout=120.0,  # Longer timeout for image processing
        )

        logger.info(
            f"Initialized Azure OpenAI Vision LLM: "
            f"deployment='{self.model}', "
            f"endpoint='{self.azure_endpoint}', "
            f"api_version='{self.api_version}', "
            f"max_image_size={self.max_image_size}"
        )

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "azure_vision"

    def chat(
        self,
        messages: list[VisionMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> VisionResponse:
        """
        Send a multimodal chat completion request to Azure OpenAI Vision API.

        Args:
            messages: List of messages (can include images)
            temperature: Override the default temperature
            max_tokens: Override the default max_tokens
            **kwargs: Additional Azure-specific parameters

        Returns:
            VisionResponse containing the generated text and metadata

        Raises:
            ValueError: If messages list is empty or invalid
            RuntimeError: If the API call fails

        Example:
            >>> image = ImageContent(
            ...     data="https://example.com/image.png",
            ...     image_type=ImageType.URL
            ... )
            >>> message = VisionMessage(
            ...     role="user",
            ...     content="Describe this image",
            ...     images=[image]
            ... )
            >>> response = vision_llm.chat([message])
            >>> print(response.content)
        """
        # Validate input
        if not messages:
            raise ValueError(
                f"[{self.provider_name}] Messages list cannot be empty. "
                f"Please provide at least one message."
            )

        if not all(isinstance(msg, VisionMessage) for msg in messages):
            raise ValueError(
                f"[{self.provider_name}] All items in messages must be VisionMessage instances. "
                f"Got types: {[type(msg).__name__ for msg in messages]}"
            )

        # Count total images in request
        total_images = sum(len(msg.images) for msg in messages)

        # Prepare request payload
        # Azure Vision API format: each message has content array with text and image items
        api_messages = []
        for msg in messages:
            content_items = []

            # Add text content
            if msg.content:
                content_items.append({
                    "type": "text",
                    "text": msg.content
                })

            # Add image content
            for image in msg.images:
                image_url = self._prepare_image_for_api(image)
                content_items.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })

            # If message has no text but has images, add empty text
            if not msg.content and msg.images:
                content_items.insert(0, {
                    "type": "text",
                    "text": ""  # Empty text for image-only messages
                })

            api_messages.append({
                "role": msg.role,
                "content": content_items
            })

        # Build request payload
        payload = {
            "messages": api_messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            **kwargs  # Pass through any additional Azure parameters
        }

        # Remove None values from payload
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            # Make API request
            logger.debug(
                f"Sending vision chat request to {self.provider_name}: "
                f"deployment='{self.model}', messages={len(messages)}, images={total_images}"
            )

            response = self._client.post(
                self.chat_completion_url,
                json=payload
            )
            response.raise_for_status()

            # Parse response
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]

            # Build VisionResponse
            vision_response = VisionResponse(
                content=message["content"],
                model=data.get("model", self.model),
                provider=self.provider_name,
                images_processed=total_images,
                usage={
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
                    "completion_tokens": data.get("usage", {}).get("completion_tokens"),
                    "total_tokens": data.get("usage", {}).get("total_tokens"),
                },
                raw_response=data
            )

            logger.debug(
                f"Received response from {self.provider_name}: "
                f"{vision_response.usage.get('total_tokens')} total tokens, "
                f"{total_images} images processed"
            )

            return vision_response

        except httpx.HTTPStatusError as e:
            # Handle HTTP errors with clear error messages
            status_code = e.response.status_code
            error_detail = "Unknown error"

            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", str(e))

                # Add Azure-specific error codes
                error_code = error_data.get("error", {}).get("code")
                if error_code:
                    error_detail = f"[{error_code}] {error_detail}"
            except Exception:
                error_detail = str(e)

            error_msg = (
                f"[{self.provider_name}] API request failed with status {status_code}. "
                f"Endpoint: '{self.azure_endpoint}'. "
                f"Deployment: '{self.model}'. "
                f"API Version: '{self.api_version}'. "
                f"Error: {error_detail}"
            )

            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except httpx.RequestError as e:
            # Handle network/connection errors
            error_msg = (
                f"[{self.provider_name}] Network error occurred while communicating with API. "
                f"Endpoint: '{self.azure_endpoint}'. "
                f"Deployment: '{self.model}'. "
                f"Error: {str(e)}"
            )

            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except Exception as e:
            # Handle unexpected errors
            error_msg = (
                f"[{self.provider_name}] Unexpected error during vision chat completion. "
                f"Endpoint: '{self.azure_endpoint}'. "
                f"Deployment: '{self.model}'. "
                f"Error type: {type(e).__name__}. "
                f"Error: {str(e)}"
            )

            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def validate_image_support(self) -> bool:
        """
        Check if the provider supports vision capabilities.

        For Azure Vision LLM, we verify that:
        1. The API endpoint is accessible
        2. The model deployment exists

        Returns:
            True if vision is supported, False otherwise
        """
        try:
            # Make a minimal API request to check connectivity
            # Use a simple text-only message to validate the endpoint
            test_message = VisionMessage(
                role="user",
                content="Hello"  # Simple text message
            )

            # Use very short timeout for validation
            original_timeout = self._client.timeout
            self._client.timeout = httpx.Timeout(5.0)

            try:
                response = self.chat([test_message], max_tokens=5)
                return True
            finally:
                # Restore original timeout
                self._client.timeout = original_timeout

        except Exception as e:
            logger.warning(
                f"[{self.provider_name}] Vision support validation failed: {e}"
            )
            return False

    def _prepare_image_for_api(self, image: ImageContent) -> str:
        """
        Prepare an image for Azure OpenAI Vision API.

        This method handles different image types:
        - URL: Used as-is
        - BASE64: Used as-is (already has data URI prefix)
        - PATH: Read from file, resize if needed, convert to base64

        Args:
            image: ImageContent to prepare

        Returns:
            Base64 data URI or URL string for the API
        """
        if image.image_type == ImageType.URL:
            # URL can be used directly
            return image.data

        elif image.image_type == ImageType.BASE64:
            # Base64 is already formatted
            # Ensure it has the data URI prefix
            if image.data.startswith("data:"):
                return image.data
            else:
                # Add data URI prefix with detected MIME type
                mime_type = image.mime_type or "image/png"
                return f"data:{mime_type};base64,{image.data}"

        elif image.image_type == ImageType.PATH:
            # Read from file and convert to base64
            image_path = Path(image.data)

            if not image_path.exists():
                raise ValueError(
                    f"[{self.provider_name}] Image file not found: {image.data}"
                )

            try:
                # Open and possibly resize the image
                with Image.open(image_path) as img:
                    # Resize if image is too large
                    img = self._resize_image_if_needed(img)

                    # Convert to base64
                    buffered = BytesIO()
                    img.save(buffered, format=img.format or "PNG")
                    img_bytes = buffered.getvalue()
                    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

                    # Detect MIME type
                    mime_type = image.mime_type or self._get_mime_type(img.format)

                    return f"data:{mime_type};base64,{img_base64}"

            except Exception as e:
                raise RuntimeError(
                    f"[{self.provider_name}] Failed to process image file '{image.data}': {e}"
                ) from e

        else:
            raise ValueError(
                f"[{self.provider_name}] Unsupported image type: {image.image_type}"
            )

    def _resize_image_if_needed(self, img: Image.Image) -> Image.Image:
        """
        Resize image if it exceeds the maximum dimension.

        Maintains aspect ratio while ensuring neither dimension exceeds max_image_size.

        Args:
            img: PIL Image to resize

        Returns:
            Resized PIL Image or original if no resizing needed
        """
        width, height = img.size

        # Check if resizing is needed
        if width <= self.max_image_size and height <= self.max_image_size:
            return img

        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = self.max_image_size
            new_height = int(height * (self.max_image_size / width))
        else:
            new_height = self.max_image_size
            new_width = int(width * (self.max_image_size / height))

        logger.debug(
            f"[{self.provider_name}] Resizing image from {width}x{height} to {new_width}x{new_height}"
        )

        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _get_mime_type(self, image_format: Optional[str]) -> str:
        """
        Get MIME type from PIL image format.

        Args:
            image_format: PIL image format (e.g., "JPEG", "PNG")

        Returns:
            MIME type string
        """
        format_to_mime = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "GIF": "image/gif",
            "WEBP": "image/webp",
            "BMP": "image/bmp",
        }

        return format_to_mime.get(image_format.upper() if image_format else "PNG", "image/png")

    def __del__(self):
        """Clean up HTTP client when instance is destroyed."""
        if hasattr(self, '_client'):
            self._client.close()
