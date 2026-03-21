"""
Unit tests for BaseVisionLLM abstract interface.

These tests verify the Vision LLM data structures and abstract interface
definitions. Tests for concrete implementations (Azure Vision LLM, etc.)
will be in separate files.

Author: Modular RAG MCP Server Project
License: MIT
"""

import pytest
from dataclasses import FrozenInstanceError

from src.libs.llm.base_vision_llm import (
    BaseVisionLLM,
    VisionMessage,
    VisionResponse,
    ImageContent,
    ImageType,
)


class TestImageType:
    """Test ImageType enum."""

    def test_image_type_values(self):
        """Test ImageType enum has correct values."""
        assert ImageType.URL.value == "url"
        assert ImageType.BASE64.value == "base64"
        assert ImageType.PATH.value == "path"


class TestImageContent:
    """Test ImageContent dataclass."""

    def test_create_image_content_with_url(self):
        """Test creating ImageContent with URL type."""
        image = ImageContent(
            data="https://example.com/image.png",
            image_type=ImageType.URL,
            mime_type="image/png"
        )

        assert image.data == "https://example.com/image.png"
        assert image.image_type == ImageType.URL
        assert image.mime_type == "image/png"
        assert image.metadata is None

    def test_create_image_content_with_base64(self):
        """Test creating ImageContent with base64 type."""
        image = ImageContent(
            data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            image_type=ImageType.BASE64
        )

        # MIME type should be auto-detected for base64
        assert image.mime_type == "image/png"

    def test_create_image_content_with_base64_prefix(self):
        """Test MIME type detection from base64 data URI prefix."""
        image = ImageContent(
            data="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gA7Q1...",
            image_type=ImageType.BASE64
        )

        # Should detect image/jpeg from prefix
        assert image.mime_type == "image/jpeg"

    def test_create_image_content_with_path(self):
        """Test creating ImageContent with PATH type."""
        image = ImageContent(
            data="/path/to/image.png",
            image_type=ImageType.PATH,
            mime_type="image/png"
        )

        assert image.data == "/path/to/image.png"
        assert image.image_type == ImageType.PATH
        assert image.mime_type == "image/png"

    def test_create_image_content_with_metadata(self):
        """Test creating ImageContent with metadata."""
        image = ImageContent(
            data="https://example.com/image.png",
            image_type=ImageType.URL,
            metadata={"width": 800, "height": 600}
        )

        assert image.metadata == {"width": 800, "height": 600}

    def test_empty_data_raises_error(self):
        """Test that empty image data raises ValueError."""
        with pytest.raises(ValueError, match="Image data cannot be empty"):
            ImageContent(data="", image_type=ImageType.URL)

    def test_to_dict(self):
        """Test ImageContent to_dict method."""
        image = ImageContent(
            data="https://example.com/image.png",
            image_type=ImageType.URL,
            mime_type="image/png"
        )

        result = image.to_dict()

        assert result["type"] == "image"
        assert result["data"] == "https://example.com/image.png"
        assert result["mime_type"] == "image/png"


class TestVisionMessage:
    """Test VisionMessage dataclass."""

    def test_create_text_only_message(self):
        """Test creating VisionMessage with only text content."""
        message = VisionMessage(
            role="user",
            content="What is in this image?"
        )

        assert message.role == "user"
        assert message.content == "What is in this image?"
        assert message.images == []
        assert not message.has_images()

    def test_create_message_with_images(self):
        """Test creating VisionMessage with images."""
        image = ImageContent(
            data="https://example.com/image.png",
            image_type=ImageType.URL
        )

        message = VisionMessage(
            role="user",
            content="Describe this image",
            images=[image]
        )

        assert message.role == "user"
        assert message.content == "Describe this image"
        assert len(message.images) == 1
        assert message.has_images()

    def test_create_message_with_multiple_images(self):
        """Test creating VisionMessage with multiple images."""
        images = [
            ImageContent(data=f"https://example.com/image{i}.png", image_type=ImageType.URL)
            for i in range(3)
        ]

        message = VisionMessage(
            role="user",
            content="Compare these images",
            images=images
        )

        assert len(message.images) == 3
        assert message.has_images()

    def test_invalid_role_raises_error(self):
        """Test that invalid role raises ValueError."""
        with pytest.raises(ValueError, match="Invalid role"):
            VisionMessage(role="invalid", content="test")

    def test_empty_content_and_images_raises_error(self):
        """Test that empty content with no images raises ValueError."""
        with pytest.raises(ValueError, match="must have either text content or images"):
            VisionMessage(role="user", content="", images=[])

    def test_has_images_returns_false_for_text_only(self):
        """Test has_images returns False for text-only messages."""
        message = VisionMessage(role="user", content="Hello")
        assert not message.has_images()

    def test_to_dict(self):
        """Test VisionMessage to_dict method."""
        image = ImageContent(
            data="https://example.com/image.png",
            image_type=ImageType.URL,
            mime_type="image/png"
        )

        message = VisionMessage(
            role="user",
            content="Describe this",
            images=[image]
        )

        result = message.to_dict()

        assert result["role"] == "user"
        assert result["content"] == "Describe this"
        assert "images" in result
        assert len(result["images"]) == 1


class TestVisionResponse:
    """Test VisionResponse dataclass."""

    def test_create_vision_response(self):
        """Test creating VisionResponse."""
        response = VisionResponse(
            content="This is a detailed description of the image",
            model="gpt-4o",
            provider="azure_vision",
            images_processed=1
        )

        assert response.content == "This is a detailed description of the image"
        assert response.model == "gpt-4o"
        assert response.provider == "azure_vision"
        assert response.images_processed == 1
        assert response.usage is None
        assert response.raw_response is None

    def test_create_vision_response_with_usage(self):
        """Test creating VisionResponse with usage metadata."""
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }

        response = VisionResponse(
            content="Description",
            model="gpt-4o",
            provider="azure_vision",
            images_processed=1,
            usage=usage
        )

        assert response.usage == usage


class TestBaseVisionLLM:
    """Test BaseVisionLLM abstract class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseVisionLLM cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseVisionLLM(model="gpt-4o")

    def test_concrete_implementation(self):
        """Test creating a concrete implementation of BaseVisionLLM."""
        class MockVisionLLM(BaseVisionLLM):
            def __init__(self, model="test-model"):
                super().__init__(model=model)

            def chat(self, messages, temperature=None, max_tokens=None, **kwargs):
                return VisionResponse(
                    content="Test response",
                    model=self.model,
                    provider="mock",
                    images_processed=sum(len(msg.images) for msg in messages)
                )

            def validate_image_support(self):
                return True

            @property
            def provider_name(self):
                return "mock"

        # Should be able to instantiate concrete class
        mock_llm = MockVisionLLM(model="test-model")

        assert mock_llm.model == "test-model"
        assert mock_llm.provider_name == "mock"

    def test_describe_image_convenience_method(self):
        """Test the describe_image convenience method."""
        class MockVisionLLM(BaseVisionLLM):
            def __init__(self):
                super().__init__(model="test-model")

            def chat(self, messages, temperature=None, max_tokens=None, **kwargs):
                return VisionResponse(
                    content="A beautiful landscape",
                    model=self.model,
                    provider="mock",
                    images_processed=sum(len(msg.images) for msg in messages)
                )

            def validate_image_support(self):
                return True

            @property
            def provider_name(self):
                return "mock"

        llm = MockVisionLLM()
        image = ImageContent(
            data="https://example.com/landscape.jpg",
            image_type=ImageType.URL
        )

        response = llm.describe_image(image, prompt="Describe this image")

        assert response.content == "A beautiful landscape"
        assert response.images_processed == 1

    def test_repr(self):
        """Test __repr__ method."""
        class MockVisionLLM(BaseVisionLLM):
            def __init__(self):
                super().__init__(model="test-model")

            def chat(self, messages, **kwargs):
                pass

            def validate_image_support(self):
                return True

            @property
            def provider_name(self):
                return "mock"

        llm = MockVisionLLM()
        repr_str = repr(llm)

        assert "MockVisionLLM" in repr_str
        assert "test-model" in repr_str
        assert "mock" in repr_str


class TestVisionMessageValidation:
    """Test VisionMessage validation edge cases."""

    def test_system_role_with_images(self):
        """Test system messages can include images (for context)."""
        image = ImageContent(data="https://example.com/image.png", image_type=ImageType.URL)
        message = VisionMessage(
            role="system",
            content="Analyze images in this context",
            images=[image]
        )

        assert message.role == "system"
        assert message.has_images()

    def test_assistant_role_with_images(self):
        """Test assistant messages can include images (multimodal output)."""
        message = VisionMessage(
            role="assistant",
            content="Here's what I see",
            images=[]
        )

        assert message.role == "assistant"

    def test_empty_text_with_images_is_valid(self):
        """Test that empty text with images is valid."""
        image = ImageContent(data="https://example.com/image.png", image_type=ImageType.URL)
        message = VisionMessage(
            role="user",
            content="",
            images=[image]
        )

        assert message.content == ""
        assert message.has_images()
