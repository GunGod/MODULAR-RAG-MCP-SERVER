"""
Unit tests for Splitter Factory and BaseSplitter interface (Task B3)

These tests verify that:
1. The BaseSplitter interface is correctly defined
2. SplitterFactory can create instances based on settings
3. Factory routing logic works correctly
4. Unsupported providers raise appropriate errors
5. FakeSplitter returns predictable chunks for testing

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import List

import pytest

from src.core.settings import Settings
from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.fake_splitter import FakeSplitter
from src.libs.splitter.splitter_factory import SplitterFactory


class TestBaseSplitter:
    """Test the BaseSplitter abstract interface."""

    def test_base_splitter_cannot_be_instantiated(self):
        """Test that BaseSplitter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSplitter(chunk_size=1000)

    def test_base_splitter_subclass_must_implement_split_text(self):
        """Test that subclasses must implement split_text method."""

        class IncompleteSplitter(BaseSplitter):
            def __init__(self):
                super().__init__(chunk_size=1000, chunk_overlap=200)

            @property
            def provider_name(self) -> str:
                return "incomplete"

            # Missing split_text() method

        with pytest.raises(TypeError):
            IncompleteSplitter()

    def test_base_splitter_subclass_must_implement_provider_name(self):
        """Test that subclasses must implement provider_name property."""

        class IncompleteSplitter(BaseSplitter):
            def __init__(self):
                super().__init__(chunk_size=1000, chunk_overlap=200)

            def split_text(self, text: str, **kwargs) -> List[str]:
                return [text]

            # Missing provider_name property

        with pytest.raises(TypeError):
            IncompleteSplitter()

    def test_base_splitter_initialization(self):
        """Test BaseSplitter initialization with parameters."""
        splitter = FakeSplitter(chunk_size=500, chunk_overlap=100)
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 100
        assert splitter.separators == ["\n\n", "\n", " ", ""]  # default

    def test_base_splitter_custom_separators(self):
        """Test BaseSplitter with custom separators."""
        custom_separators = ["\n\n\n", "\n\n", "\n", ".", " ", ""]
        splitter = FakeSplitter(separators=custom_separators)
        assert splitter.separators == custom_separators

    def test_base_splitter_repr(self):
        """Test string representation of splitter."""
        splitter = FakeSplitter(chunk_size=1000, chunk_overlap=200)
        repr_str = repr(splitter)
        assert "FakeSplitter" in repr_str
        assert "1000" in repr_str
        assert "200" in repr_str
        assert "fake" in repr_str


class TestFakeSplitter:
    """Test the FakeSplitter implementation."""

    def test_fake_splitter_returns_correct_number_of_chunks(self):
        """Test that FakeSplitter returns correct number of chunks."""
        splitter = FakeSplitter(chunk_size=100)
        text = "a" * 250  # 250 characters
        chunks = splitter.split_text(text)

        # Should split into 3 chunks: 100, 100, 50
        assert len(chunks) == 3

    def test_fake_splitter_respects_chunk_size(self):
        """Test that FakeSplitter respects chunk_size limit."""
        splitter = FakeSplitter(chunk_size=50)
        text = "x" * 150
        chunks = splitter.split_text(text)

        for chunk in chunks:
            assert len(chunk) <= 50

    def test_fake_splitter_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        splitter = FakeSplitter()
        with pytest.raises(ValueError) as exc_info:
            splitter.split_text("")
        assert "cannot be empty" in str(exc_info.value).lower()

    def test_fake_splitter_whitespace_only_text_raises_error(self):
        """Test that whitespace-only text raises ValueError."""
        splitter = FakeSplitter()
        with pytest.raises(ValueError) as exc_info:
            splitter.split_text("   \n\t  ")
        assert "cannot be empty" in str(exc_info.value).lower()

    def test_fake_splitter_tracks_call_count(self):
        """Test that FakeSplitter tracks number of split_text() calls."""
        splitter = FakeSplitter()
        assert splitter.get_call_count() == 0

        splitter.split_text("First text")
        assert splitter.get_call_count() == 1

        splitter.split_text("Second text")
        assert splitter.get_call_count() == 2

    def test_fake_splitter_tracks_total_texts_processed(self):
        """Test tracking total texts processed."""
        splitter = FakeSplitter()
        assert splitter.get_total_texts_processed() == 0

        splitter.split_text("Text 1")
        assert splitter.get_total_texts_processed() == 1

        splitter.split_text("Text 2")
        splitter.split_text("Text 3")
        assert splitter.get_total_texts_processed() == 3

    def test_fake_splitter_tracks_total_chunks_generated(self):
        """Test tracking total chunks generated."""
        splitter = FakeSplitter(chunk_size=50)
        assert splitter.get_total_chunks_generated() == 0

        splitter.split_text("a" * 150)  # 3 chunks
        assert splitter.get_total_chunks_generated() == 3

        splitter.split_text("b" * 100)  # 2 chunks
        assert splitter.get_total_chunks_generated() == 5

    def test_fake_splitter_reset_call_count(self):
        """Test resetting the call counter."""
        splitter = FakeSplitter()
        splitter.split_text("Test")
        assert splitter.get_call_count() == 1

        splitter.reset_call_count()
        assert splitter.get_call_count() == 0

    def test_fake_splitter_provider_name(self):
        """Test FakeSplitter provider_name property."""
        splitter = FakeSplitter()
        assert splitter.provider_name == "fake"

    def test_fake_splitter_with_chunk_overlap_parameter(self):
        """Test FakeSplitter accepts chunk_overlap parameter (though doesn't use it)."""
        # FakeSplitter accepts chunk_overlap but doesn't implement overlap logic
        splitter = FakeSplitter(chunk_size=100, chunk_overlap=50)
        text = "a" * 250
        chunks = splitter.split_text(text)

        # Should still split without overlap (simplified for testing)
        assert len(chunks) == 3


class TestSplitterFactory:
    """Test the SplitterFactory class."""

    def test_factory_create_with_fake_provider(self):
        """Test creating FakeSplitter through factory."""
        settings = self._create_settings()
        splitter = SplitterFactory.create(settings)

        assert isinstance(splitter, FakeSplitter)
        assert splitter.provider_name == "fake"

    def test_factory_create_with_default_chunk_size(self):
        """Test factory uses default chunk_size."""
        settings = self._create_settings()
        splitter = SplitterFactory.create(settings)

        assert splitter.chunk_size == 1000  # Default in factory
        assert splitter.chunk_overlap == 200  # Default in factory

    def test_factory_list_providers(self):
        """Test getting list of available providers."""
        providers = SplitterFactory.list_providers()
        assert isinstance(providers, list)
        assert "fake" in providers

    def test_factory_register_provider(self):
        """Test registering a custom provider."""
        # Create a custom splitter class
        class CustomSplitter(BaseSplitter):
            def __init__(self, chunk_size=1000, chunk_overlap=200, **kwargs):
                super().__init__(chunk_size, chunk_overlap, **kwargs)

            def split_text(self, text: str, **kwargs) -> List[str]:
                # Simply return the whole text as one chunk
                return [text] if text else []

            @property
            def provider_name(self) -> str:
                return "custom"

        # Register the provider
        SplitterFactory.register_provider("custom", CustomSplitter)

        # Verify it's in the list
        providers = SplitterFactory.list_providers()
        assert "custom" in providers

    def test_factory_register_non_splitter_class_raises_error(self):
        """Test that registering non-BaseSplitter class raises TypeError."""
        class NotASplitter:
            pass

        with pytest.raises(TypeError) as exc_info:
            SplitterFactory.register_provider("invalid", NotASplitter)
        assert "BaseSplitter" in str(exc_info.value)

    def test_factory_routing_logic(self):
        """Test that factory correctly routes to fake provider."""
        settings1 = self._create_settings()
        splitter1 = SplitterFactory.create(settings1)

        assert isinstance(splitter1, FakeSplitter)
        assert splitter1.provider_name == "fake"

        # Verify it's a new instance each time
        settings2 = self._create_settings()
        splitter2 = SplitterFactory.create(settings2)
        assert splitter1 is not splitter2

    def _create_settings(self) -> Settings:
        """Helper method to create test Settings object."""
        from src.core.settings import (
            LLMConfig,
            EmbeddingConfig,
            VectorStoreConfig,
            RetrievalConfig,
            RerankConfig,
            EvaluationConfig,
            ObservabilityConfig,
            DashboardConfig,
        )
        return Settings(
            llm=LLMConfig(provider="fake", model="fake-model"),
            embedding=EmbeddingConfig(provider="fake", model="fake-embedding"),
            vector_store=VectorStoreConfig(backend="chroma"),
            retrieval=RetrievalConfig(),
            rerank=RerankConfig(),
            evaluation=EvaluationConfig(),
            observability=ObservabilityConfig(),
            dashboard=DashboardConfig(),
        )


class TestSplitterIntegration:
    """Integration tests for splitter usage patterns."""

    def test_split_short_text(self):
        """Test splitting text shorter than chunk_size."""
        settings = self._create_settings()
        splitter = SplitterFactory.create(settings)

        text = "This is a short text."
        chunks = splitter.split_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_long_text(self):
        """Test splitting text longer than chunk_size."""
        settings = self._create_settings()
        splitter = SplitterFactory.create(settings)

        text = "word " * 500  # ~2500 characters
        chunks = splitter.split_text(text)

        assert len(chunks) > 1
        # Verify total content is preserved
        combined = "".join(chunks)
        assert combined == text

    def test_split_text_with_unicode(self):
        """Test splitting text with unicode characters."""
        settings = self._create_settings()
        splitter = SplitterFactory.create(settings)

        text = "中文文本 🚀 Hello 世界 " * 50
        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        # Verify no data loss
        combined = "".join(chunks)
        assert combined == text

    def test_split_preserves_content(self):
        """Test that splitting preserves all original content."""
        settings = self._create_settings()
        splitter = SplitterFactory.create(settings)

        original = "The quick brown fox jumps over the lazy dog. " * 100
        chunks = splitter.split_text(original)

        reconstructed = "".join(chunks)
        assert reconstructed == original

    def _create_settings(self) -> Settings:
        """Helper to create minimal settings for testing."""
        from src.core.settings import (
            LLMConfig,
            EmbeddingConfig,
            VectorStoreConfig,
            RetrievalConfig,
            RerankConfig,
            EvaluationConfig,
            ObservabilityConfig,
            DashboardConfig,
        )
        return Settings(
            llm=LLMConfig(provider="fake", model="fake-model"),
            embedding=EmbeddingConfig(provider="fake", model="fake-embedding"),
            vector_store=VectorStoreConfig(backend="chroma"),
            retrieval=RetrievalConfig(),
            rerank=RerankConfig(),
            evaluation=EvaluationConfig(),
            observability=ObservabilityConfig(),
            dashboard=DashboardConfig(),
        )
