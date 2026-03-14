"""
Unit tests for Embedding Factory and BaseEmbedding interface (Task B2)

These tests verify that:
1. The BaseEmbedding interface is correctly defined
2. EmbeddingFactory can create instances based on settings
3. Factory routing logic works correctly
4. Unsupported providers raise appropriate errors
5. FakeEmbedding returns stable vectors for testing

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import List

import pytest

from src.core.settings import EmbeddingConfig, Settings
from src.libs.embedding.base_embedding import BaseEmbedding, EmbeddingResult
from src.libs.embedding.fake_embedding import FakeEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory


class TestEmbeddingResult:
    """Test the EmbeddingResult dataclass."""

    def test_create_result(self):
        """Test creating an embedding result."""
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        result = EmbeddingResult(
            embeddings=embeddings,
            model="test-model",
            provider="test",
            dimension=3
        )
        assert result.embeddings == embeddings
        assert result.model == "test-model"
        assert result.provider == "test"
        assert result.dimension == 3

    def test_create_result_with_usage(self):
        """Test creating result with token usage."""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2]],
            model="test-model",
            provider="test",
            dimension=2,
            usage={"total_tokens": 10}
        )
        assert result.usage == {"total_tokens": 10}

    def test_embeddings_list_count_must_match_texts_count(self):
        """Test that embeddings list length should match input texts count."""
        # This is implicitly tested in other test cases
        # The contract is: len(embeddings) == len(texts)
        pass


class TestBaseEmbedding:
    """Test the BaseEmbedding abstract interface."""

    def test_base_embedding_cannot_be_instantiated(self):
        """Test that BaseEmbedding cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEmbedding(model="test", dimension=10)

    def test_base_embedding_subclass_must_implement_embed(self):
        """Test that subclasses must implement embed method."""

        class IncompleteEmbedding(BaseEmbedding):
            def __init__(self):
                super().__init__(model="incomplete", dimension=10)

            @property
            def provider_name(self) -> str:
                return "incomplete"

            # Missing embed() method

        with pytest.raises(TypeError):
            IncompleteEmbedding()

    def test_base_embedding_subclass_must_implement_provider_name(self):
        """Test that subclasses must implement provider_name property."""

        class IncompleteEmbedding(BaseEmbedding):
            def __init__(self):
                super().__init__(model="incomplete", dimension=10)

            def embed(self, texts, **kwargs):
                return EmbeddingResult(
                    embeddings=[[0.1]],
                    model=self.model,
                    provider="incomplete",
                    dimension=1
                )

            # Missing provider_name property

        with pytest.raises(TypeError):
            IncompleteEmbedding()

    def test_base_embedding_initialization(self):
        """Test BaseEmbedding initialization with parameters."""
        embedding = FakeEmbedding(model="test-model", dimension=768)
        assert embedding.model == "test-model"
        assert embedding.dimension == 768
        assert embedding.batch_size == 32

    def test_base_embedding_repr(self):
        """Test string representation of embedding."""
        embedding = FakeEmbedding(model="test-model", dimension=1536)
        repr_str = repr(embedding)
        assert "FakeEmbedding" in repr_str
        assert "test-model" in repr_str
        assert "1536" in repr_str
        assert "fake" in repr_str


class TestFakeEmbedding:
    """Test the FakeEmbedding implementation."""

    def test_fake_embedding_returns_correct_dimension(self):
        """Test that FakeEmbedding returns vectors with correct dimension."""
        embedding = FakeEmbedding(dimension=100)
        texts = ["Hello", "World"]
        result = embedding.embed(texts)

        assert len(result.embeddings) == 2
        for vector in result.embeddings:
            assert len(vector) == 100

    def test_fake_embedding_same_text_same_vector(self):
        """Test that same text produces same vector (stability)."""
        embedding = FakeEmbedding(dimension=50)
        texts = ["Test text"]

        result1 = embedding.embed(texts)
        result2 = embedding.embed(texts)

        assert result1.embeddings[0] == result2.embeddings[0]

    def test_fake_embedding_different_text_different_vector(self):
        """Test that different texts produce different vectors."""
        embedding = FakeEmbedding(dimension=50)
        texts = ["Text one", "Text two"]
        result = embedding.embed(texts)

        assert result.embeddings[0] != result.embeddings[1]

    def test_fake_embedding_empty_texts_raises_error(self):
        """Test that empty texts list raises ValueError."""
        embedding = FakeEmbedding()
        with pytest.raises(ValueError) as exc_info:
            embedding.embed([])
        assert "cannot be empty" in str(exc_info.value).lower()

    def test_fake_embedding_tracks_call_count(self):
        """Test that FakeEmbedding tracks number of embed() calls."""
        embedding = FakeEmbedding()
        assert embedding.get_call_count() == 0

        embedding.embed(["First"])
        assert embedding.get_call_count() == 1

        embedding.embed(["Second"])
        assert embedding.get_call_count() == 2

    def test_fake_embedding_tracks_total_texts(self):
        """Test tracking total texts processed."""
        embedding = FakeEmbedding()
        assert embedding.get_total_texts_processed() == 0

        embedding.embed(["A", "B"])
        assert embedding.get_total_texts_processed() == 2

        embedding.embed(["C", "D", "E"])
        assert embedding.get_total_texts_processed() == 5

    def test_fake_embedding_reset_call_count(self):
        """Test resetting the call counter."""
        embedding = FakeEmbedding()
        embedding.embed(["Test"])
        assert embedding.get_call_count() == 1

        embedding.reset_call_count()
        assert embedding.get_call_count() == 0

    def test_fake_embedding_provider_name(self):
        """Test FakeEmbedding provider_name property."""
        embedding = FakeEmbedding()
        assert embedding.provider_name == "fake"

    def test_fake_embedding_returns_embedding_result(self):
        """Test that embed() returns EmbeddingResult with correct fields."""
        embedding = FakeEmbedding(dimension=10)
        texts = ["test"]
        result = embedding.embed(texts)

        assert isinstance(result, EmbeddingResult)
        assert isinstance(result.embeddings, list)
        assert len(result.embeddings) == 1
        assert result.model == "fake-embedding"
        assert result.provider == "fake"
        assert result.dimension == 10
        assert result.usage is not None


class TestEmbeddingFactory:
    """Test the EmbeddingFactory class."""

    def test_factory_create_with_fake_provider(self):
        """Test creating FakeEmbedding through factory."""
        settings = self._create_settings(provider="fake")
        embedding = EmbeddingFactory.create(settings)

        assert isinstance(embedding, FakeEmbedding)
        assert embedding.model == "fake-embedding"
        assert embedding.provider_name == "fake"

    def test_factory_create_with_custom_dimension(self):
        """Test factory passes dimension to embedding."""
        settings = self._create_settings(provider="fake", dimension=768)
        embedding = EmbeddingFactory.create(settings)

        assert embedding.dimension == 768

    def test_factory_create_with_custom_batch_size(self):
        """Test factory passes batch_size to embedding."""
        settings = self._create_settings(provider="fake", batch_size=64)
        embedding = EmbeddingFactory.create(settings)

        assert embedding.batch_size == 64

    def test_factory_create_with_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises ValueError."""
        settings = self._create_settings(provider="unsupported_provider")

        with pytest.raises(ValueError) as exc_info:
            EmbeddingFactory.create(settings)
        assert "Unsupported embedding provider" in str(exc_info.value)
        assert "unsupported_provider" in str(exc_info.value)

    def test_factory_list_providers(self):
        """Test getting list of available providers."""
        providers = EmbeddingFactory.list_providers()
        assert isinstance(providers, list)
        assert "fake" in providers

    def test_factory_register_provider(self):
        """Test registering a custom provider."""
        # Create a custom embedding class
        class CustomEmbedding(BaseEmbedding):
            def __init__(self, model="custom", dimension=100, **kwargs):
                super().__init__(model, dimension, **kwargs)

            def embed(self, texts, **kwargs):
                vectors = [[0.0] * self.dimension for _ in texts]
                return EmbeddingResult(
                    embeddings=vectors,
                    model=self.model,
                    provider="custom",
                    dimension=self.dimension
                )

            @property
            def provider_name(self) -> str:
                return "custom"

        # Register the provider
        EmbeddingFactory.register_provider("custom", CustomEmbedding)

        # Verify it's in the list
        providers = EmbeddingFactory.list_providers()
        assert "custom" in providers

        # Verify it can be created
        settings = self._create_settings(provider="custom")
        embedding = EmbeddingFactory.create(settings)
        assert isinstance(embedding, CustomEmbedding)
        assert embedding.provider_name == "custom"

    def test_factory_register_non_embedding_class_raises_error(self):
        """Test that registering non-BaseEmbedding class raises TypeError."""
        class NotAnEmbedding:
            pass

        with pytest.raises(TypeError) as exc_info:
            EmbeddingFactory.register_provider("invalid", NotAnEmbedding)
        assert "BaseEmbedding" in str(exc_info.value)

    def test_factory_routing_logic(self):
        """Test that factory correctly routes to different providers."""
        # Test with fake provider
        settings1 = self._create_settings(provider="fake", model="model-1")
        embedding1 = EmbeddingFactory.create(settings1)
        assert isinstance(embedding1, FakeEmbedding)
        assert embedding1.model == "model-1"

        # Test with different model but same provider
        settings2 = self._create_settings(provider="fake", model="model-2")
        embedding2 = EmbeddingFactory.create(settings2)
        assert isinstance(embedding2, FakeEmbedding)
        assert embedding2.model == "model-2"

        # Verify they are different instances
        assert embedding1 is not embedding2

    def _create_settings(
        self,
        provider: str = "fake",
        model: str = "fake-embedding",
        dimension: int = 1536,
        batch_size: int = 32
    ) -> Settings:
        """Helper method to create test Settings object."""
        return Settings(
            llm=self._create_llm_config(),
            embedding=EmbeddingConfig(
                provider=provider,
                model=model,
                dimension=dimension,
                batch_size=batch_size
            ),
            vector_store=self._create_vector_store_config(),
            retrieval=self._create_retrieval_config(),
            rerank=self._create_rerank_config(),
            evaluation=self._create_evaluation_config(),
            observability=self._create_observability_config(),
            dashboard=self._create_dashboard_config(),
        )

    def _create_llm_config(self):
        from src.core.settings import LLMConfig
        return LLMConfig(provider="fake", model="fake-model")

    def _create_vector_store_config(self):
        from src.core.settings import VectorStoreConfig
        return VectorStoreConfig(backend="chroma")

    def _create_retrieval_config(self):
        from src.core.settings import RetrievalConfig
        return RetrievalConfig()

    def _create_rerank_config(self):
        from src.core.settings import RerankConfig
        return RerankConfig()

    def _create_evaluation_config(self):
        from src.core.settings import EvaluationConfig
        return EvaluationConfig()

    def _create_observability_config(self):
        from src.core.settings import ObservabilityConfig
        return ObservabilityConfig()

    def _create_dashboard_config(self):
        from src.core.settings import DashboardConfig
        return DashboardConfig()


class TestEmbeddingIntegration:
    """Integration tests for embedding usage patterns."""

    def test_batch_embedding(self):
        """Test embedding multiple texts at once."""
        settings = self._create_minimal_settings()
        embedding = EmbeddingFactory.create(settings)

        texts = ["Hello world", "Test text", "Another example"]
        result = embedding.embed(texts)

        assert len(result.embeddings) == 3
        assert result.dimension == 1536
        for vector in result.embeddings:
            assert len(vector) == 1536

    def test_embedding_result_contains_metadata(self):
        """Test that embedding result contains all required metadata."""
        settings = self._create_minimal_settings()
        embedding = EmbeddingFactory.create(settings)

        texts = ["Test"]
        result = embedding.embed(texts)

        assert result.embeddings is not None
        assert result.model is not None
        assert result.provider is not None
        assert result.dimension is not None
        assert result.provider == "fake"

    def test_embedding_vector_values_in_valid_range(self):
        """Test that fake embedding vectors are in valid range."""
        # Fake embeddings should be between -1 and 1
        settings = self._create_minimal_settings()
        embedding = EmbeddingFactory.create(settings)

        texts = ["Test text"]
        result = embedding.embed(texts)

        for vector in result.embeddings:
            for value in vector:
                assert -1.0 <= value <= 1.0, f"Value {value} is outside [-1, 1] range"

    def test_embedding_consistency(self):
        """Test that embeddings are consistent for the same text."""
        settings = self._create_minimal_settings()
        embedding = EmbeddingFactory.create(settings)

        text = "Consistency test"
        result1 = embedding.embed([text])
        result2 = embedding.embed([text])

        # Same text should produce same vector
        assert result1.embeddings[0] == result2.embeddings[0]

    def _create_minimal_settings(self) -> Settings:
        """Helper to create minimal settings for testing."""
        from src.core.settings import (
            LLMConfig, VectorStoreConfig, RetrievalConfig,
            RerankConfig, EvaluationConfig, ObservabilityConfig, DashboardConfig
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
