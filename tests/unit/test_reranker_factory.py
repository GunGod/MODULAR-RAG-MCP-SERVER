"""
Unit tests for Reranker Factory and BaseReranker interface (Task B5)

These tests verify that:
1. The BaseReranker interface is correctly defined
2. RerankerFactory can create instances based on settings
3. Factory routing logic works correctly
4. NoneReranker preserves original order (no reranking)
5. Error handling for unsupported providers

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import List

import pytest

from src.core.settings import (
    Settings,
    RerankConfig,
    LLMConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    RetrievalConfig,
    EvaluationConfig,
    ObservabilityConfig,
    DashboardConfig,
)
from src.libs.reranker.base_reranker import (
    BaseReranker,
    RerankCandidate,
    RerankResult,
)
from src.libs.reranker.none_reranker import NoneReranker
from src.libs.reranker.reranker_factory import RerankerFactory


class TestRerankCandidate:
    """Test the RerankCandidate dataclass."""

    def test_create_candidate(self):
        """Test creating a rerank candidate."""
        candidate = RerankCandidate(
            id="chunk-1",
            content="Sample text",
            score=0.95,
            metadata={"source": "doc1.pdf"}
        )
        assert candidate.id == "chunk-1"
        assert candidate.content == "Sample text"
        assert candidate.score == 0.95
        assert candidate.metadata == {"source": "doc1.pdf"}

    def test_create_candidate_without_metadata(self):
        """Test creating a candidate without metadata."""
        candidate = RerankCandidate(
            id="chunk-2",
            content="Another text",
            score=0.87
        )
        assert candidate.id == "chunk-2"
        assert candidate.metadata is None


class TestRerankResult:
    """Test the RerankResult dataclass."""

    def test_create_result(self):
        """Test creating a rerank result."""
        candidates = [
            RerankCandidate(id="1", content="A", score=0.9),
            RerankCandidate(id="2", content="B", score=0.8),
        ]
        result = RerankResult(
            candidates=candidates,
            scores=[0.9, 0.8],
            method="none"
        )
        assert len(result.candidates) == 2
        assert result.scores == [0.9, 0.8]
        assert result.method == "none"
        assert result.fallback_triggered is False

    def test_create_result_with_fallback(self):
        """Test creating result with fallback triggered."""
        result = RerankResult(
            candidates=[],
            scores=[],
            method="cross_encoder",
            fallback_triggered=True
        )
        assert result.fallback_triggered is True


class TestBaseReranker:
    """Test the BaseReranker abstract interface."""

    def test_base_reranker_cannot_be_instantiated(self):
        """Test that BaseReranker cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseReranker(top_k=10)

    def test_base_reranker_subclass_must_implement_rerank(self):
        """Test that subclasses must implement rerank method."""

        class IncompleteReranker(BaseReranker):
            def __init__(self):
                super().__init__(top_k=10)

            @property
            def provider_name(self) -> str:
                return "incomplete"

            # Missing rerank() method

        with pytest.raises(TypeError):
            IncompleteReranker()

    def test_base_reranker_initialization(self):
        """Test BaseReranker initialization with parameters."""
        reranker = NoneReranker(top_k=5, model="test-model")
        assert reranker.top_k == 5
        assert reranker.model == "test-model"

    def test_base_reranker_repr(self):
        """Test string representation of reranker."""
        reranker = NoneReranker(top_k=10)
        repr_str = repr(reranker)
        assert "NoneReranker" in repr_str
        assert "10" in repr_str
        assert "none" in repr_str


class TestNoneReranker:
    """Test the NoneReranker implementation."""

    def test_none_reranker_preserves_order(self):
        """Test that NoneReranker preserves original order."""
        reranker = NoneReranker(top_k=10)
        candidates = [
            RerankCandidate(id="1", content="First", score=0.9),
            RerankCandidate(id="2", content="Second", score=0.7),
            RerankCandidate(id="3", content="Third", score=0.5),
        ]

        result = reranker.rerank(query="test query", candidates=candidates)

        assert len(result.candidates) == 3
        assert result.candidates[0].id == "1"
        assert result.candidates[1].id == "2"
        assert result.candidates[2].id == "3"

    def test_none_reranker_preserves_scores(self):
        """Test that NoneReranker preserves original scores."""
        reranker = NoneReranker()
        candidates = [
            RerankCandidate(id="1", content="A", score=0.95),
            RerankCandidate(id="2", content="B", score=0.87),
            RerankCandidate(id="3", content="C", score=0.76),
        ]

        result = reranker.rerank(query="test", candidates=candidates)

        assert result.scores == [0.95, 0.87, 0.76]

    def test_none_reranker_respects_top_k(self):
        """Test that NoneReranker respects top_k parameter."""
        reranker = NoneReranker(top_k=2)
        candidates = [
            RerankCandidate(id=str(i), content=f"Text {i}", score=0.9 - i * 0.1)
            for i in range(5)
        ]

        result = reranker.rerank(query="test", candidates=candidates)

        assert len(result.candidates) == 2
        assert len(result.scores) == 2

    def test_none_reranker_empty_candidates_raises_error(self):
        """Test that empty candidates list raises ValueError."""
        reranker = NoneReranker()
        with pytest.raises(ValueError) as exc_info:
            reranker.rerank(query="test", candidates=[])
        assert "cannot be empty" in str(exc_info.value).lower()

    def test_none_reranker_method_name(self):
        """Test that NoneReranker returns correct method name."""
        reranker = NoneReranker()
        candidates = [RerankCandidate(id="1", content="Test", score=0.9)]

        result = reranker.rerank(query="test", candidates=candidates)

        assert result.method == "none"

    def test_none_reranker_fallback_not_triggered(self):
        """Test that fallback_triggered is False for NoneReranker."""
        reranker = NoneReranker()
        candidates = [RerankCandidate(id="1", content="Test", score=0.9)]

        result = reranker.rerank(query="test", candidates=candidates)

        assert result.fallback_triggered is False

    def test_none_reranker_tracks_call_count(self):
        """Test that NoneReranker tracks number of rerank() calls."""
        reranker = NoneReranker()
        assert reranker.get_call_count() == 0

        candidates = [RerankCandidate(id="1", content="Test", score=0.9)]
        reranker.rerank(query="test1", candidates=candidates)
        assert reranker.get_call_count() == 1

        reranker.rerank(query="test2", candidates=candidates)
        assert reranker.get_call_count() == 2

    def test_none_reranker_tracks_total_candidates_processed(self):
        """Test tracking total candidates processed."""
        reranker = NoneReranker()
        assert reranker.get_total_candidates_processed() == 0

        candidates1 = [RerankCandidate(id="1", content="A", score=0.9)]
        reranker.rerank(query="test1", candidates=candidates1)
        assert reranker.get_total_candidates_processed() == 1

        candidates2 = [
            RerankCandidate(id="1", content="A", score=0.9),
            RerankCandidate(id="2", content="B", score=0.8),
            RerankCandidate(id="3", content="C", score=0.7),
        ]
        reranker.rerank(query="test2", candidates=candidates2)
        assert reranker.get_total_candidates_processed() == 4

    def test_none_reranker_reset_call_count(self):
        """Test resetting the call counter."""
        reranker = NoneReranker()
        candidates = [RerankCandidate(id="1", content="Test", score=0.9)]
        reranker.rerank(query="test", candidates=candidates)
        assert reranker.get_call_count() == 1

        reranker.reset_call_count()
        assert reranker.get_call_count() == 0

    def test_none_reranker_provider_name(self):
        """Test NoneReranker provider_name property."""
        reranker = NoneReranker()
        assert reranker.provider_name == "none"

    def test_none_reranker_with_custom_top_k(self):
        """Test NoneReranker with custom top_k."""
        reranker = NoneReranker(top_k=3)
        assert reranker.top_k == 3

        candidates = [
            RerankCandidate(id=str(i), content=f"Text {i}", score=0.9)
            for i in range(10)
        ]
        result = reranker.rerank(query="test", candidates=candidates)

        assert len(result.candidates) == 3


class TestRerankerFactory:
    """Test the RerankerFactory class."""

    def test_factory_create_with_none_provider(self):
        """Test creating NoneReranker through factory."""
        settings = self._create_settings(backend="none")
        reranker = RerankerFactory.create(settings)

        assert isinstance(reranker, NoneReranker)
        assert reranker.provider_name == "none"

    def test_factory_create_with_custom_top_k(self):
        """Test factory passes top_k to reranker."""
        settings = self._create_settings(backend="none", top_k=5)
        reranker = RerankerFactory.create(settings)

        assert reranker.top_k == 5

    def test_factory_create_with_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises ValueError."""
        settings = self._create_settings(backend="unsupported_backend")

        with pytest.raises(ValueError) as exc_info:
            RerankerFactory.create(settings)
        assert "Unsupported reranker provider" in str(exc_info.value)
        assert "unsupported_backend" in str(exc_info.value)

    def test_factory_list_providers(self):
        """Test getting list of available providers."""
        providers = RerankerFactory.list_providers()
        assert isinstance(providers, list)
        assert "none" in providers

    def test_factory_register_provider(self):
        """Test registering a custom provider."""
        # Create a custom reranker class
        class CustomReranker(BaseReranker):
            def __init__(self, top_k=10, model=None, **kwargs):
                super().__init__(top_k, model, **kwargs)

            def rerank(self, query: str, candidates, **kwargs) -> RerankResult:
                # Simple reverse order reranking
                reversed_candidates = list(reversed(candidates))
                return RerankResult(
                    candidates=reversed_candidates[:self.top_k],
                    scores=[c.score for c in reversed_candidates[:self.top_k]],
                    method="custom"
                )

            @property
            def provider_name(self) -> str:
                return "custom"

        # Register the provider
        RerankerFactory.register_provider("custom", CustomReranker)

        # Verify it's in the list
        providers = RerankerFactory.list_providers()
        assert "custom" in providers

        # Verify it can be created
        settings = self._create_settings(backend="custom")
        reranker = RerankerFactory.create(settings)
        assert isinstance(reranker, CustomReranker)
        assert reranker.provider_name == "custom"

    def test_factory_register_non_reranker_class_raises_error(self):
        """Test that registering non-BaseReranker class raises TypeError."""
        class NotAReranker:
            pass

        with pytest.raises(TypeError) as exc_info:
            RerankerFactory.register_provider("invalid", NotAReranker)
        assert "BaseReranker" in str(exc_info.value)

    def test_factory_routing_logic(self):
        """Test that factory correctly routes to none provider."""
        settings1 = self._create_settings(backend="none", top_k=5)
        reranker1 = RerankerFactory.create(settings1)

        assert isinstance(reranker1, NoneReranker)
        assert reranker1.top_k == 5
        assert reranker1.provider_name == "none"

        # Verify it's a new instance each time
        settings2 = self._create_settings(backend="none")
        reranker2 = RerankerFactory.create(settings2)
        assert reranker1 is not reranker2

    def _create_settings(
        self,
        backend: str = "none",
        top_k: int = 10,
        model: str = None
    ) -> Settings:
        """Helper method to create test Settings object."""
        return Settings(
            llm=LLMConfig(provider="fake", model="fake-model"),
            embedding=EmbeddingConfig(provider="fake", model="fake-embedding"),
            vector_store=VectorStoreConfig(backend="chroma"),
            retrieval=RetrievalConfig(),
            rerank=RerankConfig(backend=backend, top_m=top_k, model=model),
            evaluation=EvaluationConfig(),
            observability=ObservabilityConfig(),
            dashboard=DashboardConfig(),
        )


class TestRerankerIntegration:
    """Integration tests for reranker usage patterns."""

    def test_full_rerank_workflow(self):
        """Test complete rerank workflow with NoneReranker."""
        settings = self._create_settings()
        reranker = RerankerFactory.create(settings)

        # Simulate retrieval results (already sorted by fusion)
        candidates = [
            RerankCandidate(id="1", content="Machine learning is...", score=0.92),
            RerankCandidate(id="2", content="Deep learning uses...", score=0.88),
            RerankCandidate(id="3", content="Neural networks have...", score=0.85),
            RerankCandidate(id="4", content="Data science involves...", score=0.79),
            RerankCandidate(id="5", content="Statistics provides...", score=0.75),
        ]

        # Rerank (should preserve order for NoneReranker)
        result = reranker.rerank(
            query="machine learning basics",
            candidates=candidates
        )

        # Verify order is preserved
        assert [c.id for c in result.candidates] == ["1", "2", "3", "4", "5"]
        # Verify scores are preserved
        assert result.scores == [0.92, 0.88, 0.85, 0.79, 0.75]
        # Verify method is "none"
        assert result.method == "none"

    def test_rerank_with_fewer_candidates_than_top_k(self):
        """Test reranking when candidates < top_k."""
        settings = self._create_settings(top_k=10)
        reranker = RerankerFactory.create(settings)

        candidates = [
            RerankCandidate(id="1", content="A", score=0.9),
            RerankCandidate(id="2", content="B", score=0.8),
        ]

        result = reranker.rerank(query="test", candidates=candidates)

        assert len(result.candidates) == 2  # All candidates returned

    def test_none_reranker_idempotency(self):
        """Test that NoneReranker is idempotent (same input → same output)."""
        settings = self._create_settings()
        reranker = RerankerFactory.create(settings)

        candidates = [
            RerankCandidate(id="1", content="A", score=0.9),
            RerankCandidate(id="2", content="B", score=0.8),
            RerankCandidate(id="3", content="C", score=0.7),
        ]

        # Rerank twice
        result1 = reranker.rerank(query="test", candidates=candidates)
        result2 = reranker.rerank(query="test", candidates=result1.candidates)

        # Results should be identical
        assert [c.id for c in result1.candidates] == [c.id for c in result2.candidates]
        assert result1.scores == result2.scores

    def _create_settings(self, backend: str = "none", top_k: int = 10) -> Settings:
        """Helper to create minimal settings for testing."""
        return Settings(
            llm=LLMConfig(provider="fake", model="fake-model"),
            embedding=EmbeddingConfig(provider="fake", model="fake-embedding"),
            vector_store=VectorStoreConfig(backend="chroma"),
            retrieval=RetrievalConfig(),
            rerank=RerankConfig(backend=backend, top_m=top_k),
            evaluation=EvaluationConfig(),
            observability=ObservabilityConfig(),
            dashboard=DashboardConfig(),
        )
