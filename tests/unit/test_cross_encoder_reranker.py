"""
Unit tests for Cross-Encoder Reranker provider (Task B7.8)

These tests verify that:
1. RerankerFactory can create Cross-Encoder Reranker provider
2. CrossEncoderReranker scores candidates correctly
3. CrossEncoderReranker handles timeout and errors gracefully
4. CrossEncoderReranker provides fallback on failure
5. Custom scorer functions can be injected for testing
6. Statistics tracking works correctly

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import List
import time

import pytest

from src.core.settings import LLMConfig, RerankConfig, Settings
from src.libs.reranker.base_reranker import RerankCandidate, RerankResult
from src.libs.reranker.reranker_factory import RerankerFactory
from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker


# ===== Test Fixtures =====

def create_reranker_config(
    backend: str = "cross_encoder",
    top_m: int = 10,
    model: str = "ms-marco-MiniLM-L-6",
    **kwargs
) -> RerankConfig:
    """Helper to create RerankConfig for testing."""
    return RerankConfig(
        backend=backend,
        top_m=top_m,
        model=model,
        **kwargs
    )


def create_mock_scorer(scores: dict) -> callable:
    """Helper to create a mock scorer that returns predefined scores."""
    def scorer(query: str, content: str) -> float:
        # Simple scoring based on content keywords
        if "machine learning" in content.lower():
            return 0.95
        elif "deep learning" in content.lower():
            return 0.85
        elif "neural network" in content.lower():
            return 0.75
        elif "ai" in content.lower():
            return 0.65
        else:
            return 0.50
    return scorer


def create_slow_scorer(delay: float) -> callable:
    """Helper to create a scorer that simulates slow scoring."""
    def scorer(query: str, content: str) -> float:
        time.sleep(delay)
        return 0.8
    return scorer


def create_failing_scorer() -> callable:
    """Helper to create a scorer that always fails."""
    def scorer(query: str, content: str) -> float:
        raise RuntimeError("Scorer failed!")
    return scorer


# ===== Cross-Encoder Reranker Tests =====

class TestCrossEncoderReranker:
    """Test Cross-Encoder Reranker provider."""

    def test_factory_creates_cross_encoder_reranker(self):
        """Test that RerankerFactory can create Cross-Encoder Reranker (Acceptance Criteria 1)."""
        config = create_reranker_config(
            backend="cross_encoder",
            top_m=5,
            model="ms-marco-MiniLM-L-6"
        )
        settings = Settings(
            llm=self._create_llm_config(),
            embedding=self._create_embedding_config(),
            vector_store=self._create_vector_store_config(),
            retrieval=self._create_retrieval_config(),
            rerank=config,
            evaluation=self._create_evaluation_config(),
            observability=self._create_observability_config(),
            dashboard=self._create_dashboard_config(),
        )

        reranker = RerankerFactory.create(settings)

        assert isinstance(reranker, CrossEncoderReranker)
        assert reranker.provider_name == "cross_encoder"
        assert reranker.top_k == 5
        assert reranker.model == "ms-marco-MiniLM-L-6"

    def test_cross_encoder_reranker_initialization(self):
        """Test CrossEncoderReranker initialization with parameters."""
        scorer = create_mock_scorer({})

        reranker = CrossEncoderReranker(
            top_k=5,
            model="ms-marco-MiniLM-L-6",
            scorer=scorer,
            timeout=10.0
        )

        assert reranker.top_k == 5
        assert reranker.model == "ms-marco-MiniLM-L-6"
        assert reranker.scorer == scorer
        assert reranker.timeout == 10.0
        assert reranker.provider_name == "cross_encoder"

    def test_cross_encoder_reranker_default_scorer(self):
        """Test that CrossEncoderReranker uses default scorer when none provided."""
        reranker = CrossEncoderReranker(top_k=10)

        assert reranker.scorer is None
        assert reranker._default_scorer("test", "test content") >= 0.0

    def test_cross_encoder_reranker_validates_empty_candidates(self):
        """Test that CrossEncoderReranker validates candidates list is not empty."""
        reranker = CrossEncoderReranker(top_k=10)
        query = "test query"
        candidates = []

        with pytest.raises(ValueError) as exc_info:
            reranker.rerank(query=query, candidates=candidates)

        assert "cannot be empty" in str(exc_info.value).lower()
        assert "cross_encoder" in str(exc_info.value).lower()

    def test_cross_encoder_reranker_successful_reranking(self):
        """Test successful reranking with cross-encoder scoring."""
        scorer = create_mock_scorer({})
        reranker = CrossEncoderReranker(top_k=10, scorer=scorer)

        # Create candidates with varying relevance
        candidates = [
            RerankCandidate(id="chunk1", content="Introduction to AI concepts", score=0.6),
            RerankCandidate(id="chunk2", content="Deep learning and neural networks", score=0.7),
            RerankCandidate(id="chunk3", content="Machine learning basics and algorithms", score=0.8),
        ]

        # Rerank
        result = reranker.rerank(
            query="machine learning",
            candidates=candidates
        )

        # Verify result
        assert isinstance(result, RerankResult)
        assert len(result.candidates) == 3
        assert result.method == "cross_encoder"
        assert result.fallback_triggered is False

        # Verify ranking (chunk3 with "machine learning" should be first)
        assert result.candidates[0].id == "chunk3"
        assert result.candidates[0].score == 0.95  # Mock scorer returns 0.95
        assert result.candidates[0].score > result.candidates[1].score

    def test_cross_encoder_reranker_trims_to_top_k(self):
        """Test that reranker respects top_k parameter."""
        scorer = create_mock_scorer({})
        reranker = CrossEncoderReranker(top_k=3, scorer=scorer)

        # Create 5 candidates
        candidates = [
            RerankCandidate(id=f"chunk{i}", content=f"Content {i}", score=0.5)
            for i in range(5)
        ]

        # Rerank
        result = reranker.rerank(
            query="test query",
            candidates=candidates
        )

        # Should only return 3 candidates
        assert len(result.candidates) == 3
        assert len(result.scores) == 3

    def test_cross_encoder_reranker_scores_are_descending(self):
        """Test that reranked candidates are sorted by score descending."""
        scorer = create_mock_scorer({})
        reranker = CrossEncoderReranker(top_k=10, scorer=scorer)

        candidates = [
            RerankCandidate(id="chunk1", content="AI and machine learning", score=0.6),
            RerankCandidate(id="chunk2", content="Deep learning", score=0.7),
            RerankCandidate(id="chunk3", content="Machine learning algorithms", score=0.8),
        ]

        result = reranker.rerank(query="machine learning", candidates=candidates)

        # Verify descending order
        scores = [c.score for c in result.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_cross_encoder_reranker_fallback_on_timeout(self):
        """Test that CrossEncoderReranker handles timeout gracefully (Acceptance Criteria 2)."""
        slow_scorer = create_slow_scorer(delay=0.2)  # 200ms delay
        reranker = CrossEncoderReranker(
            top_k=10,
            scorer=slow_scorer,
            timeout=0.1  # 100ms timeout
        )

        candidates = [
            RerankCandidate(id="chunk1", content="Content 1", score=0.8),
            RerankCandidate(id="chunk2", content="Content 2", score=0.7),
        ]

        # Rerank (should timeout but still return results)
        result = reranker.rerank(
            query="test",
            candidates=candidates
        )

        # Should return candidates with original scores due to timeout
        assert len(result.candidates) == 2
        # Timeout candidates keep their original scores
        assert all(c.score in [0.7, 0.8] for c in result.candidates)

    def test_cross_encoder_reranker_fallback_on_scorer_error(self):
        """Test that CrossEncoderReranker handles scorer errors gracefully (Acceptance Criteria 2)."""
        failing_scorer = create_failing_scorer()
        reranker = CrossEncoderReranker(top_k=10, scorer=failing_scorer)

        candidates = [
            RerankCandidate(id="chunk1", content="Content 1", score=0.8),
            RerankCandidate(id="chunk2", content="Content 2", score=0.7),
        ]

        # Rerank (scorer will fail)
        result = reranker.rerank(
            query="test",
            candidates=candidates
        )

        # Should return candidates with original scores due to error
        assert len(result.candidates) == 2
        assert result.candidates[0].score == 0.8
        assert result.candidates[1].score == 0.7

    def test_cross_encoder_reranker_custom_scorer_injection(self):
        """Test that custom scorer can be injected for testing."""
        # Custom scorer that returns fixed scores
        def custom_scorer(query: str, content: str) -> float:
            return 0.99

        reranker = CrossEncoderReranker(
            top_k=10,
            scorer=custom_scorer
        )

        candidates = [
            RerankCandidate(id="chunk1", content="Any content", score=0.5),
        ]

        result = reranker.rerank(query="test", candidates=candidates)

        # All candidates should have score 0.99 from custom scorer
        assert result.candidates[0].score == 0.99

    def test_cross_encoder_reranker_statistics_tracking(self):
        """Test that CrossEncoderReranker tracks statistics correctly."""
        scorer = create_mock_scorer({})
        reranker = CrossEncoderReranker(top_k=5, scorer=scorer)

        # Initial statistics
        stats = reranker.get_statistics()
        assert stats["total_rerank_calls"] == 0
        assert stats["total_candidates_scored"] == 0
        assert stats["total_fallback_count"] == 0

        # Perform reranking
        candidates = [
            RerankCandidate(id="chunk1", content="Content 1", score=0.5),
            RerankCandidate(id="chunk2", content="Content 2", score=0.6),
        ]
        reranker.rerank(query="test", candidates=candidates)

        # Check statistics after first call
        stats = reranker.get_statistics()
        assert stats["total_rerank_calls"] == 1
        assert stats["total_candidates_scored"] == 2
        assert stats["total_fallback_count"] == 0

        # Reset statistics
        reranker.reset_statistics()
        stats = reranker.get_statistics()
        assert stats["total_rerank_calls"] == 0
        assert stats["total_candidates_scored"] == 0

    def test_cross_encoder_reranker_preserves_metadata(self):
        """Test that CrossEncoderReranker preserves candidate metadata."""
        scorer = create_mock_scorer({})
        reranker = CrossEncoderReranker(top_k=10, scorer=scorer)

        candidates = [
            RerankCandidate(
                id="chunk1",
                content="Content",
                score=0.5,
                metadata={"source": "doc.pdf", "page": 1}
            ),
        ]

        result = reranker.rerank(query="test", candidates=candidates)

        # Metadata should be preserved
        assert result.candidates[0].metadata == {"source": "doc.pdf", "page": 1}

    def test_cross_encoder_reranker_deterministic_scoring(self):
        """Test that scoring is deterministic with same input."""
        scorer = create_mock_scorer({})
        reranker = CrossEncoderReranker(top_k=10, scorer=scorer)

        candidates = [
            RerankCandidate(id="chunk1", content="Machine learning", score=0.5),
            RerankCandidate(id="chunk2", content="Deep learning", score=0.6),
        ]

        # Rerank twice
        result1 = reranker.rerank(query="test", candidates=candidates)
        result2 = reranker.rerank(query="test", candidates=candidates)

        # Results should be identical
        assert len(result1.candidates) == len(result2.candidates)
        for c1, c2 in zip(result1.candidates, result2.candidates):
            assert c1.id == c2.id
            assert c1.score == c2.score

    def test_cross_encoder_reranker_string_representation(self):
        """Test that CrossEncoderReranker has correct string representation."""
        reranker = CrossEncoderReranker(top_k=5, model="test-model")

        repr_str = repr(reranker)
        assert "CrossEncoderReranker" in repr_str
        assert "top_k=5" in repr_str
        assert "cross_encoder" in repr_str


# ===== Helper Methods for Creating Settings =====

@staticmethod
def _create_llm_config():
    return LLMConfig(provider="fake", model="fake-model")

@staticmethod
def _create_embedding_config():
    from src.core.settings import EmbeddingConfig
    return EmbeddingConfig(provider="fake", model="fake-model", dimension=768)

@staticmethod
def _create_vector_store_config():
    from src.core.settings import VectorStoreConfig
    return VectorStoreConfig(backend="chroma")

@staticmethod
def _create_retrieval_config():
    from src.core.settings import RetrievalConfig
    return RetrievalConfig()

@staticmethod
def _create_rerank_config():
    return RerankConfig(backend="none")

@staticmethod
def _create_evaluation_config():
    from src.core.settings import EvaluationConfig
    return EvaluationConfig()

@staticmethod
def _create_observability_config():
    from src.core.settings import ObservabilityConfig
    return ObservabilityConfig()

@staticmethod
def _create_dashboard_config():
    from src.core.settings import DashboardConfig
    return DashboardConfig()


@staticmethod
def _create_settings():
    """Create a complete Settings object."""
    return Settings(
        llm=TestCrossEncoderReranker._create_llm_config(),
        embedding=TestCrossEncoderReranker._create_embedding_config(),
        vector_store=TestCrossEncoderReranker._create_vector_store_config(),
        retrieval=TestCrossEncoderReranker._create_retrieval_config(),
        rerank=TestCrossEncoderReranker._create_rerank_config(),
        evaluation=TestCrossEncoderReranker._create_evaluation_config(),
        observability=TestCrossEncoderReranker._create_observability_config(),
        dashboard=TestCrossEncoderReranker._create_dashboard_config(),
    )


# Add helper methods to test class
TestCrossEncoderReranker._create_llm_config = _create_llm_config
TestCrossEncoderReranker._create_embedding_config = _create_embedding_config
TestCrossEncoderReranker._create_vector_store_config = _create_vector_store_config
TestCrossEncoderReranker._create_retrieval_config = _create_retrieval_config
TestCrossEncoderReranker._create_rerank_config = _create_rerank_config
TestCrossEncoderReranker._create_evaluation_config = _create_evaluation_config
TestCrossEncoderReranker._create_observability_config = _create_observability_config
TestCrossEncoderReranker._create_dashboard_config = _create_dashboard_config
TestCrossEncoderReranker._create_settings = _create_settings
