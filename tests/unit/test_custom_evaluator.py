"""
Unit tests for Custom Evaluator and EvaluatorFactory (Task B6)

These tests verify that:
1. The BaseEvaluator interface is correctly defined
2. EvaluatorFactory can create instances based on settings
3. Factory routing logic works correctly
4. CustomEvaluator computes Hit Rate and MRR correctly
5. Metrics are stable and deterministic

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import List

import pytest

from src.core.settings import (
    Settings,
    EvaluationConfig,
    LLMConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    RetrievalConfig,
    RerankConfig,
    ObservabilityConfig,
    DashboardConfig,
)
from src.libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvaluationQuery,
    EvaluationMetrics,
    EvaluationResult,
)
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.libs.evaluator.evaluator_factory import EvaluatorFactory


class TestEvaluationQuery:
    """Test the EvaluationQuery dataclass."""

    def test_create_query(self):
        """Test creating an evaluation query."""
        query = EvaluationQuery(
            query_id="q1",
            query="machine learning",
            retrieved_ids=["doc1", "doc2", "doc3"],
            golden_ids=["doc2"]
        )
        assert query.query_id == "q1"
        assert query.query == "machine learning"
        assert query.retrieved_ids == ["doc1", "doc2", "doc3"]
        assert query.golden_ids == ["doc2"]


class TestEvaluationMetrics:
    """Test the EvaluationMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating evaluation metrics."""
        metrics = EvaluationMetrics(
            hit_rate_at_k=1.0,
            mrr=0.5,
            ndcg_at_k=0.8
        )
        assert metrics.hit_rate_at_k == 1.0
        assert metrics.mrr == 0.5
        assert metrics.ndcg_at_k == 0.8

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = EvaluationMetrics(
            hit_rate_at_k=0.8,
            mrr=0.65,
            custom_metrics={"precision": 0.7}
        )
        result = metrics.to_dict()

        assert result["hit_rate_at_k"] == 0.8
        assert result["mrr"] == 0.65
        assert result["precision"] == 0.7

    def test_to_dict_with_ndcg(self):
        """Test to_dict includes NDCG when present."""
        metrics = EvaluationMetrics(
            hit_rate_at_k=1.0,
            mrr=1.0,
            ndcg_at_k=0.9
        )
        result = metrics.to_dict()

        assert "ndcg_at_k" in result
        assert result["ndcg_at_k"] == 0.9

    def test_to_dict_without_ndcg(self):
        """Test to_dict excludes NDCG when absent."""
        metrics = EvaluationMetrics(
            hit_rate_at_k=1.0,
            mrr=1.0
        )
        result = metrics.to_dict()

        assert "ndcg_at_k" not in result


class TestBaseEvaluator:
    """Test the BaseEvaluator abstract interface."""

    def test_base_evaluator_cannot_be_instantiated(self):
        """Test that BaseEvaluator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEvaluator(top_k=10)

    def test_base_evaluator_subclass_must_implement_evaluate(self):
        """Test that subclasses must implement evaluate method."""

        class IncompleteEvaluator(BaseEvaluator):
            def __init__(self):
                super().__init__(top_k=10)

            @property
            def evaluator_name(self) -> str:
                return "incomplete"

            # Missing evaluate() and evaluate_single() methods

        with pytest.raises(TypeError):
            IncompleteEvaluator()

    def test_base_evaluator_initialization(self):
        """Test BaseEvaluator initialization with parameters."""
        evaluator = CustomEvaluator(top_k=5)
        assert evaluator.top_k == 5

    def test_base_evaluator_repr(self):
        """Test string representation of evaluator."""
        evaluator = CustomEvaluator(top_k=10)
        repr_str = repr(evaluator)
        assert "CustomEvaluator" in repr_str
        assert "10" in repr_str
        assert "custom" in repr_str


class TestCustomEvaluator:
    """Test the CustomEvaluator implementation."""

    def test_hit_rate_with_hit_in_top_k(self):
        """Test Hit Rate when golden ID is in top-K."""
        evaluator = CustomEvaluator(top_k=3)
        query = EvaluationQuery(
            query_id="q1",
            query="test",
            retrieved_ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
            golden_ids=["doc3"]
        )

        metrics = evaluator.evaluate_single(query)

        assert metrics.hit_rate_at_k == 1.0

    def test_hit_rate_with_no_hit_in_top_k(self):
        """Test Hit Rate when golden ID is not in top-K."""
        evaluator = CustomEvaluator(top_k=2)
        query = EvaluationQuery(
            query_id="q1",
            query="test",
            retrieved_ids=["doc1", "doc2", "doc3"],
            golden_ids=["doc5"]
        )

        metrics = evaluator.evaluate_single(query)

        assert metrics.hit_rate_at_k == 0.0

    def test_hit_rate_respects_top_k(self):
        """Test Hit Rate respects top_K parameter."""
        evaluator = CustomEvaluator(top_k=2)
        query = EvaluationQuery(
            query_id="q1",
            query="test",
            retrieved_ids=["doc1", "doc2", "doc3", "doc4"],
            golden_ids=["doc3"]
        )

        metrics = evaluator.evaluate_single(query)

        # doc3 is at rank 3, but top_k=2
        assert metrics.hit_rate_at_k == 0.0

    def test_mrr_with_first_hit_at_rank_1(self):
        """Test MRR when first golden ID is at rank 1."""
        evaluator = CustomEvaluator()
        query = EvaluationQuery(
            query_id="q1",
            query="test",
            retrieved_ids=["doc1", "doc2", "doc3"],
            golden_ids=["doc1", "doc5"]
        )

        metrics = evaluator.evaluate_single(query)

        assert metrics.mrr == 1.0

    def test_mrr_with_first_hit_at_rank_3(self):
        """Test MRR when first golden ID is at rank 3."""
        evaluator = CustomEvaluator()
        query = EvaluationQuery(
            query_id="q1",
            query="test",
            retrieved_ids=["doc1", "doc2", "doc3", "doc4"],
            golden_ids=["doc3"]
        )

        metrics = evaluator.evaluate_single(query)

        assert metrics.mrr == pytest.approx(1.0 / 3)

    def test_mrr_with_no_hit(self):
        """Test MRR when no golden ID is found."""
        evaluator = CustomEvaluator()
        query = EvaluationQuery(
            query_id="q1",
            query="test",
            retrieved_ids=["doc1", "doc2", "doc3"],
            golden_ids=["doc5"]
        )

        metrics = evaluator.evaluate_single(query)

        assert metrics.mrr == 0.0

    def test_evaluate_single_queries(self):
        """Test evaluating a single query."""
        evaluator = CustomEvaluator(top_k=5)
        query = EvaluationQuery(
            query_id="q1",
            query="ml basics",
            retrieved_ids=["doc1", "doc2", "doc3"],
            golden_ids=["doc2"]
        )

        result = evaluator.evaluate([query])

        assert result.total_queries == 1
        assert "q1" in result.query_metrics
        assert result.metrics.hit_rate_at_k == 1.0

    def test_evaluate_multiple_queries(self):
        """Test evaluating multiple queries."""
        evaluator = CustomEvaluator(top_k=3)
        queries = [
            EvaluationQuery(
                query_id="q1",
                query="test1",
                retrieved_ids=["doc1", "doc2"],
                golden_ids=["doc1"]
            ),
            EvaluationQuery(
                query_id="q2",
                query="test2",
                retrieved_ids=["doc3", "doc4"],
                golden_ids=["doc5"]
            ),
        ]

        result = evaluator.evaluate(queries)

        assert result.total_queries == 2
        assert len(result.query_metrics) == 2
        # Average: (1.0 + 0.0) / 2 = 0.5
        assert result.metrics.hit_rate_at_k == 0.5

    def test_evaluate_empty_queries_raises_error(self):
        """Test that empty queries list raises ValueError."""
        evaluator = CustomEvaluator()
        with pytest.raises(ValueError) as exc_info:
            evaluator.evaluate([])
        assert "cannot be empty" in str(exc_info.value).lower()

    def test_aggregate_metrics_averages_correctly(self):
        """Test that aggregation averages metrics across queries."""
        evaluator = CustomEvaluator()
        queries = [
            EvaluationQuery("q1", "t1", ["d1"], ["d1"]),  # Hit=1, MRR=1
            EvaluationQuery("q2", "t2", ["d2"], ["d3"]),  # Hit=0, MRR=0
        ]

        result = evaluator.evaluate(queries)

        assert result.metrics.hit_rate_at_k == 0.5  # (1+0)/2
        assert result.metrics.mrr == 0.5  # (1+0)/2

    def test_evaluator_name(self):
        """Test CustomEvaluator evaluator_name property."""
        evaluator = CustomEvaluator()
        assert evaluator.evaluator_name == "custom"

    def test_metrics_stability(self):
        """Test that metrics are stable (same input → same output)."""
        evaluator = CustomEvaluator(top_k=3)
        query = EvaluationQuery(
            query_id="q1",
            query="test",
            retrieved_ids=["doc1", "doc2", "doc3"],
            golden_ids=["doc2"]
        )

        metrics1 = evaluator.evaluate_single(query)
        metrics2 = evaluator.evaluate_single(query)

        assert metrics1.hit_rate_at_k == metrics2.hit_rate_at_k
        assert metrics1.mrr == metrics2.mrr


class TestEvaluatorFactory:
    """Test the EvaluatorFactory class."""

    def test_factory_create_with_custom_provider(self):
        """Test creating CustomEvaluator through factory."""
        settings = self._create_settings(backends=["custom"])
        evaluator = EvaluatorFactory.create(settings)

        assert isinstance(evaluator, CustomEvaluator)
        assert evaluator.evaluator_name == "custom"

    def test_factory_create_with_custom_top_k(self):
        """Test factory uses default top_k for evaluator."""
        settings = self._create_settings(backends=["custom"])
        evaluator = EvaluatorFactory.create(settings)

        # Default top_k is used (from factory's default value)
        assert evaluator.top_k == 10

    def test_factory_create_falls_back_to_custom(self):
        """Test factory falls back to custom if preferred backend unavailable."""
        settings = self._create_settings(backends=["ragas", "custom"])
        evaluator = EvaluatorFactory.create(settings)

        # Should fall back to custom since ragas is not implemented
        assert isinstance(evaluator, CustomEvaluator)

    def test_factory_create_with_no_supported_provider_raises_error(self):
        """Test that no supported providers raises ValueError."""
        settings = self._create_settings(backends=["ragas", "deepeval"])

        with pytest.raises(ValueError) as exc_info:
            EvaluatorFactory.create(settings)
        assert "Unable to create any evaluator" in str(exc_info.value)

    def test_factory_list_providers(self):
        """Test getting list of available providers."""
        providers = EvaluatorFactory.list_providers()
        assert isinstance(providers, list)
        assert "custom" in providers

    def test_factory_register_provider(self):
        """Test registering a custom provider."""
        # Create a custom evaluator class
        class CustomEvaluator2(BaseEvaluator):
            def __init__(self, top_k=10, **kwargs):
                super().__init__(top_k, **kwargs)

            def evaluate(self, queries, **kwargs):
                return EvaluationResult(
                    metrics=EvaluationMetrics(hit_rate_at_k=0.5, mrr=0.5),
                    query_metrics={},
                    evaluator_name="custom2",
                    total_queries=len(queries)
                )

            def evaluate_single(self, query, **kwargs):
                return EvaluationMetrics(hit_rate_at_k=0.5, mrr=0.5)

            @property
            def evaluator_name(self):
                return "custom2"

        # Register the provider
        EvaluatorFactory.register_provider("custom2", CustomEvaluator2)

        # Verify it's in the list
        providers = EvaluatorFactory.list_providers()
        assert "custom2" in providers

    def test_factory_register_non_evaluator_class_raises_error(self):
        """Test that registering non-BaseEvaluator class raises TypeError."""
        class NotAnEvaluator:
            pass

        with pytest.raises(TypeError) as exc_info:
            EvaluatorFactory.register_provider("invalid", NotAnEvaluator)
        assert "BaseEvaluator" in str(exc_info.value)

    def _create_settings(
        self,
        backends: List[str] = None,
        top_k: int = 10
    ) -> Settings:
        """Helper method to create test Settings object."""
        if backends is None:
            backends = ["custom"]

        return Settings(
            llm=LLMConfig(provider="fake", model="fake-model"),
            embedding=EmbeddingConfig(provider="fake", model="fake-embedding"),
            vector_store=VectorStoreConfig(backend="chroma"),
            retrieval=RetrievalConfig(),
            rerank=RerankConfig(),
            evaluation=EvaluationConfig(backends=backends),
            observability=ObservabilityConfig(),
            dashboard=DashboardConfig(),
        )


class TestEvaluatorIntegration:
    """Integration tests for evaluator usage patterns."""

    def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow."""
        settings = self._create_settings()
        evaluator = EvaluatorFactory.create(settings)

        queries = [
            EvaluationQuery(
                query_id="q1",
                query="machine learning basics",
                retrieved_ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
                golden_ids=["doc2", "doc5"]
            ),
            EvaluationQuery(
                query_id="q2",
                query="deep learning",
                retrieved_ids=["doc6", "doc7", "doc8"],
                golden_ids=["doc9"]
            ),
        ]

        result = evaluator.evaluate(queries)

        # Verify aggregated metrics
        assert result.total_queries == 2
        assert 0 <= result.metrics.hit_rate_at_k <= 1
        assert 0 <= result.metrics.mrr <= 1

        # Verify per-query metrics
        assert "q1" in result.query_metrics
        assert "q2" in result.query_metrics

    def test_evaluation_result_structure(self):
        """Test EvaluationResult contains all required fields."""
        settings = self._create_settings()
        evaluator = EvaluatorFactory.create(settings)

        query = EvaluationQuery(
            query_id="q1",
            query="test",
            retrieved_ids=["doc1"],
            golden_ids=["doc1"]
        )

        result = evaluator.evaluate([query])

        # Check result structure
        assert hasattr(result, 'metrics')
        assert hasattr(result, 'query_metrics')
        assert hasattr(result, 'evaluator_name')
        assert hasattr(result, 'total_queries')

        # Check metrics structure
        assert isinstance(result.metrics, EvaluationMetrics)
        assert result.evaluator_name == "custom"
        assert result.total_queries == 1

    def test_perfect_retrieval_scenario(self):
        """Test evaluation with perfect retrieval (all hits)."""
        settings = self._create_settings(top_k=5)
        evaluator = EvaluatorFactory.create(settings)

        queries = [
            EvaluationQuery(f"q{i}", f"query {i}", [f"doc{i}"], [f"doc{i}"])
            for i in range(10)
        ]

        result = evaluator.evaluate(queries)

        # Perfect retrieval should yield Hit Rate = 1.0 and MRR = 1.0
        assert result.metrics.hit_rate_at_k == 1.0
        assert result.metrics.mrr == 1.0

    def test_zero_retrieval_scenario(self):
        """Test evaluation with zero retrieval (no hits)."""
        settings = self._create_settings(top_k=5)
        evaluator = EvaluatorFactory.create(settings)

        queries = [
            EvaluationQuery(f"q{i}", f"query {i}", [f"doc{i}"], [f"doc{i}_wrong"])
            for i in range(10)
        ]

        result = evaluator.evaluate(queries)

        # Zero retrieval should yield Hit Rate = 0.0 and MRR = 0.0
        assert result.metrics.hit_rate_at_k == 0.0
        assert result.metrics.mrr == 0.0

    def _create_settings(self, backends: List[str] = None, top_k: int = 10) -> Settings:
        """Helper to create minimal settings for testing."""
        if backends is None:
            backends = ["custom"]

        return Settings(
            llm=LLMConfig(provider="fake", model="fake-model"),
            embedding=EmbeddingConfig(provider="fake", model="fake-embedding"),
            vector_store=VectorStoreConfig(backend="chroma"),
            retrieval=RetrievalConfig(),
            rerank=RerankConfig(),
            evaluation=EvaluationConfig(backends=backends),
            observability=ObservabilityConfig(),
            dashboard=DashboardConfig(),
        )
