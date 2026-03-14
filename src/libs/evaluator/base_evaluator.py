"""
Base Evaluator abstract interface for the Modular RAG MCP Server.

This module defines the abstract interface that all evaluator providers must implement.
Evaluators measure the quality of RAG retrieval and generation results.

Author: Modular RAG MCP Server Project
License: MIT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class EvaluationQuery:
    """
    A single query to be evaluated.

    Attributes:
        query_id: Unique identifier for the query
        query: The search query text
        retrieved_ids: List of retrieved document IDs (ranked by relevance)
        golden_ids: List of relevant document IDs (ground truth)
    """
    query_id: str
    query: str
    retrieved_ids: List[str]
    golden_ids: List[str]


@dataclass
class EvaluationMetrics:
    """
    Evaluation metrics for a single query or overall.

    Attributes:
        hit_rate_at_k: Hit Rate@K (1 if any golden ID in top-K, else 0)
        mrr: Mean Reciprocal Rank (1/rank of first golden ID)
        ndcg_at_k: Normalized Discounted Cumulative Gain@K (optional)
        custom_metrics: Additional custom metrics (provider-specific)
    """
    hit_rate_at_k: float
    mrr: float
    ndcg_at_k: Optional[float] = None
    custom_metrics: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, float]:
        """
        Convert metrics to a dictionary.

        Returns:
            Dictionary with all metric names as keys
        """
        result = {
            "hit_rate_at_k": self.hit_rate_at_k,
            "mrr": self.mrr,
        }
        if self.ndcg_at_k is not None:
            result["ndcg_at_k"] = self.ndcg_at_k
        if self.custom_metrics:
            result.update(self.custom_metrics)
        return result


@dataclass
class EvaluationResult:
    """
    Result from an evaluation operation.

    Attributes:
        metrics: Aggregated metrics across all queries
        query_metrics: Per-query metrics (for detailed analysis)
        evaluator_name: Name of the evaluator used
        total_queries: Total number of queries evaluated
    """
    metrics: EvaluationMetrics
    query_metrics: Dict[str, EvaluationMetrics]
    evaluator_name: str
    total_queries: int


class BaseEvaluator(ABC):
    """
    Abstract base class for evaluator providers.

    All evaluator implementations (Custom, Ragas, DeepEval, etc.) must inherit from
    this class and implement the evaluate() method.

    The interface focuses on:
    - Computing retrieval metrics (Hit Rate, MRR, NDCG)
    - Supporting per-query and aggregated metrics
    - Extensibility for custom metrics
    """

    def __init__(
        self,
        top_k: int = 10,
        **kwargs
    ):
        """
        Initialize the evaluator.

        Args:
            top_k: Top-K value for metrics (e.g., Hit Rate@10)
            **kwargs: Additional provider-specific parameters
        """
        self.top_k = top_k
        self._provider_config = kwargs

    @abstractmethod
    def evaluate(
        self,
        queries: List[EvaluationQuery],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate retrieval quality for a list of queries.

        This method should compute metrics for each query and aggregate them
        across all queries (typically by averaging).

        Args:
            queries: List of queries to evaluate
            **kwargs: Additional provider-specific parameters

        Returns:
            EvaluationResult containing aggregated and per-query metrics

        Raises:
            RuntimeError: If the evaluation fails
            ValueError: If queries list is empty or parameters are invalid

        Example:
            >>> evaluator = EvaluatorFactory.create(settings)
            >>> queries = [
            ...     EvaluationQuery(
            ...         query_id="q1",
            ...         query="machine learning",
            ...         retrieved_ids=["doc1", "doc2", "doc3"],
            ...         golden_ids=["doc2", "doc5"]
            ...     ),
            ... ]
            >>> result = evaluator.evaluate(queries)
            >>> print(f"Hit Rate: {result.metrics.hit_rate_at_k}")
            >>> print(f"MRR: {result.metrics.mrr}")
        """
        pass

    @abstractmethod
    def evaluate_single(
        self,
        query: EvaluationQuery,
        **kwargs
    ) -> EvaluationMetrics:
        """
        Evaluate a single query.

        This method should compute metrics for one query.

        Args:
            query: Single query to evaluate
            **kwargs: Additional provider-specific parameters

        Returns:
            EvaluationMetrics for the query

        Raises:
            RuntimeError: If the evaluation fails
            ValueError: If query parameters are invalid
        """
        pass

    @property
    @abstractmethod
    def evaluator_name(self) -> str:
        """
        Get the evaluator name (e.g., "custom", "ragas", "deepeval").

        Returns:
            Evaluator name as a string
        """
        pass

    def __repr__(self) -> str:
        """String representation of the evaluator instance."""
        return f"{self.__class__.__name__}(top_k={self.top_k}, evaluator='{self.evaluator_name}')"
