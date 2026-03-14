"""
Custom Evaluator implementation with lightweight metrics.

This module provides a simple evaluator implementation with standard retrieval
metrics (Hit Rate, MRR) without external dependencies.

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import List, Dict

from src.libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvaluationQuery,
    EvaluationMetrics,
    EvaluationResult,
)


class CustomEvaluator(BaseEvaluator):
    """
    Custom evaluator with lightweight retrieval metrics.

    This evaluator implements standard retrieval metrics without requiring
    external libraries or LLM calls. It's useful for:
    - Quick evaluation during development
    - Regression testing
    - Baseline comparison

    Supported metrics:
        - Hit Rate@K: 1 if any golden ID appears in top-K, else 0
        - MRR (Mean Reciprocal Rank): 1/rank of first golden ID
        - NDCG@K: Normalized Discounted Cumulative Gain (optional)
    """

    def __init__(
        self,
        top_k: int = 10,
        compute_ndcg: bool = False,
        **kwargs
    ):
        """
        Initialize the Custom Evaluator.

        Args:
            top_k: Top-K value for metrics (default: 10)
            compute_ndcg: Whether to compute NDCG@K (default: False)
            **kwargs: Additional ignored parameters
        """
        super().__init__(top_k, **kwargs)
        self.compute_ndcg = compute_ndcg

    def evaluate(
        self,
        queries: List[EvaluationQuery],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate multiple queries and aggregate metrics.

        Args:
            queries: List of queries to evaluate
            **kwargs: Additional ignored parameters

        Returns:
            EvaluationResult with aggregated metrics

        Raises:
            ValueError: If queries list is empty
        """
        if not queries:
            raise ValueError("queries list cannot be empty")

        # Evaluate each query
        query_metrics = {}
        for query in queries:
            metrics = self.evaluate_single(query)
            query_metrics[query.query_id] = metrics

        # Aggregate metrics (average across queries)
        aggregated = self._aggregate_metrics(query_metrics)

        return EvaluationResult(
            metrics=aggregated,
            query_metrics=query_metrics,
            evaluator_name=self.evaluator_name,
            total_queries=len(queries)
        )

    def evaluate_single(
        self,
        query: EvaluationQuery,
        **kwargs
    ) -> EvaluationMetrics:
        """
        Evaluate a single query.

        Args:
            query: Query to evaluate
            **kwargs: Additional ignored parameters

        Returns:
            EvaluationMetrics for the query
        """
        # Compute Hit Rate@K
        hit_rate_at_k = self._compute_hit_rate(
            query.retrieved_ids,
            query.golden_ids
        )

        # Compute MRR
        mrr = self._compute_mrr(
            query.retrieved_ids,
            query.golden_ids
        )

        # Compute NDCG@K (optional)
        ndcg_at_k = None
        if self.compute_ndcg:
            ndcg_at_k = self._compute_ndcg(
                query.retrieved_ids,
                query.golden_ids
            )

        return EvaluationMetrics(
            hit_rate_at_k=hit_rate_at_k,
            mrr=mrr,
            ndcg_at_k=ndcg_at_k
        )

    def _compute_hit_rate(
        self,
        retrieved_ids: List[str],
        golden_ids: List[str]
    ) -> float:
        """
        Compute Hit Rate@K.

        Hit Rate@K = 1 if any golden ID appears in top-K retrieved IDs, else 0.

        Args:
            retrieved_ids: List of retrieved document IDs (ranked)
            golden_ids: List of relevant document IDs

        Returns:
            Hit Rate@K score (0 or 1)
        """
        # Consider only top-K retrieved IDs
        top_k_retrieved = retrieved_ids[:self.top_k]

        # Check if any golden ID is in top-K
        for golden_id in golden_ids:
            if golden_id in top_k_retrieved:
                return 1.0

        return 0.0

    def _compute_mrr(
        self,
        retrieved_ids: List[str],
        golden_ids: List[str]
    ) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).

        MRR = 1 / rank of first golden ID in retrieved list

        Args:
            retrieved_ids: List of retrieved document IDs (ranked)
            golden_ids: List of relevant document IDs

        Returns:
            MRR score (0 if no golden ID is found)
        """
        # Find the rank of the first golden ID
        for rank, retrieved_id in enumerate(retrieved_ids, start=1):
            if retrieved_id in golden_ids:
                return 1.0 / rank

        return 0.0

    def _compute_ndcg(
        self,
        retrieved_ids: List[str],
        golden_ids: List[str]
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain@K (NDCG@K).

        NDCG@K measures ranking quality considering:
        - Relevance (binary: 1 if golden, else 0)
        - Position (discounted gain: log2(rank + 1))
        - Normalization by ideal DCG

        Args:
            retrieved_ids: List of retrieved document IDs (ranked)
            golden_ids: List of relevant document IDs

        Returns:
            NDCG@K score (0 to 1)
        """
        # Compute DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for rank, retrieved_id in enumerate(retrieved_ids[:self.top_k], start=1):
            # Binary relevance: 1 if golden, else 0
            relevance = 1.0 if retrieved_id in golden_ids else 0.0
            # Discount: log2(rank + 1)
            dcg += relevance / (rank + 1).bit_length()  # Using bit_length for log2

        # Compute Ideal DCG (all golden IDs at top positions)
        ideal_dcg = 0.0
        for rank in range(1, min(len(golden_ids), self.top_k) + 1):
            ideal_dcg += 1.0 / (rank + 1).bit_length()

        # Normalize
        if ideal_dcg == 0:
            return 0.0

        return dcg / ideal_dcg

    def _aggregate_metrics(
        self,
        query_metrics: Dict[str, EvaluationMetrics]
    ) -> EvaluationMetrics:
        """
        Aggregate metrics across multiple queries.

        Args:
            query_metrics: Per-query metrics

        Returns:
            Aggregated metrics (averaged)
        """
        if not query_metrics:
            return EvaluationMetrics(
                hit_rate_at_k=0.0,
                mrr=0.0
            )

        # Average each metric
        total_hit_rate = sum(m.hit_rate_at_k for m in query_metrics.values())
        total_mrr = sum(m.mrr for m in query_metrics.values())
        count = len(query_metrics)

        # Handle NDCG (only if computed)
        ndcg_values = [m.ndcg_at_k for m in query_metrics.values() if m.ndcg_at_k is not None]
        avg_ndcg = sum(ndcg_values) / len(ndcg_values) if ndcg_values else None

        return EvaluationMetrics(
            hit_rate_at_k=total_hit_rate / count,
            mrr=total_mrr / count,
            ndcg_at_k=avg_ndcg
        )

    @property
    def evaluator_name(self) -> str:
        """Get the evaluator name."""
        return "custom"
