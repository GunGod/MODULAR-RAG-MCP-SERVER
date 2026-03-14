"""
None Reranker implementation for the Modular RAG MCP Server.

This module provides a no-op reranker that preserves the original order.
It's used as the default fallback when reranking is disabled or unavailable.

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import List, Optional

from src.libs.reranker.base_reranker import (
    BaseReranker,
    RerankCandidate,
    RerankResult,
)


class NoneReranker(BaseReranker):
    """
    No-op reranker that preserves original order.

    This reranker does not perform any actual reranking. It simply returns
    the candidates in their original order. It's useful for:
    - Disabling reranking (backend="none")
    - Fallback when other rerankers fail
    - Testing retrieval without reranking interference

    The output scores are the same as input scores, preserving the original
    retrieval ranking from fusion (RRF) stage.
    """

    def __init__(
        self,
        top_k: int = 10,
        model: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the None Reranker.

        Args:
            top_k: Maximum number of candidates to return (default: 10)
            model: Ignored (none reranker doesn't use a model)
            **kwargs: Additional ignored parameters
        """
        super().__init__(top_k, model, **kwargs)
        self._call_count = 0
        self._total_candidates_processed = 0

    def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        **kwargs
    ) -> RerankResult:
        """
        Return candidates in original order (no reranking).

        This method preserves the original order and scores from the retrieval
        stage. It simply trims to top_k if necessary.

        Args:
            query: Search query (ignored in none reranker)
            candidates: List of candidates to "rerank"
            **kwargs: Additional ignored parameters

        Returns:
            RerankResult with candidates in original order

        Raises:
            ValueError: If candidates list is empty
        """
        if not candidates:
            raise ValueError("candidates list cannot be empty")

        self._call_count += 1
        self._total_candidates_processed += len(candidates)

        # Preserve original order and scores
        trimmed_candidates = candidates[:self.top_k]
        scores = [c.score for c in trimmed_candidates]

        return RerankResult(
            candidates=trimmed_candidates,
            scores=scores,
            method="none",
            fallback_triggered=False
        )

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "none"

    def get_call_count(self) -> int:
        """
        Get the number of times rerank() has been called.

        Returns:
            Number of rerank() calls
        """
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the call counter to zero."""
        self._call_count = 0

    def get_total_candidates_processed(self) -> int:
        """
        Get total number of candidates processed across all calls.

        Returns:
            Total number of candidates
        """
        return self._total_candidates_processed
