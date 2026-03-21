"""
Cross-Encoder Reranker provider implementation.

This module implements the BaseReranker interface using cross-encoder models
for fine-grained relevance scoring and reranking of retrieved candidates.

Cross-encoder models take (query, document) pairs as input and output a
relevance score, making them more accurate than bi-encoders but slower.

Author: Modular RAG MCP Server Project
License: MIT
"""

import time
from typing import List, Optional, Callable

from src.libs.reranker.base_reranker import (
    BaseReranker,
    RerankCandidate,
    RerankResult,
)
from src.observability.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder-based reranker provider implementation.

    This class uses cross-encoder models to score and rerank retrieved candidates
    based on their relevance to the query. Cross-encoders provide higher accuracy
    than bi-encoders by jointly processing query-document pairs.

    Key features:
    - Pluggable scorer function (supports real models and mocks)
    - Timeout protection for long-running models
    - Fallback mechanism on failure
    - Deterministic scoring for testing

    The scorer function should:
    - Take (query: str, content: str) as input
    - Return a relevance score (float) in [0, 1] range
    - Be thread-safe if used in concurrent scenarios

    Example with real model (future):
        >>> from sentence_transformers import CrossEncoder
        >>> model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        >>> reranker = CrossEncoderReranker(
        ...     top_k=5,
        ...     model='ms-marco-MiniLM-L-6',
        ...     scorer=lambda q, c: model.predict([[q, c]])[0]
        ... )
        >>> result = reranker.rerank(query="machine learning", candidates=[...])

    Example with mock scorer (testing):
        >>> def mock_scorer(query: str, content: str) -> float:
        ...     return 0.9 if "machine" in content.lower() else 0.5
        >>> reranker = CrossEncoderReranker(
        ...     top_k=5,
        ...     scorer=mock_scorer
        ... )
        >>> result = reranker.rerank(query="test", candidates=[...])
    """

    def __init__(
        self,
        top_k: int = 10,
        model: Optional[str] = None,
        scorer: Optional[Callable[[str, str], float]] = None,
        timeout: float = 30.0,
        **kwargs
    ):
        """
        Initialize Cross-Encoder Reranker.

        Args:
            top_k: Maximum number of candidates to return after reranking
            model: Model name or identifier (for logging/tracking)
            scorer: Scoring function that takes (query, content) and returns score [0, 1]
                   If None, uses a default placeholder scorer (for testing)
            timeout: Maximum seconds to wait for scoring (default: 30.0)
            **kwargs: Additional provider-specific parameters
        """
        super().__init__(top_k, model, **kwargs)

        self.scorer = scorer
        self.timeout = timeout

        # Statistics
        self._total_rerank_calls = 0
        self._total_candidates_scored = 0
        self._total_fallback_count = 0

        logger.debug(
            f"Initialized CrossEncoderReranker: top_k={top_k}, model={model}, "
            f"scorer={'custom' if scorer else 'default'}, timeout={timeout}s"
        )

    def _default_scorer(self, query: str, content: str) -> float:
        """
        Default placeholder scorer for testing.

        This scorer uses simple keyword matching as a fallback.
        In production, a real cross-encoder model should be used.

        Args:
            query: Search query
            content: Candidate content

        Returns:
            Relevance score in [0, 1] range
        """
        query_lower = query.lower()
        content_lower = content.lower()

        # Simple keyword matching score
        words = query_lower.split()
        matches = sum(1 for word in words if word in content_lower)
        score = matches / max(len(words), 1)

        return min(score, 1.0)

    def _score_with_timeout(
        self,
        query: str,
        candidate: RerankCandidate
    ) -> float:
        """
        Score a single candidate with timeout protection.

        Args:
            query: Search query
            candidate: Candidate to score

        Returns:
            Relevance score

        Raises:
            TimeoutError: If scoring exceeds timeout
        """
        scorer_func = self.scorer if self.scorer else self._default_scorer

        start_time = time.time()

        try:
            # Call scorer function
            score = scorer_func(query, candidate.content)

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                raise TimeoutError(
                    f"Scoring exceeded timeout of {self.timeout}s "
                    f"(took {elapsed:.2f}s)"
                )

            return float(score)

        except TimeoutError:
            raise
        except Exception as e:
            logger.warning(
                f"[{self.provider_name}] Scoring failed for candidate "
                f"'{candidate.id}': {str(e)}"
            )
            raise

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "cross_encoder"

    def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        **kwargs
    ) -> RerankResult:
        """
        Rerank candidates using cross-encoder scoring.

        This method scores each (query, candidate) pair using the cross-encoder
        model (or scorer function), then sorts candidates by score descending.

        Args:
            query: The search query
            candidates: List of candidates to rerank
            **kwargs: Additional parameters (unused in base implementation)

        Returns:
            RerankResult containing reranked candidates

        Raises:
            ValueError: If candidates list is empty
            RuntimeError: If reranking fails and fallback is not triggered
        """
        if not candidates:
            raise ValueError(
                f"[{self.provider_name}] Candidates list cannot be empty."
            )

        self._total_rerank_calls += 1

        try:
            # Score all candidates
            scored_candidates = []

            for candidate in candidates:
                try:
                    # Score with timeout protection
                    score = self._score_with_timeout(query, candidate)

                    # Create new candidate with updated score
                    updated_candidate = RerankCandidate(
                        id=candidate.id,
                        content=candidate.content,
                        score=score,
                        metadata=candidate.metadata
                    )

                    scored_candidates.append(updated_candidate)
                    self._total_candidates_scored += 1

                except TimeoutError as e:
                    logger.warning(
                        f"[{self.provider_name}] Timeout scoring candidate "
                        f"'{candidate.id}': {str(e)}"
                    )
                    # On timeout, use original score
                    scored_candidates.append(candidate)
                except Exception as e:
                    logger.warning(
                        f"[{self.provider_name}] Error scoring candidate "
                        f"'{candidate.id}': {str(e)}"
                    )
                    # On error, use original score
                    scored_candidates.append(candidate)

            # Sort by score descending
            scored_candidates.sort(key=lambda c: c.score, reverse=True)

            # Trim to top_k
            final_candidates = scored_candidates[:self.top_k]
            final_scores = [c.score for c in final_candidates]

            logger.debug(
                f"[{self.provider_name}] Successfully reranked {len(candidates)} "
                f"candidates to {len(final_candidates)} results"
            )

            return RerankResult(
                candidates=final_candidates,
                scores=final_scores,
                method="cross_encoder",
                fallback_triggered=False
            )

        except Exception as e:
            # On failure, return candidates in original order with fallback signal
            error_msg = (
                f"[{self.provider_name}] Failed to rerank with cross-encoder: "
                f"{str(e)}. Returning candidates in original order with fallback."
            )
            logger.warning(error_msg)

            self._total_fallback_count += 1

            # Return original order with fallback triggered
            trimmed_candidates = candidates[:self.top_k]
            scores = [c.score for c in trimmed_candidates]

            return RerankResult(
                candidates=trimmed_candidates,
                scores=scores,
                method="cross_encoder",
                fallback_triggered=True
            )

    def get_statistics(self) -> dict:
        """
        Get reranking statistics.

        Returns:
            Dictionary with statistics:
            - total_rerank_calls: Total number of rerank() calls
            - total_candidates_scored: Total candidates successfully scored
            - total_fallback_count: Total times fallback was triggered
        """
        return {
            "total_rerank_calls": self._total_rerank_calls,
            "total_candidates_scored": self._total_candidates_scored,
            "total_fallback_count": self._total_fallback_count,
        }

    def reset_statistics(self) -> None:
        """Reset all statistics counters to zero."""
        self._total_rerank_calls = 0
        self._total_candidates_scored = 0
        self._total_fallback_count = 0
