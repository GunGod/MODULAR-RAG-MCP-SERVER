"""
Base Reranker abstract interface for the Modular RAG MCP Server.

This module defines the abstract interface that all reranker providers must implement.
Rerankers perform fine-grained reordering of retrieved candidates to improve relevance.

Author: Modular RAG MCP Server Project
License: MIT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class RerankCandidate:
    """
    A single candidate to be reranked.

    Attributes:
        id: Unique identifier for the candidate (e.g., chunk_id)
        content: The text content of the candidate
        score: Original relevance score from retrieval (e.g., similarity score)
        metadata: Optional metadata (e.g., source, page, title)
    """
    id: str
    content: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RerankResult:
    """
    Result from a reranking operation.

    Attributes:
        candidates: Reranked list of candidates (sorted by new_score descending)
        scores: List of new relevance scores after reranking
        method: Reranking method used (e.g., "none", "cross_encoder", "llm")
        fallback_triggered: Whether fallback to original order was triggered
    """
    candidates: List[RerankCandidate]
    scores: List[float]
    method: str
    fallback_triggered: bool = False


class BaseReranker(ABC):
    """
    Abstract base class for reranker providers.

    All reranker implementations (None, CrossEncoder, LLM, etc.) must inherit from
    this class and implement the rerank() method.

    The interface focuses on:
    - Reordering candidates based on query relevance
    - Providing fallback mechanisms for reliability
    - Tracking whether fallback was triggered
    """

    def __init__(
        self,
        top_k: int = 10,
        model: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the reranker.

        Args:
            top_k: Maximum number of candidates to return after reranking
            model: Model name or identifier (provider-specific)
            **kwargs: Additional provider-specific parameters
        """
        self.top_k = top_k
        self.model = model
        self._provider_config = kwargs

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        **kwargs
    ) -> RerankResult:
        """
        Rerank candidates based on query relevance.

        This method should reorder the input candidates based on their relevance
        to the query. The implementation may use various strategies:
        - None: Return candidates in original order
        - CrossEncoder: Use a cross-encoder model to score candidates
        - LLM: Use an LLM to rank candidates

        Args:
            query: The search query
            candidates: List of candidates to rerank (from retrieval stage)
            **kwargs: Additional provider-specific parameters

        Returns:
            RerankResult containing reranked candidates and new scores

        Raises:
            RuntimeError: If the reranking operation fails
            ValueError: If candidates list is empty or parameters are invalid

        Example:
            >>> reranker = RerankerFactory.create(settings)
            >>> results = reranker.rerank(
            ...     query="machine learning basics",
            ...     candidates=[candidate1, candidate2, candidate3]
            ... )
            >>> for candidate in results.candidates:
            ...     print(f"{candidate.id}: {candidate.score}")
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Get the provider name (e.g., "none", "cross_encoder", "llm").

        Returns:
            Provider name as a string
        """
        pass

    def __repr__(self) -> str:
        """String representation of the reranker instance."""
        return f"{self.__class__.__name__}(top_k={self.top_k}, provider='{self.provider_name}')"
