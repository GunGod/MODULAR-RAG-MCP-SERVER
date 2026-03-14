"""
Fake VectorStore implementation for testing purposes.

This module provides a mock vector store that stores vectors in memory.
It's used for testing the factory routing logic and for development when
actual vector databases are not available.

Author: Modular RAG MCP Server Project
License: MIT
"""

import math
from typing import List, Dict, Any, Optional

from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult,
    UpsertResult,
)


class FakeVectorStore(BaseVectorStore):
    """
    Fake vector store implementation for testing.

    This store keeps all data in memory (Python dictionary). It does not
    persist any data to disk. It's useful for:
    - Testing the VectorStoreFactory routing logic
    - Development without database dependencies
    - Unit tests that don't require external services

    Similarity computation uses simple cosine similarity.
    """

    def __init__(
        self,
        collection_name: str = "default",
        persist_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Fake VectorStore.

        Args:
            collection_name: Name of the collection
            persist_path: Ignored (fake implementation doesn't persist)
            **kwargs: Additional ignored parameters
        """
        super().__init__(collection_name, persist_path, **kwargs)
        self._records: Dict[str, VectorRecord] = {}
        self._upsert_count = 0
        self._query_count = 0

    def upsert(
        self,
        records: List[VectorRecord],
        **kwargs
    ) -> UpsertResult:
        """
        Upsert vector records (in-memory storage).

        Args:
            records: List of vector records to upsert
            **kwargs: Additional ignored parameters

        Returns:
            UpsertResult with count of upserted records

        Raises:
            ValueError: If records list is empty
        """
        if not records:
            raise ValueError("records list cannot be empty")

        for record in records:
            self._records[record.id] = record

        self._upsert_count += len(records)
        return UpsertResult(upserted_count=len(records))

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[QueryResult]:
        """
        Query the vector store using cosine similarity.

        Args:
            vector: Query vector
            top_k: Maximum number of results to return
            filters: Optional metadata filters
            **kwargs: Additional ignored parameters

        Returns:
            List of QueryResult objects, sorted by score (descending)

        Raises:
            ValueError: If vector is empty or top_k <= 0
        """
        if not vector:
            raise ValueError("query vector cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        self._query_count += 1

        # Compute cosine similarity for all records
        results = []
        for record_id, record in self._records.items():
            # Apply metadata filters if provided
            if filters and not self._matches_filters(record.metadata, filters):
                continue

            # Compute cosine similarity
            score = self._cosine_similarity(vector, record.vector)
            results.append(
                QueryResult(
                    id=record_id,
                    score=score,
                    metadata=record.metadata,
                    payload=record.payload,
                )
            )

        # Sort by score (descending) and return top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def delete(self, ids: List[str], **kwargs) -> int:
        """
        Delete records by IDs.

        Args:
            ids: List of record IDs to delete
            **kwargs: Additional ignored parameters

        Returns:
            Number of records deleted
        """
        count = 0
        for record_id in ids:
            if record_id in self._records:
                del self._records[record_id]
                count += 1
        return count

    def count(self) -> int:
        """
        Get the total number of records in the collection.

        Returns:
            Number of records
        """
        return len(self._records)

    def delete_collection(self, **kwargs) -> None:
        """
        Delete the entire collection (clear all records).

        This is a destructive operation that cannot be undone.
        """
        self._records.clear()

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "fake"

    def get_upsert_count(self) -> int:
        """
        Get total number of records upserted across all calls.

        Returns:
            Total upsert count
        """
        return self._upsert_count

    def get_query_count(self) -> int:
        """
        Get total number of queries executed.

        Returns:
            Total query count
        """
        return self._query_count

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (between -1 and 1, where 1 is identical)
        """
        if len(vec1) != len(vec2):
            raise ValueError("vector dimensions must match")

        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(v * v for v in vec1))
        magnitude2 = math.sqrt(sum(v * v for v in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _matches_filters(
        self,
        metadata: Optional[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> bool:
        """
        Check if metadata matches all filters.

        Args:
            metadata: Metadata dictionary
            filters: Filter criteria

        Returns:
            True if all filters match, False otherwise
        """
        if not metadata:
            return not filters  # No metadata matches only empty filters

        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False

        return True
