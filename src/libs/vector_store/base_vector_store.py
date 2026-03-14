"""
Base VectorStore abstract interface for the Modular RAG MCP Server.

This module defines the abstract interface that all vector store providers must implement.
Vector stores are responsible for storing and retrieving embeddings with associated metadata.

Author: Modular RAG MCP Server Project
License: MIT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class VectorRecord:
    """
    A single vector record with associated metadata.

    Attributes:
        id: Unique identifier for this record
        vector: The embedding vector (list of floats)
        metadata: Optional metadata dictionary (e.g., source, page, title)
        payload: Optional payload data (e.g., chunk text)
    """
    id: str
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None
    payload: Optional[str] = None


@dataclass
class QueryResult:
    """
    Result from a vector store query.

    Attributes:
        id: Record identifier
        score: Similarity score (higher = more similar)
        metadata: Associated metadata
        payload: Payload data (e.g., chunk text)
    """
    id: str
    score: float
    metadata: Optional[Dict[str, Any]] = None
    payload: Optional[str] = None


@dataclass
class UpsertResult:
    """
    Result from an upsert operation.

    Attributes:
        upserted_count: Number of records upserted
        deleted_count: Number of records deleted (if any)
    """
    upserted_count: int
    deleted_count: int = 0


class BaseVectorStore(ABC):
    """
    Abstract base class for vector store providers.

    All vector store implementations (Chroma, Qdrant, Pinecone, etc.) must inherit from
    this class and implement the upsert() and query() methods.

    The interface focuses on:
    - Batch upsert operations for efficient bulk inserts
    - Vector similarity search with optional metadata filtering
    - Collection management (create, delete, list)
    """

    def __init__(
        self,
        collection_name: str = "default",
        persist_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the collection/table
            persist_path: Path for persistent storage (if applicable)
            **kwargs: Additional provider-specific parameters
        """
        self.collection_name = collection_name
        self.persist_path = persist_path
        self._provider_config = kwargs

    @abstractmethod
    def upsert(
        self,
        records: List[VectorRecord],
        **kwargs
    ) -> UpsertResult:
        """
        Upsert (insert or update) vector records.

        This method should handle batch upserts efficiently. If a record with the same
        ID already exists, it should be updated; otherwise, it should be inserted.

        Args:
            records: List of vector records to upsert
            **kwargs: Additional provider-specific parameters

        Returns:
            UpsertResult containing counts of upserted and deleted records

        Raises:
            RuntimeError: If the upsert operation fails
            ValueError: If records list is empty or parameters are invalid

        Example:
            >>> store = VectorStoreFactory.create(settings)
            >>> records = [
            ...     VectorRecord(id="1", vector=[0.1, 0.2], metadata={"source": "doc1.pdf"}),
            ...     VectorRecord(id="2", vector=[0.3, 0.4], metadata={"source": "doc1.pdf"}),
            ... ]
            >>> result = store.upsert(records)
            >>> print(result.upserted_count)  # 2
        """
        pass

    @abstractmethod
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[QueryResult]:
        """
        Query the vector store for similar vectors.

        This method should perform vector similarity search and return the top-k
        most similar records. Results should be sorted by similarity score in
        descending order (most similar first).

        Args:
            vector: Query vector
            top_k: Maximum number of results to return
            filters: Optional metadata filters (e.g., {"source": "doc1.pdf"})
            **kwargs: Additional provider-specific parameters

        Returns:
            List of QueryResult objects, sorted by score (descending)

        Raises:
            RuntimeError: If the query operation fails
            ValueError: If vector dimension doesn't match stored vectors

        Example:
            >>> store = VectorStoreFactory.create(settings)
            >>> results = store.query(
            ...     vector=[0.15, 0.25],
            ...     top_k=5,
            ...     filters={"source": "doc1.pdf"}
            ... )
            >>> for result in results:
            ...     print(f"{result.id}: {result.score:.3f}")
        """
        pass

    @abstractmethod
    def delete(
        self,
        ids: List[str],
        **kwargs
    ) -> int:
        """
        Delete records by IDs.

        Args:
            ids: List of record IDs to delete
            **kwargs: Additional provider-specific parameters

        Returns:
            Number of records deleted

        Raises:
            RuntimeError: If the delete operation fails
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Get the total number of records in the collection.

        Returns:
            Number of records
        """
        pass

    @abstractmethod
    def delete_collection(self, **kwargs) -> None:
        """
        Delete the entire collection.

        This is a destructive operation that cannot be undone.

        Raises:
            RuntimeError: If the deletion fails
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Get the provider name (e.g., "chroma", "qdrant", "pinecone").

        Returns:
            Provider name as a string
        """
        pass

    def __repr__(self) -> str:
        """String representation of the vector store instance."""
        return f"{self.__class__.__name__}(collection='{self.collection_name}', provider='{self.provider_name}')"
