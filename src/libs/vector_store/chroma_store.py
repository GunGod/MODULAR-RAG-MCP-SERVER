"""
ChromaStore provider implementation.

This module implements the BaseVectorStore interface using ChromaDB as the backend.
ChromaDB is an embedded vector database that requires no external services, making it
ideal for local development and rapid prototyping.

Author: Modular RAG MCP Server Project
License: MIT
"""

import os
import shutil
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult,
    UpsertResult,
)
from src.observability.logger import get_logger

logger = get_logger(__name__)


class ChromaStore(BaseVectorStore):
    """
    ChromaDB vector store implementation.

    This class provides a production-ready vector store using ChromaDB as the backend.
    ChromaDB is an embedded database that persists data to disk and requires no external
    services, making it perfect for local development and testing.

    Key features:
    - Local persistence to disk (configurable path)
    - Automatic collection management
    - Metadata filtering support
    - Efficient batch upsert operations
    - Vector similarity search with cosine distance

    Example:
        >>> store = ChromaStore(
        ...     collection_name="documents",
        ...     persist_path="./data/chroma"
        ... )
        >>> records = [
        ...     VectorRecord(id="1", vector=[0.1, 0.2], metadata={"source": "doc1.pdf"}),
        ...     VectorRecord(id="2", vector=[0.3, 0.4], metadata={"source": "doc1.pdf"}),
        ... ]
        >>> result = store.upsert(records)
        >>> print(result.upserted_count)  # 2
        >>>
        >>> results = store.query(vector=[0.15, 0.25], top_k=5)
        >>> for r in results:
        ...     print(f"{r.id}: {r.score:.3f}")
    """

    def __init__(
        self,
        collection_name: str = "default",
        persist_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize ChromaStore.

        Args:
            collection_name: Name of the collection
            persist_path: Path for persistent storage (default: ./data/db/chroma/)
            **kwargs: Additional ChromaDB-specific parameters:
                - distance_metric: Distance metric to use (default: "cosine")
                                 Options: "cosine", "l2", "ip"
        """
        super().__init__(collection_name, persist_path, **kwargs)

        # Set default persist path if not provided
        if self.persist_path is None:
            self.persist_path = "./data/db/chroma"

        # Extract provider-specific config
        self._distance_metric = kwargs.get("distance_metric", "cosine")

        # Initialize ChromaDB client
        self._init_chroma_client()

        logger.debug(
            f"Initialized ChromaStore: collection={collection_name}, "
            f"path={persist_path}, metric={self._distance_metric}"
        )

    def _init_chroma_client(self):
        """
        Initialize ChromaDB client and get/create collection.

        This method sets up the ChromaDB client with persistent storage and
        gets or creates the collection. If the collection doesn't exist,
        it will be created with the configured distance metric.
        """
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_path, exist_ok=True)

        # Initialize ChromaDB client with persistent storage
        self._client = chromadb.PersistentClient(
            path=self.persist_path,
            settings=ChromaSettings(
                anonymized_telemetry=False,  # Disable telemetry
                allow_reset=True  # Allow database reset
            )
        )

        # Get or create collection
        try:
            self._collection = self._client.get_collection(
                name=self.collection_name,
            )
            logger.debug(f"Using existing collection: {self.collection_name}")
        except Exception:
            # Collection doesn't exist, create it
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self._distance_metric}
            )
            logger.debug(f"Created new collection: {self.collection_name}")

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "chroma"

    def upsert(
        self,
        records: List[VectorRecord],
        **kwargs
    ) -> UpsertResult:
        """
        Upsert vector records to ChromaDB.

        This method performs a batch upsert operation. If a record with the same
        ID exists, it will be updated; otherwise, it will be inserted.

        Args:
            records: List of vector records to upsert
            **kwargs: Additional ChromaDB-specific parameters (currently unused)

        Returns:
            UpsertResult containing count of upserted records

        Raises:
            ValueError: If records list is empty
            RuntimeError: If the upsert operation fails
        """
        if not records:
            raise ValueError(
                f"[{self.provider_name}] Cannot upsert empty records list."
            )

        try:
            # Ensure collection exists
            self._ensure_collection()

            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents = []

            for record in records:
                ids.append(record.id)
                embeddings.append(record.vector)

                # ChromaDB metadata handling:
                # - Cannot be None (must be a dict)
                # - Cannot be an empty dict (must have at least one key)
                # - If record.metadata is None or empty, add a dummy key
                if record.metadata:
                    metadatas.append(record.metadata)
                else:
                    # Add dummy metadata to satisfy ChromaDB requirements
                    metadatas.append({"_dummy": "true"})

                # Store payload in documents field for text retrieval
                documents.append(record.payload if record.payload else "")

            # Perform batch upsert
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )

            logger.debug(
                f"[{self.provider_name}] Upserted {len(records)} records "
                f"to collection '{self.collection_name}'"
            )

            return UpsertResult(upserted_count=len(records))

        except Exception as e:
            error_msg = (
                f"[{self.provider_name}] Failed to upsert records: {str(e)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[QueryResult]:
        """
        Query ChromaDB for similar vectors.

        This method performs vector similarity search and returns the top-k most
        similar records, sorted by distance score. ChromaDB returns distances where
        smaller = more similar, so we convert to scores where larger = more similar.

        Args:
            vector: Query vector
            top_k: Maximum number of results to return
            filters: Optional metadata filters (e.g., {"source": "doc1.pdf"})
            **kwargs: Additional ChromaDB-specific parameters (currently unused)

        Returns:
            List of QueryResult objects, sorted by score (descending)

        Raises:
            ValueError: If vector is empty or top_k <= 0
            RuntimeError: If the query operation fails
        """
        if not vector:
            raise ValueError(
                f"[{self.provider_name}] Query vector cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                f"[{self.provider_name}] top_k must be positive, got {top_k}."
            )

        try:
            # Ensure collection exists
            self._ensure_collection()

            # Build query arguments
            query_kwargs = {
                "query_embeddings": [vector],
                "n_results": top_k,
            }

            # Add metadata filters if provided
            if filters:
                # ChromaDB expects filters in a specific format
                query_kwargs["where"] = filters

            # Execute query
            results = self._collection.query(**query_kwargs)

            # Process results
            query_results = []
            if results and results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    # Convert distance to score
                    # ChromaDB returns distances (smaller = more similar)
                    # We convert to scores (larger = more similar)
                    distance = results['distances'][0][i]

                    # For cosine distance, convert to similarity score
                    # cosine_similarity = 1 - cosine_distance
                    score = 1.0 - distance

                    query_results.append(
                        QueryResult(
                            id=doc_id,
                            score=score,
                            metadata=results['metadatas'][0][i] if results['metadatas'] else None,
                            payload=results['documents'][0][i] if results['documents'] else None,
                        )
                    )

            logger.debug(
                f"[{self.provider_name}] Queried collection '{self.collection_name}', "
                f"returned {len(query_results)} results"
            )

            return query_results

        except Exception as e:
            error_msg = (
                f"[{self.provider_name}] Failed to query collection: {str(e)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def delete(self, ids: List[str], **kwargs) -> int:
        """
        Delete records by IDs from ChromaDB.

        Args:
            ids: List of record IDs to delete
            **kwargs: Additional ChromaDB-specific parameters (currently unused)

        Returns:
            Number of records deleted

        Raises:
            RuntimeError: If the delete operation fails
        """
        if not ids:
            return 0

        try:
            # Ensure collection exists
            self._ensure_collection()

            # Get count before deletion
            count_before = self._collection.count()

            # Delete records
            self._collection.delete(ids=ids)

            # Get count after deletion
            count_after = self._collection.count()

            deleted_count = count_before - count_after

            logger.debug(
                f"[{self.provider_name}] Deleted {deleted_count} records "
                f"from collection '{self.collection_name}'"
            )

            return deleted_count

        except Exception as e:
            error_msg = (
                f"[{self.provider_name}] Failed to delete records: {str(e)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def count(self) -> int:
        """
        Get the total number of records in the collection.

        Returns:
            Number of records
        """
        try:
            # Ensure collection exists
            self._ensure_collection()
            return self._collection.count()
        except Exception as e:
            logger.error(f"[{self.provider_name}] Failed to get count: {e}")
            return 0

    def delete_collection(self, **kwargs) -> None:
        """
        Delete the entire collection from ChromaDB.

        This is a destructive operation that cannot be undone. It will
        permanently delete all records in the collection.

        Note: After deletion, the collection reference is invalidated.
        Any further operations will need to reinitialize the store.

        Raises:
            RuntimeError: If the deletion fails
        """
        try:
            self._client.delete_collection(name=self.collection_name)
            self._collection = None  # Invalidate collection reference
            logger.info(
                f"[{self.provider_name}] Deleted collection '{self.collection_name}'"
            )

        except Exception as e:
            error_msg = (
                f"[{self.provider_name}] Failed to delete collection: {str(e)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _ensure_collection(self) -> None:
        """
        Ensure the collection exists, creating it if necessary.

        This is called automatically before operations that need a valid collection.
        """
        if self._collection is None:
            try:
                self._collection = self._client.get_collection(name=self.collection_name)
            except Exception:
                # Collection doesn't exist, create it
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": self._distance_metric}
                )

    def reset_database(self) -> None:
        """
        Reset the entire ChromaDB database.

        This is a destructive operation that will delete ALL collections
        and data. Use with caution!

        Raises:
            RuntimeError: If the reset operation fails
        """
        try:
            # Delete the persist directory
            if os.path.exists(self.persist_path):
                shutil.rmtree(self.persist_path)
                logger.warning(
                    f"[{self.provider_name}] Reset database: deleted {self.persist_path}"
                )

            # Reinitialize
            self._init_chroma_client()

        except Exception as e:
            error_msg = (
                f"[{self.provider_name}] Failed to reset database: {str(e)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def close(self) -> None:
        """
        Close the ChromaDB client and release resources.

        This should be called before deleting the store to ensure
        all file handles are properly closed (especially on Windows).
        """
        try:
            if self._client is not None:
                # Reset collection reference
                self._collection = None
                # Note: ChromaDB doesn't have an explicit close method,
                # but we reset our references to help garbage collection
                self._client = None
                logger.debug(f"[{self.provider_name}] Closed client connection")
        except Exception as e:
            logger.warning(f"[{self.provider_name}] Error during close: {e}")
