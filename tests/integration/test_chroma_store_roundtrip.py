"""
Integration tests for ChromaStore (Task B7.6)

These tests verify the complete upsert→query roundtrip functionality with
ChromaDB as the backend. Tests use temporary directories for persistence
and clean up after completion.

Coverage:
- Factory creation (Acceptance Criteria 1)
- Basic upsert operation
- Vector similarity query
- top_k parameter handling
- Metadata filtering (Acceptance Criteria 3)
- Persistence across restarts
- Delete operations
- Count operations

Author: Modular RAG MCP Server Project
License: MIT
"""

import os
import shutil
import tempfile
from typing import List

import pytest

from src.core.settings import VectorStoreConfig, Settings
from src.libs.vector_store.base_vector_store import VectorRecord, QueryResult
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.libs.vector_store.chroma_store import ChromaStore


# ===== Test Fixtures =====

@pytest.fixture
def temp_persist_dir():
    """
    Create a temporary directory for ChromaDB persistence.

    The directory is automatically cleaned up after each test.
    """
    import gc
    import time

    temp_dir = tempfile.mkdtemp(prefix="chroma_test_")
    yield temp_dir
    # Cleanup: remove the temporary directory
    # Force garbage collection to release file handles (especially on Windows)
    gc.collect()
    time.sleep(0.2)  # Give more time for Windows to release locks

    if os.path.exists(temp_dir):
        def handle_remove_readonly(func, path, exc):
            """Handle Windows file lock issues."""
            import stat
            if not os.access(path, os.W_OK):
                os.chmod(path, stat.S_IWUSR)
                func(path)
            else:
                raise

        try:
            shutil.rmtree(temp_dir, onerror=handle_remove_readonly)
        except Exception:
            # If cleanup still fails, log but don't fail the test
            # The temp directory will be cleaned by the OS eventually
            pass


@pytest.fixture
def chroma_store(temp_persist_dir):
    """
    Create a ChromaStore instance with temporary persistence.

    Uses a unique collection name for each test to avoid conflicts.
    """
    import uuid
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    store = ChromaStore(
        collection_name=collection_name,
        persist_path=temp_persist_dir
    )
    yield store
    # Cleanup: delete the collection first, then close
    try:
        store.delete_collection()
    except Exception:
        pass  # Ignore cleanup errors
    try:
        store.close()
    except Exception:
        pass  # Ignore close errors


def create_test_vector(
    value: float,
    dimension: int = 384
) -> List[float]:
    """
    Helper to create a test vector with a base value.

    Args:
        value: Base value for the vector
        dimension: Vector dimension (default: 384 for embedding models)

    Returns:
        List of floats representing the vector
    """
    return [value + i * 0.01 for i in range(dimension)]


# ===== ChromaStore Integration Tests =====

class TestChromaStoreRoundtrip:
    """Test complete upsert→query roundtrip with ChromaStore."""

    def test_factory_creates_chroma_store(self):
        """Test that VectorStoreFactory can create ChromaStore (Acceptance Criteria 1)."""
        import tempfile
        import gc
        import time

        # Use mkdtemp instead of TemporaryDirectory to have more control over cleanup
        temp_dir = tempfile.mkdtemp(prefix="chroma_factory_test_")
        try:
            config = VectorStoreConfig(
                backend="chroma",
                persist_path=temp_dir
            )
            settings = Settings(
                llm=self._create_llm_config(),
                embedding=self._create_embedding_config(),
                vector_store=config,
                retrieval=self._create_retrieval_config(),
                rerank=self._create_rerank_config(),
                evaluation=self._create_evaluation_config(),
                observability=self._create_observability_config(),
                dashboard=self._create_dashboard_config(),
            )

            store = VectorStoreFactory.create(settings)

            assert isinstance(store, ChromaStore)
            assert store.provider_name == "chroma"
            assert store.collection_name == "default"

            # Close the store to release file handles
            store.close()

        finally:
            # Cleanup
            gc.collect()
            time.sleep(0.2)
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass  # Ignore cleanup errors

    def test_basic_upsert_and_query(self, chroma_store):
        """Test basic upsert and query roundtrip (Acceptance Criteria 2)."""
        # Create test records
        records = [
            VectorRecord(
                id="doc1_chunk1",
                vector=create_test_vector(0.1),
                metadata={"source": "doc1.pdf", "page": 1},
                payload="This is the first chunk of document 1."
            ),
            VectorRecord(
                id="doc1_chunk2",
                vector=create_test_vector(0.2),
                metadata={"source": "doc1.pdf", "page": 2},
                payload="This is the second chunk of document 1."
            ),
            VectorRecord(
                id="doc2_chunk1",
                vector=create_test_vector(0.5),
                metadata={"source": "doc2.pdf", "page": 1},
                payload="This is the first chunk of document 2."
            ),
        ]

        # Upsert records
        upsert_result = chroma_store.upsert(records)
        assert upsert_result.upserted_count == 3

        # Query with vector similar to doc1_chunk1 (0.1)
        query_vector = create_test_vector(0.12)
        results = chroma_store.query(vector=query_vector, top_k=10)

        # Verify results
        assert len(results) == 3
        # Results should be sorted by score (descending)
        # Most similar should be doc1_chunk1 (vector 0.1)
        assert results[0].id == "doc1_chunk1"
        assert results[0].score > results[1].score  # Score should be descending
        assert results[0].metadata["source"] == "doc1.pdf"
        assert results[0].payload == "This is the first chunk of document 1."

    def test_top_k_parameter(self, chroma_store):
        """Test that top_k parameter limits results correctly (Acceptance Criteria 3)."""
        # Create 10 test records
        records = [
            VectorRecord(
                id=f"doc{i}",
                vector=create_test_vector(i * 0.1),
                metadata={"index": i},
                payload=f"Document {i}"
            )
            for i in range(10)
        ]

        chroma_store.upsert(records)

        # Query with top_k=5
        query_vector = create_test_vector(0.5)
        results = chroma_store.query(vector=query_vector, top_k=5)

        # Should return exactly 5 results
        assert len(results) == 5

        # All results should have valid scores
        for result in results:
            assert result.score >= 0.0
            assert result.score <= 1.01  # Allow small floating point errors

    def test_metadata_filters(self, chroma_store):
        """Test metadata filtering functionality (Acceptance Criteria 3)."""
        # Create records with different metadata
        records = [
            VectorRecord(
                id="doc1",
                vector=create_test_vector(0.1),
                metadata={"source": "report.pdf", "category": "finance"},
                payload="Financial report"
            ),
            VectorRecord(
                id="doc2",
                vector=create_test_vector(0.2),
                metadata={"source": "email.txt", "category": "communication"},
                payload="Email message"
            ),
            VectorRecord(
                id="doc3",
                vector=create_test_vector(0.3),
                metadata={"source": "report.pdf", "category": "hr"},
                payload="HR report"
            ),
        ]

        chroma_store.upsert(records)

        # Query without filters - should return all 3
        query_vector = create_test_vector(0.2)
        results_no_filter = chroma_store.query(vector=query_vector, top_k=10)
        assert len(results_no_filter) == 3

        # Query with source filter
        results_with_filter = chroma_store.query(
            vector=query_vector,
            top_k=10,
            filters={"source": "report.pdf"}
        )
        assert len(results_with_filter) == 2
        assert all(r.metadata["source"] == "report.pdf" for r in results_with_filter)

        # Query with category filter
        results_category = chroma_store.query(
            vector=query_vector,
            top_k=10,
            filters={"category": "finance"}
        )
        assert len(results_category) == 1
        assert results_category[0].id == "doc1"

    def test_upsert_updates_existing_records(self, chroma_store):
        """Test that upsert updates existing records with the same ID."""
        # Initial upsert
        record = VectorRecord(
            id="doc1",
            vector=create_test_vector(0.1),
            metadata={"version": 1},
            payload="Original content"
        )
        chroma_store.upsert([record])
        assert chroma_store.count() == 1

        # Update the record
        updated_record = VectorRecord(
            id="doc1",  # Same ID
            vector=create_test_vector(0.9),  # Different vector
            metadata={"version": 2},
            payload="Updated content"
        )
        chroma_store.upsert([updated_record])

        # Should still have only 1 record
        assert chroma_store.count() == 1

        # Query should return updated content
        results = chroma_store.query(vector=create_test_vector(0.9), top_k=1)
        assert len(results) == 1
        assert results[0].id == "doc1"
        assert results[0].metadata["version"] == 2
        assert results[0].payload == "Updated content"

    def test_delete_records(self, chroma_store):
        """Test deleting records by IDs."""
        # Create test records
        records = [
            VectorRecord(
                id=f"doc{i}",
                vector=create_test_vector(i * 0.1),
                metadata={"index": i}
            )
            for i in range(5)
        ]
        chroma_store.upsert(records)
        assert chroma_store.count() == 5

        # Delete specific records
        deleted_count = chroma_store.delete(ids=["doc1", "doc3"])
        assert deleted_count == 2
        assert chroma_store.count() == 3

        # Verify deleted records are not returned in queries
        results = chroma_store.query(vector=create_test_vector(0.1), top_k=10)
        result_ids = [r.id for r in results]
        assert "doc1" not in result_ids
        assert "doc3" not in result_ids

    def test_count_operation(self, chroma_store):
        """Test count operation."""
        # Initially empty
        assert chroma_store.count() == 0

        # Add records
        records = [
            VectorRecord(id=f"doc{i}", vector=create_test_vector(i * 0.1))
            for i in range(7)
        ]
        chroma_store.upsert(records)
        assert chroma_store.count() == 7

        # Delete some records
        chroma_store.delete(ids=["doc2", "doc4"])
        assert chroma_store.count() == 5

    def test_persistence_across_restarts(self, temp_persist_dir):
        """Test that data persists across ChromaStore restarts."""
        import uuid

        collection_name = f"test_persist_{uuid.uuid4().hex[:8]}"

        # Create first store instance and add data
        store1 = ChromaStore(
            collection_name=collection_name,
            persist_path=temp_persist_dir
        )
        records = [
            VectorRecord(
                id="persistent_doc",
                vector=create_test_vector(0.5),
                metadata={"source": "persistent.pdf"},
                payload="This data should persist"
            )
        ]
        store1.upsert(records)
        count1 = store1.count()

        # Create second store instance with same collection
        store2 = ChromaStore(
            collection_name=collection_name,
            persist_path=temp_persist_dir
        )
        count2 = store2.count()

        # Data should have persisted
        assert count1 == count2 == 1

        # Verify data integrity
        results = store2.query(vector=create_test_vector(0.5), top_k=1)
        assert len(results) == 1
        assert results[0].id == "persistent_doc"
        assert results[0].payload == "This data should persist"

        # Cleanup
        store2.delete_collection()

    def test_delete_collection(self, chroma_store):
        """Test deleting entire collection."""
        # Add some data
        records = [
            VectorRecord(id=f"doc{i}", vector=create_test_vector(i * 0.1))
            for i in range(5)
        ]
        chroma_store.upsert(records)
        assert chroma_store.count() == 5

        # Delete collection
        chroma_store.delete_collection()

        # Collection reference should be invalidated
        # Count will auto-recreate the collection
        # After deletion and auto-recreation, count should be 0
        assert chroma_store.count() == 0

    def test_query_returns_correct_ordering(self, chroma_store):
        """Test that query results are sorted by score (most similar first)."""
        # Create records with varying similarity
        query_center = 0.5
        records = [
            VectorRecord(
                id="similar",
                vector=create_test_vector(query_center),  # Very similar to query
                payload="Similar document"
            ),
            VectorRecord(
                id="somewhat_similar",
                vector=create_test_vector(query_center - 0.2),  # Somewhat similar
                payload="Somewhat similar document"
            ),
            VectorRecord(
                id="dissimilar",
                vector=create_test_vector(0.0),  # Not similar
                payload="Dissimilar document"
            ),
        ]

        chroma_store.upsert(records)

        # Query near the center
        query_vector = create_test_vector(query_center)
        results = chroma_store.query(vector=query_vector, top_k=10)

        # Results should be in descending score order
        assert len(results) >= 2
        for i in range(len(results) - 1):
            assert results[i].score >= results[i+1].score, \
                f"Results not sorted: {results[i].score} < {results[i+1].score}"

        # Most similar should be first
        assert results[0].id == "similar"

    def test_empty_query_vector_raises_error(self, chroma_store):
        """Test that empty query vector raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            chroma_store.query(vector=[], top_k=5)

        assert "cannot be empty" in str(exc_info.value).lower()
        assert "chroma" in str(exc_info.value).lower()

    def test_invalid_top_k_raises_error(self, chroma_store):
        """Test that invalid top_k raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            chroma_store.query(vector=create_test_vector(0.1), top_k=0)

        assert "must be positive" in str(exc_info.value).lower()

    def test_empty_upsert_list_raises_error(self, chroma_store):
        """Test that empty upsert list raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            chroma_store.upsert([])

        assert "cannot upsert empty" in str(exc_info.value).lower()


# ===== Helper Methods for Creating Settings =====

@staticmethod
def _create_llm_config():
    from src.core.settings import LLMConfig
    return LLMConfig(provider="fake", model="fake-model")

@staticmethod
def _create_embedding_config():
    from src.core.settings import EmbeddingConfig
    return EmbeddingConfig(provider="fake", model="fake-model", dimension=384)

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
    from src.core.settings import RerankConfig
    return RerankConfig()

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


# Add helper methods to test class
TestChromaStoreRoundtrip._create_llm_config = _create_llm_config
TestChromaStoreRoundtrip._create_embedding_config = _create_embedding_config
TestChromaStoreRoundtrip._create_vector_store_config = _create_vector_store_config
TestChromaStoreRoundtrip._create_retrieval_config = _create_retrieval_config
TestChromaStoreRoundtrip._create_rerank_config = _create_rerank_config
TestChromaStoreRoundtrip._create_evaluation_config = _create_evaluation_config
TestChromaStoreRoundtrip._create_observability_config = _create_observability_config
TestChromaStoreRoundtrip._create_dashboard_config = _create_dashboard_config
