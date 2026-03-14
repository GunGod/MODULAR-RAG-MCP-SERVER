"""
Unit tests for VectorStore contract and VectorStoreFactory (Task B4)

These tests verify that:
1. The BaseVectorStore interface is correctly defined
2. VectorStoreFactory can create instances based on settings
3. Factory routing logic works correctly
4. FakeVectorStore implements the contract correctly
5. Contract tests constrain input/output shapes

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import List

import pytest

from src.core.settings import (
    Settings,
    VectorStoreConfig,
    LLMConfig,
    EmbeddingConfig,
    RetrievalConfig,
    RerankConfig,
    EvaluationConfig,
    ObservabilityConfig,
    DashboardConfig,
)
from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult,
    UpsertResult,
)
from src.libs.vector_store.fake_vector_store import FakeVectorStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


class TestVectorRecord:
    """Test the VectorRecord dataclass."""

    def test_create_record(self):
        """Test creating a vector record."""
        record = VectorRecord(
            id="test-1",
            vector=[0.1, 0.2, 0.3],
            metadata={"source": "doc1.pdf"},
            payload="Sample text"
        )
        assert record.id == "test-1"
        assert record.vector == [0.1, 0.2, 0.3]
        assert record.metadata == {"source": "doc1.pdf"}
        assert record.payload == "Sample text"

    def test_create_record_without_optional_fields(self):
        """Test creating a record with only required fields."""
        record = VectorRecord(id="test-2", vector=[0.4, 0.5])
        assert record.id == "test-2"
        assert record.vector == [0.4, 0.5]
        assert record.metadata is None
        assert record.payload is None


class TestQueryResult:
    """Test the QueryResult dataclass."""

    def test_create_result(self):
        """Test creating a query result."""
        result = QueryResult(
            id="doc-1",
            score=0.95,
            metadata={"page": 1},
            payload="Retrieved text"
        )
        assert result.id == "doc-1"
        assert result.score == 0.95
        assert result.metadata == {"page": 1}
        assert result.payload == "Retrieved text"


class TestUpsertResult:
    """Test the UpsertResult dataclass."""

    def test_create_result(self):
        """Test creating an upsert result."""
        result = UpsertResult(upserted_count=10, deleted_count=2)
        assert result.upserted_count == 10
        assert result.deleted_count == 2

    def test_default_deleted_count(self):
        """Test default deleted_count is zero."""
        result = UpsertResult(upserted_count=5)
        assert result.upserted_count == 5
        assert result.deleted_count == 0


class TestBaseVectorStore:
    """Test the BaseVectorStore abstract interface."""

    def test_base_vector_store_cannot_be_instantiated(self):
        """Test that BaseVectorStore cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseVectorStore(collection_name="test")

    def test_base_vector_store_subclass_must_implement_upsert(self):
        """Test that subclasses must implement upsert method."""

        class IncompleteStore(BaseVectorStore):
            def __init__(self):
                super().__init__(collection_name="incomplete")

            @property
            def provider_name(self) -> str:
                return "incomplete"

            # Missing upsert(), query(), delete(), count(), delete_collection()

        with pytest.raises(TypeError):
            IncompleteStore()

    def test_base_vector_store_subclass_must_implement_query(self):
        """Test that subclasses must implement query method."""

        class IncompleteStore(BaseVectorStore):
            def __init__(self):
                super().__init__(collection_name="incomplete")

            def upsert(self, records, **kwargs):
                return UpsertResult(upserted_count=len(records))

            @property
            def provider_name(self) -> str:
                return "incomplete"

            # Missing query(), delete(), count(), delete_collection()

        with pytest.raises(TypeError):
            IncompleteStore()

    def test_base_vector_store_initialization(self):
        """Test BaseVectorStore initialization with parameters."""
        store = FakeVectorStore(collection_name="test_collection")
        assert store.collection_name == "test_collection"
        assert store.persist_path is None

    def test_base_vector_store_repr(self):
        """Test string representation of vector store."""
        store = FakeVectorStore(collection_name="my_collection")
        repr_str = repr(store)
        assert "FakeVectorStore" in repr_str
        assert "my_collection" in repr_str
        assert "fake" in repr_str


class TestFakeVectorStore:
    """Test the FakeVectorStore implementation."""

    def test_upsert_single_record(self):
        """Test upserting a single record."""
        store = FakeVectorStore()
        record = VectorRecord(id="1", vector=[0.1, 0.2], metadata={"source": "doc1"})
        result = store.upsert([record])

        assert result.upserted_count == 1
        assert result.deleted_count == 0
        assert store.count() == 1

    def test_upsert_multiple_records(self):
        """Test upserting multiple records."""
        store = FakeVectorStore()
        records = [
            VectorRecord(id="1", vector=[0.1, 0.2]),
            VectorRecord(id="2", vector=[0.3, 0.4]),
            VectorRecord(id="3", vector=[0.5, 0.6]),
        ]
        result = store.upsert(records)

        assert result.upserted_count == 3
        assert store.count() == 3

    def test_upsert_empty_list_raises_error(self):
        """Test that empty records list raises ValueError."""
        store = FakeVectorStore()
        with pytest.raises(ValueError) as exc_info:
            store.upsert([])
        assert "cannot be empty" in str(exc_info.value).lower()

    def test_upsert_updates_existing_record(self):
        """Test that upsert updates existing records."""
        store = FakeVectorStore()
        record1 = VectorRecord(id="1", vector=[0.1, 0.2], metadata={"v": 1})
        store.upsert([record1])

        # Update the same ID
        record2 = VectorRecord(id="1", vector=[0.3, 0.4], metadata={"v": 2})
        result = store.upsert([record2])

        assert result.upserted_count == 1
        assert store.count() == 1  # Still only 1 record

    def test_query_returns_sorted_results(self):
        """Test that query returns results sorted by score (descending)."""
        store = FakeVectorStore()
        records = [
            VectorRecord(id="1", vector=[1.0, 0.0]),
            VectorRecord(id="2", vector=[0.0, 1.0]),
            VectorRecord(id="3", vector=[1.0, 1.0]),
        ]
        store.upsert(records)

        # Query with [1.0, 1.0] should match record 3 best, then 1, then 2
        results = store.query(vector=[1.0, 1.0], top_k=3)

        assert len(results) == 3
        assert results[0].id == "3"  # Most similar
        assert results[0].score >= results[1].score
        assert results[1].score >= results[2].score

    def test_query_respects_top_k(self):
        """Test that query respects the top_k parameter."""
        store = FakeVectorStore()
        records = [
            VectorRecord(id=str(i), vector=[i, i]) for i in range(10)
        ]
        store.upsert(records)

        results = store.query(vector=[5, 5], top_k=3)
        assert len(results) == 3

    def test_query_with_filters(self):
        """Test query with metadata filters."""
        store = FakeVectorStore()
        records = [
            VectorRecord(id="1", vector=[1.0, 0.0], metadata={"category": "A"}),
            VectorRecord(id="2", vector=[1.0, 0.0], metadata={"category": "B"}),
            VectorRecord(id="3", vector=[1.0, 0.0], metadata={"category": "A"}),
        ]
        store.upsert(records)

        results = store.query(
            vector=[1.0, 0.0],
            top_k=10,
            filters={"category": "A"}
        )

        assert len(results) == 2
        assert all(r.metadata.get("category") == "A" for r in results)

    def test_query_empty_vector_raises_error(self):
        """Test that empty query vector raises ValueError."""
        store = FakeVectorStore()
        with pytest.raises(ValueError) as exc_info:
            store.query(vector=[], top_k=10)
        assert "cannot be empty" in str(exc_info.value).lower()

    def test_query_invalid_top_k_raises_error(self):
        """Test that top_k <= 0 raises ValueError."""
        store = FakeVectorStore()
        with pytest.raises(ValueError) as exc_info:
            store.query(vector=[0.1, 0.2], top_k=0)
        assert "must be positive" in str(exc_info.value).lower()

    def test_delete_by_ids(self):
        """Test deleting records by IDs."""
        store = FakeVectorStore()
        records = [
            VectorRecord(id="1", vector=[0.1, 0.2]),
            VectorRecord(id="2", vector=[0.3, 0.4]),
            VectorRecord(id="3", vector=[0.5, 0.6]),
        ]
        store.upsert(records)

        deleted_count = store.delete(ids=["1", "3"])
        assert deleted_count == 2
        assert store.count() == 1

    def test_delete_non_existent_ids(self):
        """Test deleting non-existent IDs returns 0."""
        store = FakeVectorStore()
        record = VectorRecord(id="1", vector=[0.1, 0.2])
        store.upsert([record])

        deleted_count = store.delete(ids=["999"])
        assert deleted_count == 0
        assert store.count() == 1

    def test_count_returns_correct_number(self):
        """Test that count returns the correct number of records."""
        store = FakeVectorStore()
        assert store.count() == 0

        store.upsert([VectorRecord(id=str(i), vector=[0.0, 0.0]) for i in range(5)])
        assert store.count() == 5

    def test_delete_collection_clears_all_records(self):
        """Test that delete_collection clears all records."""
        store = FakeVectorStore()
        records = [
            VectorRecord(id=str(i), vector=[0.0, 0.0]) for i in range(10)
        ]
        store.upsert(records)
        assert store.count() == 10

        store.delete_collection()
        assert store.count() == 0

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity for identical vectors."""
        store = FakeVectorStore()
        record = VectorRecord(id="1", vector=[1.0, 2.0, 3.0])
        store.upsert([record])

        results = store.query(vector=[1.0, 2.0, 3.0], top_k=1)
        assert results[0].score == pytest.approx(1.0)  # Perfect match

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity for orthogonal vectors."""
        store = FakeVectorStore()
        record = VectorRecord(id="1", vector=[1.0, 0.0])
        store.upsert([record])

        results = store.query(vector=[0.0, 1.0], top_k=1)
        assert results[0].score == pytest.approx(0.0)  # Orthogonal

    def test_get_upsert_count(self):
        """Test tracking total upserted records."""
        store = FakeVectorStore()
        assert store.get_upsert_count() == 0

        store.upsert([VectorRecord(id="1", vector=[0.1, 0.2])])
        assert store.get_upsert_count() == 1

        store.upsert([
            VectorRecord(id="2", vector=[0.3, 0.4]),
            VectorRecord(id="3", vector=[0.5, 0.6]),
        ])
        assert store.get_upsert_count() == 3

    def test_get_query_count(self):
        """Test tracking query count."""
        store = FakeVectorStore()
        assert store.get_query_count() == 0

        store.query(vector=[0.1, 0.2], top_k=5)
        assert store.get_query_count() == 1

        store.query(vector=[0.3, 0.4], top_k=5)
        assert store.get_query_count() == 2

    def test_provider_name(self):
        """Test FakeVectorStore provider_name property."""
        store = FakeVectorStore()
        assert store.provider_name == "fake"


class TestVectorStoreFactory:
    """Test the VectorStoreFactory class."""

    def test_factory_create_with_fake_provider(self):
        """Test creating FakeVectorStore through factory."""
        settings = self._create_settings(backend="fake")
        store = VectorStoreFactory.create(settings)

        assert isinstance(store, FakeVectorStore)
        assert store.provider_name == "fake"

    def test_factory_create_with_default_collection(self):
        """Test factory creates store with default collection name."""
        settings = self._create_settings(backend="fake")
        store = VectorStoreFactory.create(settings)

        assert store.collection_name == "default"

    def test_factory_create_with_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises ValueError."""
        settings = self._create_settings(backend="unsupported_backend")

        with pytest.raises(ValueError) as exc_info:
            VectorStoreFactory.create(settings)
        assert "Unsupported vector store provider" in str(exc_info.value)
        assert "unsupported_backend" in str(exc_info.value)

    def test_factory_list_providers(self):
        """Test getting list of available providers."""
        providers = VectorStoreFactory.list_providers()
        assert isinstance(providers, list)
        assert "fake" in providers

    def test_factory_register_provider(self):
        """Test registering a custom provider."""
        # Create a custom vector store class
        class CustomStore(BaseVectorStore):
            def __init__(self, collection_name="custom", **kwargs):
                super().__init__(collection_name, **kwargs)
                self._records = {}

            def upsert(self, records, **kwargs):
                for r in records:
                    self._records[r.id] = r
                return UpsertResult(upserted_count=len(records))

            def query(self, vector, top_k=10, filters=None, **kwargs):
                return []

            def delete(self, ids, **kwargs):
                return len(ids)

            def count(self):
                return len(self._records)

            def delete_collection(self, **kwargs):
                self._records.clear()

            @property
            def provider_name(self):
                return "custom"

        # Register the provider
        VectorStoreFactory.register_provider("custom", CustomStore)

        # Verify it's in the list
        providers = VectorStoreFactory.list_providers()
        assert "custom" in providers

    def test_factory_register_non_vector_store_class_raises_error(self):
        """Test that registering non-BaseVectorStore class raises TypeError."""
        class NotAVectorStore:
            pass

        with pytest.raises(TypeError) as exc_info:
            VectorStoreFactory.register_provider("invalid", NotAVectorStore)
        assert "BaseVectorStore" in str(exc_info.value)

    def _create_settings(self, backend: str = "fake") -> Settings:
        """Helper method to create test Settings object."""
        return Settings(
            llm=LLMConfig(provider="fake", model="fake-model"),
            embedding=EmbeddingConfig(provider="fake", model="fake-embedding"),
            vector_store=VectorStoreConfig(backend=backend),
            retrieval=RetrievalConfig(),
            rerank=RerankConfig(),
            evaluation=EvaluationConfig(),
            observability=ObservabilityConfig(),
            dashboard=DashboardConfig(),
        )


class TestVectorStoreContract:
    """Contract tests for vector store behavior."""

    def test_contract_upsert_returns_result_with_counts(self):
        """Test contract: upsert returns UpsertResult with counts."""
        store = FakeVectorStore()
        records = [VectorRecord(id=str(i), vector=[0.1, 0.2]) for i in range(5)]

        result = store.upsert(records)

        assert isinstance(result, UpsertResult)
        assert isinstance(result.upserted_count, int)
        assert isinstance(result.deleted_count, int)
        assert result.upserted_count == 5

    def test_contract_query_returns_list_of_query_results(self):
        """Test contract: query returns List[QueryResult]."""
        store = FakeVectorStore()
        store.upsert([VectorRecord(id="1", vector=[0.1, 0.2])])

        results = store.query(vector=[0.1, 0.2], top_k=10)

        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, QueryResult)
            assert isinstance(result.id, str)
            assert isinstance(result.score, float)
            assert result.metadata is None or isinstance(result.metadata, dict)

    def test_contract_query_results_sorted_by_score(self):
        """Test contract: query results are sorted by score descending."""
        store = FakeVectorStore()
        records = [
            VectorRecord(id="low", vector=[0.0, 1.0]),
            VectorRecord(id="high", vector=[1.0, 1.0]),
            VectorRecord(id="mid", vector=[0.5, 0.5]),
        ]
        store.upsert(records)

        results = store.query(vector=[1.0, 1.0], top_k=10)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_contract_count_returns_int(self):
        """Test contract: count returns int."""
        store = FakeVectorStore()
        count = store.count()
        assert isinstance(count, int)

    def test_contract_delete_returns_int_count(self):
        """Test contract: delete returns int count of deleted records."""
        store = FakeVectorStore()
        store.upsert([VectorRecord(id="1", vector=[0.1, 0.2])])

        deleted = store.delete(ids=["1"])
        assert isinstance(deleted, int)
        assert deleted == 1

    def test_contract_metadata_filtering_works(self):
        """Test contract: query with filters returns only matching records."""
        store = FakeVectorStore()
        records = [
            VectorRecord(id="1", vector=[1.0, 1.0], metadata={"type": "A"}),
            VectorRecord(id="2", vector=[1.0, 1.0], metadata={"type": "B"}),
            VectorRecord(id="3", vector=[1.0, 1.0], metadata={"type": "A"}),
        ]
        store.upsert(records)

        results = store.query(vector=[1.0, 1.0], top_k=10, filters={"type": "A"})

        assert len(results) == 2
        assert all(r.metadata["type"] == "A" for r in results)
