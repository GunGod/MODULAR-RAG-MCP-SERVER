"""
Unit tests for OpenAI & Azure Embedding providers (Task B7.3)

These tests verify that:
1. EmbeddingFactory can create OpenAI and Azure providers
2. Providers handle input validation correctly (empty, invalid inputs)
3. Batch processing works correctly
4. Error messages are clear and include provider information
5. Azure reuses OpenAI's core logic (inheritance)
6. HTTP requests are properly formatted (mocked, no real network calls)

Author: Modular RAG MCP Server Project
License: MIT
"""

from unittest.mock import Mock, patch
from typing import Any

import pytest
import httpx

from src.core.settings import EmbeddingConfig, Settings
from src.libs.embedding.base_embedding import EmbeddingResult
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.openai_embedding import OpenAIEmbedding
from src.libs.embedding.azure_embedding import AzureEmbedding


# ===== Test Fixtures =====

def create_embedding_config(
    provider: str = "openai",
    model: str = "text-embedding-3-small",
    dimension: int = 1536,
    **kwargs
) -> EmbeddingConfig:
    """Helper to create EmbeddingConfig for testing."""
    return EmbeddingConfig(
        provider=provider,
        model=model,
        dimension=dimension,
        **kwargs
    )


def create_mock_embedding_response(
    embeddings: list[list[float]],
    model: str = "text-embedding-3-small",
    total_tokens: int = 100
) -> dict[str, Any]:
    """Helper to create a mock embedding API response."""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": embeddings[i]
            }
            for i in range(len(embeddings))
        ],
        "model": model,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    }


# ===== OpenAI Embedding Tests =====

class TestOpenAIEmbedding:
    """Test OpenAI Embedding provider."""

    def test_factory_creates_openai_embedding(self):
        """Test that EmbeddingFactory can create OpenAI Embedding (Acceptance Criteria 1)."""
        config = create_embedding_config(
            provider="openai",
            model="text-embedding-3-small",
            dimension=1536,
            api_key="test-api-key"  # Add API key
        )
        settings = Settings(
            llm=self._create_llm_config(),
            embedding=config,
            vector_store=self._create_vector_store_config(),
            retrieval=self._create_retrieval_config(),
            rerank=self._create_rerank_config(),
            evaluation=self._create_evaluation_config(),
            observability=self._create_observability_config(),
            dashboard=self._create_dashboard_config(),
        )

        embedding = EmbeddingFactory.create(settings)

        assert isinstance(embedding, OpenAIEmbedding)
        assert embedding.provider_name == "openai"
        assert embedding.model == "text-embedding-3-small"
        assert embedding.dimension == 1536

    def test_openai_embedding_requires_api_key(self):
        """Test that OpenAI Embedding raises error without API key."""
        with pytest.raises(ValueError) as exc_info:
            OpenAIEmbedding(
                model="text-embedding-3-small",
                dimension=1536,
                api_key=None
            )

        assert "API key is required" in str(exc_info.value)
        assert "openai" in str(exc_info.value).lower()

    def test_openai_embed_validates_texts_not_empty(self):
        """Test that OpenAI Embedding validates texts list is not empty (Acceptance Criteria 3)."""
        embedding = OpenAIEmbedding(
            model="text-embedding-3-small",
            dimension=1536,
            api_key="test-key"
        )

        with pytest.raises(ValueError) as exc_info:
            embedding.embed([])

        assert "cannot be empty" in str(exc_info.value)
        assert "openai" in str(exc_info.value).lower()

    def test_openai_embed_validates_text_types(self):
        """Test that OpenAI Embedding validates text types."""
        embedding = OpenAIEmbedding(
            model="text-embedding-3-small",
            dimension=1536,
            api_key="test-key"
        )

        with pytest.raises(ValueError) as exc_info:
            embedding.embed(["valid", 123])  # Invalid type

        assert "must be strings" in str(exc_info.value)
        assert "openai" in str(exc_info.value).lower()

    @patch('src.libs.embedding.openai_embedding.httpx.Client')
    def test_openai_embed_single_text(self, mock_client_class):
        """Test embedding a single text."""
        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = create_mock_embedding_response(
            embeddings=[[0.1, 0.2, 0.3]],
            model="text-embedding-3-small"
        )
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Create embedding and make request
        embedding = OpenAIEmbedding(
            model="text-embedding-3-small",
            dimension=3,
            api_key="test-key"
        )
        result = embedding.embed(["Hello, world!"])

        # Verify result
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 3
        assert result.model == "text-embedding-3-small"
        assert result.provider == "openai"
        assert result.dimension == 3

    @patch('src.libs.embedding.openai_embedding.httpx.Client')
    def test_openai_embed_batch_texts(self, mock_client_class):
        """Test embedding multiple texts in batch."""
        # Setup mock
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]
        mock_response = Mock()
        mock_response.json.return_value = create_mock_embedding_response(
            embeddings=embeddings,
            total_tokens=30
        )
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Create embedding and make request
        embedding = OpenAIEmbedding(
            model="text-embedding-3-small",
            dimension=3,
            api_key="test-key"
        )
        result = embedding.embed(["Text 1", "Text 2", "Text 3"])

        # Verify result
        assert len(result.embeddings) == 3
        assert result.embeddings == embeddings
        assert result.usage["total_tokens"] == 30

    @patch('src.libs.embedding.openai_embedding.httpx.Client')
    def test_openai_embed_handles_batch_size(self, mock_client_class):
        """Test that batch_size is respected (Acceptance Criteria 3)."""
        # Setup mock to return different results for each call
        embeddings_batch1 = [[0.1, 0.2], [0.3, 0.4]]
        embeddings_batch2 = [[0.5, 0.6]]

        mock_response1 = Mock()
        mock_response1.json.return_value = create_mock_embedding_response(
            embeddings=embeddings_batch1,
            total_tokens=20
        )

        mock_response2 = Mock()
        mock_response2.json.return_value = create_mock_embedding_response(
            embeddings=embeddings_batch2,
            total_tokens=10
        )

        mock_client = Mock()
        mock_client.post.side_effect = [mock_response1, mock_response2]
        mock_client_class.return_value = mock_client

        # Create embedding with batch_size=2
        embedding = OpenAIEmbedding(
            model="text-embedding-3-small",
            dimension=2,
            api_key="test-key",
            batch_size=2
        )
        result = embedding.embed(["Text 1", "Text 2", "Text 3"])

        # Verify batch processing
        assert mock_client.post.call_count == 2  # 2 batches
        assert len(result.embeddings) == 3
        assert result.usage["total_tokens"] == 30

    @patch('src.libs.embedding.openai_embedding.httpx.Client')
    def test_openai_http_error_includes_provider_info(self, mock_client_class):
        """Test that HTTP errors include clear provider information."""
        # Setup mock to raise HTTP error
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": {"message": "Invalid API key"}
        }
        mock_client = Mock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=Mock(), response=mock_response
        )
        mock_client_class.return_value = mock_client

        # Create embedding and make request
        embedding = OpenAIEmbedding(
            model="text-embedding-3-small",
            dimension=1536,
            api_key="invalid-key"
        )

        # Verify error message is clear
        with pytest.raises(RuntimeError) as exc_info:
            embedding.embed(["Hello"])

        error_msg = str(exc_info.value)
        assert "openai" in error_msg.lower()
        assert "401" in error_msg
        assert "text-embedding-3-small" in error_msg


# ===== Azure Embedding Tests =====

class TestAzureEmbedding:
    """Test Azure OpenAI Embedding provider."""

    def test_factory_creates_azure_embedding(self):
        """Test that EmbeddingFactory can create Azure Embedding (Acceptance Criteria 2)."""
        config = create_embedding_config(
            provider="azure",
            model="text-embedding-3-small",
            dimension=1536,
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )
        settings = Settings(
            llm=self._create_llm_config(),
            embedding=config,
            vector_store=self._create_vector_store_config(),
            retrieval=self._create_retrieval_config(),
            rerank=self._create_rerank_config(),
            evaluation=self._create_evaluation_config(),
            observability=self._create_observability_config(),
            dashboard=self._create_dashboard_config(),
        )

        embedding = EmbeddingFactory.create(settings)

        assert isinstance(embedding, AzureEmbedding)
        assert embedding.provider_name == "azure"
        assert embedding.model == "text-embedding-3-small"

    def test_azure_embedding_requires_endpoint(self):
        """Test that Azure Embedding raises error without endpoint."""
        with pytest.raises(ValueError) as exc_info:
            AzureEmbedding(
                model="text-embedding-3-small",
                dimension=1536,
                api_key="test-key",
                azure_endpoint=None
            )

        assert "azure_endpoint is required" in str(exc_info.value)
        assert "azure" in str(exc_info.value).lower()

    def test_azure_embedding_requires_api_key(self):
        """Test that Azure Embedding raises error without API key."""
        with pytest.raises(ValueError) as exc_info:
            AzureEmbedding(
                model="text-embedding-3-small",
                dimension=1536,
                api_key=None,
                azure_endpoint="https://test.openai.azure.com"
            )

        assert "API key is required" in str(exc_info.value)
        assert "azure" in str(exc_info.value).lower()

    @patch('src.libs.embedding.azure_embedding.httpx.Client')
    def test_azure_embed_success(self, mock_client_class):
        """Test successful Azure embedding (Acceptance Criteria 4: reuses OpenAI logic)."""
        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = create_mock_embedding_response(
            embeddings=[[0.1, 0.2, 0.3]],
            model="text-embedding-3-small"
        )
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Create embedding and make request
        embedding = AzureEmbedding(
            model="text-embedding-3-small",
            dimension=3,
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )
        result = embedding.embed(["Hello, Azure!"])

        # Verify result
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 3
        assert result.provider == "azure"

        # Verify Azure-specific URL was used
        args, kwargs = mock_client.post.call_args
        assert "test.openai.azure.com" in args[0]
        assert "text-embedding-3-small" in args[0]
        assert "api-version" in args[0]

    @patch('src.libs.embedding.azure_embedding.httpx.Client')
    def test_azure_http_error_includes_azure_info(self, mock_client_class):
        """Test that HTTP errors include Azure-specific information."""
        # Setup mock to raise HTTP error
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": {"message": "Invalid API key"}
        }
        mock_client = Mock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=Mock(), response=mock_response
        )
        mock_client_class.return_value = mock_client

        # Create embedding and make request
        embedding = AzureEmbedding(
            model="text-embedding-3-small",
            dimension=1536,
            api_key="invalid-key",
            azure_endpoint="https://test.openai.azure.com"
        )

        # Verify error message includes Azure details
        with pytest.raises(RuntimeError) as exc_info:
            embedding.embed(["Hello"])

        error_msg = str(exc_info.value)
        assert "azure" in error_msg.lower()
        assert "401" in error_msg
        assert "test.openai.azure.com" in error_msg
        assert "text-embedding-3-small" in error_msg

    def test_azure_reuses_openai_logic(self):
        """Test that Azure Embedding inherits from OpenAI Embedding (Acceptance Criteria 4)."""
        embedding = AzureEmbedding(
            model="text-embedding-3-small",
            dimension=1536,
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )

        # Verify inheritance
        assert isinstance(embedding, OpenAIEmbedding)
        assert hasattr(embedding, 'embed')
        assert hasattr(embedding, '_embed_batch')

    @patch('src.libs.embedding.azure_embedding.httpx.Client')
    def test_azure_batch_processing_works(self, mock_client_class):
        """Test that Azure batch processing works (reuses OpenAI logic)."""
        # Setup mock
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        mock_response = Mock()
        mock_response.json.return_value = create_mock_embedding_response(
            embeddings=embeddings,
            total_tokens=20
        )
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Create embedding
        embedding = AzureEmbedding(
            model="text-embedding-3-small",
            dimension=2,
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )
        result = embedding.embed(["Text 1", "Text 2"])

        # Verify batch processing works (inherited from OpenAI)
        assert len(result.embeddings) == 2
        assert result.provider == "azure"


# ===== Helper Methods for Creating Settings =====

@staticmethod
def _create_llm_config():
    from src.core.settings import LLMConfig
    return LLMConfig(provider="fake", model="fake-model")

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


# Add helper methods to test classes
TestOpenAIEmbedding._create_llm_config = _create_llm_config
TestOpenAIEmbedding._create_vector_store_config = _create_vector_store_config
TestOpenAIEmbedding._create_retrieval_config = _create_retrieval_config
TestOpenAIEmbedding._create_rerank_config = _create_rerank_config
TestOpenAIEmbedding._create_evaluation_config = _create_evaluation_config
TestOpenAIEmbedding._create_observability_config = _create_observability_config
TestOpenAIEmbedding._create_dashboard_config = _create_dashboard_config

TestAzureEmbedding._create_llm_config = _create_llm_config
TestAzureEmbedding._create_vector_store_config = _create_vector_store_config
TestAzureEmbedding._create_retrieval_config = _create_retrieval_config
TestAzureEmbedding._create_rerank_config = _create_rerank_config
TestAzureEmbedding._create_evaluation_config = _create_evaluation_config
TestAzureEmbedding._create_observability_config = _create_observability_config
TestAzureEmbedding._create_dashboard_config = _create_dashboard_config
