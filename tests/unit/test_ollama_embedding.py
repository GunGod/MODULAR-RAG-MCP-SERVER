"""
Unit tests for Ollama Embedding provider (Task B7.4)

These tests verify that:
1. EmbeddingFactory can create Ollama provider
2. Ollama handles input validation correctly
3. Batch processing works correctly
4. Error messages are clear and include provider information
5. Connection failures provide helpful guidance
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
from src.libs.embedding.ollama_embedding import OllamaEmbedding


# ===== Test Fixtures =====

def create_embedding_config(
    provider: str = "ollama",
    model: str = "nomic-embed-text",
    dimension: int = 768,
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
    model: str = "nomic-embed-text",
    total_tokens: int = 100
) -> dict[str, Any]:
    """Helper to create a mock embedding API response (OpenAI-compatible format)."""
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


def create_mock_legacy_response(
    embedding: list[float],
    model: str = "nomic-embed-text",
    prompt_eval_count: int = 50
) -> dict[str, Any]:
    """Helper to create a mock Ollama legacy API response."""
    return {
        "embedding": embedding,
        "model": model,
        "prompt_eval_count": prompt_eval_count
    }


# ===== Ollama Embedding Tests =====

class TestOllamaEmbedding:
    """Test Ollama Embedding provider."""

    def test_factory_creates_ollama_embedding(self):
        """Test that EmbeddingFactory can create Ollama Embedding (Acceptance Criteria 1)."""
        config = create_embedding_config(
            provider="ollama",
            model="nomic-embed-text",
            dimension=768
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

        assert isinstance(embedding, OllamaEmbedding)
        assert embedding.provider_name == "ollama"
        assert embedding.model == "nomic-embed-text"
        assert embedding.dimension == 768

    def test_ollama_embedding_initialization_with_custom_base_url(self):
        """Test Ollama Embedding can be initialized with custom base URL."""
        embedding = OllamaEmbedding(
            model="nomic-embed-text",
            dimension=768,
            base_url="http://192.168.1.100:11434"
        )

        assert embedding.base_url == "http://192.168.1.100:11434"
        assert embedding.provider_name == "ollama"

    def test_ollama_embed_validates_texts_not_empty(self):
        """Test that Ollama Embedding validates texts list is not empty (Acceptance Criteria 5)."""
        embedding = OllamaEmbedding(
            model="nomic-embed-text",
            dimension=768
        )

        with pytest.raises(ValueError) as exc_info:
            embedding.embed([])

        assert "cannot be empty" in str(exc_info.value)
        assert "ollama" in str(exc_info.value).lower()

    def test_ollama_embed_validates_text_types(self):
        """Test that Ollama Embedding validates text types."""
        embedding = OllamaEmbedding(
            model="nomic-embed-text",
            dimension=768
        )

        with pytest.raises(ValueError) as exc_info:
            embedding.embed(["valid", 123])

        assert "must be strings" in str(exc_info.value)
        assert "ollama" in str(exc_info.value).lower()

    @patch('src.libs.embedding.ollama_embedding.httpx.Client')
    def test_ollama_embed_single_text(self, mock_client_class):
        """Test embedding a single text."""
        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = create_mock_embedding_response(
            embeddings=[[0.1, 0.2, 0.3]],
            model="nomic-embed-text"
        )
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Create embedding and make request
        embedding = OllamaEmbedding(
            model="nomic-embed-text",
            dimension=3,
            base_url="http://localhost:11434"
        )
        result = embedding.embed(["Hello, Ollama!"])

        # Verify result
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 3
        assert result.model == "nomic-embed-text"
        assert result.provider == "ollama"
        assert result.dimension == 3

    @patch('src.libs.embedding.ollama_embedding.httpx.Client')
    def test_ollama_embed_batch_texts(self, mock_client_class):
        """Test embedding multiple texts in batch (Acceptance Criteria 4)."""
        # Setup mock
        embeddings = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
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
        embedding = OllamaEmbedding(
            model="nomic-embed-text",
            dimension=2,
            base_url="http://localhost:11434"
        )
        result = embedding.embed(["Text 1", "Text 2", "Text 3"])

        # Verify result
        assert len(result.embeddings) == 3
        assert result.embeddings == embeddings
        assert result.usage["total_tokens"] == 30

    @patch('src.libs.embedding.ollama_embedding.httpx.Client')
    def test_ollama_embed_handles_batch_size(self, mock_client_class):
        """Test that batch_size is respected (Acceptance Criteria 4)."""
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
        embedding = OllamaEmbedding(
            model="nomic-embed-text",
            dimension=2,
            base_url="http://localhost:11434",
            batch_size=2
        )
        result = embedding.embed(["Text 1", "Text 2", "Text 3"])

        # Verify batch processing
        assert mock_client.post.call_count == 2  # 2 batches
        assert len(result.embeddings) == 3
        assert result.usage["total_tokens"] == 30

    @patch('src.libs.embedding.ollama_embedding.httpx.Client')
    def test_ollama_http_error_includes_provider_info(self, mock_client_class):
        """Test that HTTP errors include clear provider information."""
        # Setup mock to raise HTTP error
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {
            "error": "Internal server error"
        }
        mock_client = Mock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "Internal server error", request=Mock(), response=mock_response
        )
        mock_client_class.return_value = mock_client

        # Create embedding and make request
        embedding = OllamaEmbedding(
            model="nomic-embed-text",
            dimension=768,
            base_url="http://localhost:11434"
        )

        # Verify error message is clear
        with pytest.raises(RuntimeError) as exc_info:
            embedding.embed(["Hello"])

        error_msg = str(exc_info.value)
        assert "ollama" in error_msg.lower()
        assert "500" in error_msg
        assert "nomic-embed-text" in error_msg
        assert "localhost:11434" in error_msg

    @patch('src.libs.embedding.ollama_embedding.httpx.Client')
    def test_ollama_connection_error_provides_helpful_guidance(self, mock_client_class):
        """Test that connection errors provide helpful guidance."""
        # Setup mock to raise connection error
        mock_client = Mock()
        mock_client.post.side_effect = httpx.RequestError("Connection refused")
        mock_client_class.return_value = mock_client

        # Create embedding and make request
        embedding = OllamaEmbedding(
            model="nomic-embed-text",
            dimension=768,
            base_url="http://localhost:11434"
        )

        # Verify error message includes helpful guidance
        with pytest.raises(RuntimeError) as exc_info:
            embedding.embed(["Hello"])

        error_msg = str(exc_info.value)
        assert "ollama" in error_msg.lower()
        assert "unable to connect" in error_msg.lower()
        assert "localhost:11434" in error_msg
        assert "ollama serve" in error_msg.lower()  # Helpful command
        assert "ollama pull" in error_msg.lower()  # Helpful command to pull model

    @patch('src.libs.embedding.ollama_embedding.httpx.Client')
    def test_ollama_chat_success_legacy_format(self, mock_client_class):
        """Test successful Ollama embedding with legacy API format."""
        # Setup mock - first call returns 404 (OpenAI endpoint not found)
        mock_response_404 = Mock()
        mock_response_404.status_code = 404

        # Second call returns legacy format
        mock_response_success = Mock()
        mock_response_success.json.return_value = create_mock_legacy_response(
            embedding=[0.1, 0.2, 0.3],
            model="nomic-embed-text"
        )

        mock_client = Mock()
        mock_client.post.side_effect = [mock_response_404, mock_response_success]
        mock_client_class.return_value = mock_client

        # Create embedding and make request
        embedding = OllamaEmbedding(
            model="nomic-embed-text",
            dimension=3,
            base_url="http://localhost:11434"
        )
        result = embedding.embed(["Hello!"])

        # Verify response
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 3
        assert result.provider == "ollama"

    def test_ollama_dimension_warning_for_known_models(self):
        """Test that dimension mismatch with known models triggers warning."""
        import logging
        from src.observability.logger import get_logger

        # Capture log output
        logger_instance = get_logger("src.libs.embedding.ollama_embedding")
        with patch.object(logger_instance, 'warning') as mock_warning:
            embedding = OllamaEmbedding(
                model="nomic-embed-text",  # Expected 768
                dimension=1536  # Wrong dimension
            )

            # Verify warning was called
            mock_warning.assert_called_once()
            args, _ = mock_warning.call_args
            assert "768" in args[0]
            assert "1536" in args[0]
            assert "nomic-embed-text" in args[0]

    @patch('src.libs.embedding.ollama_embedding.httpx.Client')
    def test_ollama_embed_with_different_models(self, mock_client_class):
        """Test Ollama works with different model names."""
        models = ["nomic-embed-text", "mxbai-embed-large", "all-minilm"]

        for model in models:
            # Setup mock
            mock_response = Mock()
            mock_response.json.return_value = create_mock_embedding_response(
                embeddings=[[0.1, 0.2]],
                model=model
            )
            mock_client = Mock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            mock_client_class.reset_mock()

            # Create embedding and make request
            dimension = 768 if "nomic" in model or "all-minilm" in model else 1024
            embedding = OllamaEmbedding(
                model=model,
                dimension=dimension,
                base_url="http://localhost:11434"
            )
            result = embedding.embed(["Hello!"])

            # Verify
            assert result.model == model
            assert isinstance(result, EmbeddingResult)


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


# Add helper methods to test class
TestOllamaEmbedding._create_llm_config = _create_llm_config
TestOllamaEmbedding._create_vector_store_config = _create_vector_store_config
TestOllamaEmbedding._create_retrieval_config = _create_retrieval_config
TestOllamaEmbedding._create_rerank_config = _create_rerank_config
TestOllamaEmbedding._create_evaluation_config = _create_evaluation_config
TestOllamaEmbedding._create_observability_config = _create_observability_config
TestOllamaEmbedding._create_dashboard_config = _create_dashboard_config
