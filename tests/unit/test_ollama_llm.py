"""
Unit tests for Ollama LLM provider (Task B7.2)

These tests verify that:
1. LLMFactory can create Ollama provider
2. Ollama handles input validation correctly
3. Error messages are clear and don't leak sensitive configuration
4. Connection failures provide helpful guidance
5. HTTP requests are properly formatted (mocked, no real network calls)

Author: Modular RAG MCP Server Project
License: MIT
"""

from unittest.mock import Mock, patch
from typing import Any

import pytest
import httpx

from src.core.settings import LLMConfig, Settings
from src.libs.llm.base_llm import LLMResponse, Message
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.ollama_llm import OllamaLLM


# ===== Test Fixtures =====

def create_llm_config(
    provider: str = "ollama",
    model: str = "llama3",
    **kwargs
) -> LLMConfig:
    """Helper to create LLMConfig for testing."""
    return LLMConfig(
        provider=provider,
        model=model,
        **kwargs
    )


def create_mock_response(
    content: str = "Test response",
    model: str = "llama3",
    total_tokens: int = 100
) -> dict[str, Any]:
    """Helper to create a mock API response (OpenAI-compatible format)."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 30,
            "total_tokens": total_tokens
        }
    }


def create_mock_legacy_response(
    content: str = "Test response",
    model: str = "llama3"
) -> dict[str, Any]:
    """Helper to create a mock Ollama legacy API response."""
    return {
        "model": model,
        "created_at": "2024-01-01T00:00:00Z",
        "message": {
            "role": "assistant",
            "content": content
        },
        "done": True
    }


# ===== Ollama LLM Tests =====

class TestOllamaLLM:
    """Test Ollama LLM provider."""

    def test_factory_creates_ollama_llm(self):
        """Test that LLMFactory can create Ollama LLM (Acceptance Criteria 1)."""
        config = create_llm_config(provider="ollama", model="llama3")
        settings = Settings(
            llm=config,
            embedding=create_llm_config(provider="openai", model="text-embedding-3-small"),
            vector_store=self._create_vector_store_config(),
            retrieval=self._create_retrieval_config(),
            rerank=self._create_rerank_config(),
            evaluation=self._create_evaluation_config(),
            observability=self._create_observability_config(),
            dashboard=self._create_dashboard_config(),
        )

        llm = LLMFactory.create(settings)

        assert isinstance(llm, OllamaLLM)
        assert llm.provider_name == "ollama"
        assert llm.model == "llama3"

    def test_ollama_llm_initialization_with_custom_base_url(self):
        """Test Ollama LLM can be initialized with custom base URL."""
        llm = OllamaLLM(model="llama3", base_url="http://192.168.1.100:11434")

        assert llm.base_url == "http://192.168.1.100:11434"
        assert llm.provider_name == "ollama"

    def test_ollama_chat_validates_messages_not_empty(self):
        """Test that Ollama LLM validates messages list is not empty."""
        llm = OllamaLLM(model="llama3")

        with pytest.raises(ValueError) as exc_info:
            llm.chat([])

        assert "cannot be empty" in str(exc_info.value)
        assert "ollama" in str(exc_info.value).lower()

    def test_ollama_chat_validates_message_types(self):
        """Test that Ollama LLM validates message types."""
        llm = OllamaLLM(model="llama3")

        with pytest.raises(ValueError) as exc_info:
            llm.chat([{"role": "user", "content": "test"}])  # Dict instead of Message

        assert "Message instances" in str(exc_info.value)
        assert "ollama" in str(exc_info.value).lower()

    @patch('src.libs.llm.ollama_llm.httpx.Client')
    def test_ollama_chat_success_openai_format(self, mock_client_class):
        """Test successful Ollama chat completion with OpenAI-compatible format."""
        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = create_mock_response(
            content="Hello from Ollama!",
            model="llama3"
        )
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Create LLM and make request
        llm = OllamaLLM(model="llama3")
        messages = [Message(role="user", content="Hello!")]
        response = llm.chat(messages)

        # Verify response
        assert isinstance(response, LLMResponse)
        assert response.content == "Hello from Ollama!"
        assert response.model == "llama3"
        assert response.provider == "ollama"
        assert response.usage["total_tokens"] == 100

    @patch('src.libs.llm.ollama_llm.httpx.Client')
    def test_ollama_chat_success_legacy_format(self, mock_client_class):
        """Test successful Ollama chat completion with legacy API format."""
        # Setup mock - first call returns 404 (OpenAI endpoint not found)
        mock_response_404 = Mock()
        mock_response_404.status_code = 404

        # Second call returns legacy format
        mock_response_success = Mock()
        mock_response_success.json.return_value = create_mock_legacy_response(
            content="Hello from Ollama legacy!",
            model="llama3"
        )

        mock_client = Mock()
        mock_client.post.side_effect = [mock_response_404, mock_response_success]
        mock_client_class.return_value = mock_client

        # Create LLM and make request
        llm = OllamaLLM(model="llama3")
        messages = [Message(role="user", content="Hello!")]
        response = llm.chat(messages)

        # Verify response
        assert isinstance(response, LLMResponse)
        assert response.content == "Hello from Ollama legacy!"
        assert response.model == "llama3"
        assert response.provider == "ollama"

    @patch('src.libs.llm.ollama_llm.httpx.Client')
    def test_ollama_chat_http_error_includes_provider_info(self, mock_client_class):
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

        # Create LLM and make request
        llm = OllamaLLM(model="llama3", base_url="http://localhost:11434")
        messages = [Message(role="user", content="Hello!")]

        # Verify error message is clear
        with pytest.raises(RuntimeError) as exc_info:
            llm.chat(messages)

        error_msg = str(exc_info.value)
        assert "ollama" in error_msg.lower()
        assert "500" in error_msg
        assert "llama3" in error_msg
        assert "localhost:11434" in error_msg

    @patch('src.libs.llm.ollama_llm.httpx.Client')
    def test_ollama_connection_error_provides_helpful_guidance(self, mock_client_class):
        """Test that connection errors provide helpful guidance (Acceptance Criteria 2)."""
        # Setup mock to raise connection error
        mock_client = Mock()
        mock_client.post.side_effect = httpx.RequestError("Connection refused")
        mock_client_class.return_value = mock_client

        # Create LLM and make request
        llm = OllamaLLM(model="llama3", base_url="http://localhost:11434")
        messages = [Message(role="user", content="Hello!")]

        # Verify error message includes helpful guidance
        with pytest.raises(RuntimeError) as exc_info:
            llm.chat(messages)

        error_msg = str(exc_info.value)
        assert "ollama" in error_msg.lower()
        assert "unable to connect" in error_msg.lower()
        assert "localhost:11434" in error_msg
        assert "ollama serve" in error_msg.lower()  # Helpful command to start Ollama

    def test_error_sanitization_removes_api_keys(self):
        """Test that error messages sanitize API keys (Acceptance Criteria 2)."""
        llm = OllamaLLM(model="llama3")

        # Test various sensitive patterns
        test_cases = [
            ("Bearer sk-1234567890abcdef", "Bearer [REDACTED]"),
            ("api_key=secret123", "api_key=[REDACTED]"),
            ("token=abc123def456", "token=[REDACTED]"),
        ]

        for input_msg, expected_substring in test_cases:
            sanitized = llm._sanitize_error_message(input_msg)
            assert expected_substring in sanitized
            assert "secret" not in sanitized.lower()
            assert "123" not in sanitized  # Check numbers are removed

    def test_error_sanitization_preserves_normal_messages(self):
        """Test that error sanitization doesn't corrupt normal error messages."""
        llm = OllamaLLM(model="llama3")

        normal_msg = "Model llama3 not found. Please run: ollama pull llama3"
        sanitized = llm._sanitize_error_message(normal_msg)

        assert sanitized == normal_msg

    @patch('src.libs.llm.ollama_llm.httpx.Client')
    def test_ollama_timeout_error_handling(self, mock_client_class):
        """Test that timeout errors are handled gracefully."""
        # Setup mock to raise timeout error
        mock_client = Mock()
        mock_client.post.side_effect = httpx.TimeoutException("Request timed out")
        mock_client_class.return_value = mock_client

        # Create LLM and make request
        llm = OllamaLLM(model="llama3")
        messages = [Message(role="user", content="Hello!")]

        # Verify timeout is handled
        with pytest.raises(RuntimeError) as exc_info:
            llm.chat(messages)

        error_msg = str(exc_info.value)
        assert "ollama" in error_msg.lower()
        assert "llama3" in error_msg

    @patch('src.libs.llm.ollama_llm.httpx.Client')
    def test_ollama_chat_with_different_models(self, mock_client_class):
        """Test Ollama works with different model names."""
        models = ["llama3", "mistral", "codellama", "phi3"]

        for model in models:
            # Setup mock
            mock_response = Mock()
            mock_response.json.return_value = create_mock_response(content=f"Response from {model}", model=model)
            mock_client = Mock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            mock_client_class.reset_mock()

            # Create LLM and make request
            llm = OllamaLLM(model=model)
            messages = [Message(role="user", content="Hello!")]
            response = llm.chat(messages)

            # Verify
            assert response.model == model
            assert isinstance(response, LLMResponse)


# ===== Helper Methods for Creating Settings =====

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
TestOllamaLLM._create_vector_store_config = _create_vector_store_config
TestOllamaLLM._create_retrieval_config = _create_retrieval_config
TestOllamaLLM._create_rerank_config = _create_rerank_config
TestOllamaLLM._create_evaluation_config = _create_evaluation_config
TestOllamaLLM._create_observability_config = _create_observability_config
TestOllamaLLM._create_dashboard_config = _create_dashboard_config
