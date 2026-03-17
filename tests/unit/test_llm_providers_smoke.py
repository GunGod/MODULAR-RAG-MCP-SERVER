"""
Unit tests for OpenAI-compatible LLM providers (Task B7.1)

These tests verify that:
1. LLMFactory can create OpenAI, Azure, DeepSeek, and GLM providers
2. Providers handle input validation correctly
3. Error messages are clear and include provider information
4. HTTP requests are properly formatted (mocked, no real network calls)

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
from src.libs.llm.openai_llm import OpenAILLM
from src.libs.llm.azure_llm import AzureLLM
from src.libs.llm.deepseek_llm import DeepSeekLLM
from src.libs.llm.glm_llm import GLMLLM


# ===== Test Fixtures =====

def create_llm_config(
    provider: str = "openai",
    model: str = "gpt-4o",
    api_key: str = "test-api-key",
    **kwargs
) -> LLMConfig:
    """Helper to create LLMConfig for testing."""
    return LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        **kwargs
    )


def create_mock_response(
    content: str = "Test response",
    model: str = "gpt-4o",
    total_tokens: int = 100
) -> dict[str, Any]:
    """Helper to create a mock API response."""
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


# ===== OpenAI LLM Tests =====

class TestOpenAILLM:
    """Test OpenAI LLM provider."""

    def test_factory_creates_openai_llm(self):
        """Test that LLMFactory can create OpenAI LLM."""
        config = create_llm_config(provider="openai", model="gpt-4o")
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

        assert isinstance(llm, OpenAILLM)
        assert llm.provider_name == "openai"
        assert llm.model == "gpt-4o"

    def test_openai_llm_requires_api_key(self):
        """Test that OpenAI LLM raises error without API key."""
        with pytest.raises(ValueError) as exc_info:
            OpenAILLM(model="gpt-4o", api_key=None)

        assert "API key is required" in str(exc_info.value)
        assert "openai" in str(exc_info.value).lower()

    def test_openai_chat_validates_messages_not_empty(self):
        """Test that OpenAI LLM validates messages list is not empty."""
        llm = OpenAILLM(model="gpt-4o", api_key="test-key")

        with pytest.raises(ValueError) as exc_info:
            llm.chat([])

        assert "cannot be empty" in str(exc_info.value)
        assert "openai" in str(exc_info.value).lower()

    def test_openai_chat_validates_message_types(self):
        """Test that OpenAI LLM validates message types."""
        llm = OpenAILLM(model="gpt-4o", api_key="test-key")

        with pytest.raises(ValueError) as exc_info:
            llm.chat([{"role": "user", "content": "test"}])  # Dict instead of Message

        assert "Message instances" in str(exc_info.value)
        assert "openai" in str(exc_info.value).lower()

    @patch('src.libs.llm.openai_llm.httpx.Client')
    def test_openai_chat_success(self, mock_client_class):
        """Test successful OpenAI chat completion."""
        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = create_mock_response(
            content="Hello from OpenAI!",
            model="gpt-4o"
        )
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Create LLM and make request
        llm = OpenAILLM(model="gpt-4o", api_key="test-key")
        messages = [Message(role="user", content="Hello!")]
        response = llm.chat(messages)

        # Verify response
        assert isinstance(response, LLMResponse)
        assert response.content == "Hello from OpenAI!"
        assert response.model == "gpt-4o"
        assert response.provider == "openai"
        assert response.usage["total_tokens"] == 100

    @patch('src.libs.llm.openai_llm.httpx.Client')
    def test_openai_chat_http_error_includes_provider_info(self, mock_client_class):
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

        # Create LLM and make request
        llm = OpenAILLM(model="gpt-4o", api_key="invalid-key")
        messages = [Message(role="user", content="Hello!")]

        # Verify error message is clear
        with pytest.raises(RuntimeError) as exc_info:
            llm.chat(messages)

        error_msg = str(exc_info.value)
        assert "openai" in error_msg.lower()
        assert "401" in error_msg
        assert "gpt-4o" in error_msg

    @patch('src.libs.llm.openai_llm.httpx.Client')
    def test_openai_chat_network_error_includes_provider_info(self, mock_client_class):
        """Test that network errors include clear provider information."""
        # Setup mock to raise network error
        mock_client = Mock()
        mock_client.post.side_effect = httpx.RequestError("Connection failed")
        mock_client_class.return_value = mock_client

        # Create LLM and make request
        llm = OpenAILLM(model="gpt-4o", api_key="test-key")
        messages = [Message(role="user", content="Hello!")]

        # Verify error message is clear
        with pytest.raises(RuntimeError) as exc_info:
            llm.chat(messages)

        error_msg = str(exc_info.value)
        assert "openai" in error_msg.lower()
        assert "network error" in error_msg.lower()
        assert "gpt-4o" in error_msg


# ===== Azure LLM Tests =====

class TestAzureLLM:
    """Test Azure OpenAI LLM provider."""

    def test_factory_creates_azure_llm(self):
        """Test that LLMFactory can create Azure LLM."""
        config = create_llm_config(
            provider="azure",
            model="gpt-4o",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )
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

        assert isinstance(llm, AzureLLM)
        assert llm.provider_name == "azure"
        assert llm.model == "gpt-4o"

    def test_azure_llm_requires_endpoint(self):
        """Test that Azure LLM raises error without endpoint."""
        with pytest.raises(ValueError) as exc_info:
            AzureLLM(
                model="gpt-4o",
                api_key="test-key",
                azure_endpoint=None
            )

        assert "azure_endpoint is required" in str(exc_info.value)
        assert "azure" in str(exc_info.value).lower()

    def test_azure_llm_requires_api_key(self):
        """Test that Azure LLM raises error without API key."""
        with pytest.raises(ValueError) as exc_info:
            AzureLLM(
                model="gpt-4o",
                api_key=None,
                azure_endpoint="https://test.openai.azure.com"
            )

        assert "API key is required" in str(exc_info.value)
        assert "azure" in str(exc_info.value).lower()

    @patch('src.libs.llm.azure_llm.httpx.Client')
    def test_azure_chat_success(self, mock_client_class):
        """Test successful Azure chat completion."""
        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = create_mock_response(
            content="Hello from Azure!",
            model="gpt-4o"
        )
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Create LLM and make request
        llm = AzureLLM(
            model="gpt-4o",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )
        messages = [Message(role="user", content="Hello!")]
        response = llm.chat(messages)

        # Verify response
        assert isinstance(response, LLMResponse)
        assert response.content == "Hello from Azure!"
        assert response.provider == "azure"

    @patch('src.libs.llm.azure_llm.httpx.Client')
    def test_azure_chat_http_error_includes_detailed_info(self, mock_client_class):
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

        # Create LLM and make request
        llm = AzureLLM(
            model="gpt-4o",
            api_key="invalid-key",
            azure_endpoint="https://test.openai.azure.com"
        )
        messages = [Message(role="user", content="Hello!")]

        # Verify error message includes Azure details
        with pytest.raises(RuntimeError) as exc_info:
            llm.chat(messages)

        error_msg = str(exc_info.value)
        assert "azure" in error_msg.lower()
        assert "401" in error_msg
        assert "test.openai.azure.com" in error_msg
        assert "gpt-4o" in error_msg


# ===== DeepSeek LLM Tests =====

class TestDeepSeekLLM:
    """Test DeepSeek LLM provider."""

    def test_factory_creates_deepseek_llm(self):
        """Test that LLMFactory can create DeepSeek LLM."""
        config = create_llm_config(
            provider="deepseek",
            model="deepseek-chat",
            api_key="test-key"
        )
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

        assert isinstance(llm, DeepSeekLLM)
        assert llm.provider_name == "deepseek"
        assert llm.model == "deepseek-chat"

    def test_deepseek_llm_requires_api_key(self):
        """Test that DeepSeek LLM raises error without API key."""
        with pytest.raises(ValueError) as exc_info:
            DeepSeekLLM(model="deepseek-chat", api_key=None)

        assert "API key is required" in str(exc_info.value)
        assert "deepseek" in str(exc_info.value).lower()

    @patch('src.libs.llm.deepseek_llm.httpx.Client')
    def test_deepseek_chat_success(self, mock_client_class):
        """Test successful DeepSeek chat completion."""
        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = create_mock_response(
            content="Hello from DeepSeek!",
            model="deepseek-chat"
        )
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Create LLM and make request
        llm = DeepSeekLLM(model="deepseek-chat", api_key="test-key")
        messages = [Message(role="user", content="Hello!")]
        response = llm.chat(messages)

        # Verify response
        assert isinstance(response, LLMResponse)
        assert response.content == "Hello from DeepSeek!"
        assert response.model == "deepseek-chat"
        assert response.provider == "deepseek"

    @patch('src.libs.llm.deepseek_llm.httpx.Client')
    def test_deepseek_chat_http_error_includes_provider_info(self, mock_client_class):
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

        # Create LLM and make request
        llm = DeepSeekLLM(model="deepseek-chat", api_key="invalid-key")
        messages = [Message(role="user", content="Hello!")]

        # Verify error message is clear
        with pytest.raises(RuntimeError) as exc_info:
            llm.chat(messages)

        error_msg = str(exc_info.value)
        assert "deepseek" in error_msg.lower()
        assert "401" in error_msg
        assert "deepseek-chat" in error_msg


# ===== GLM LLM Tests =====

class TestGLMLLM:
    """Test GLM (智谱AI) LLM provider."""

    def test_factory_creates_glm_llm(self):
        """Test that LLMFactory can create GLM LLM."""
        config = create_llm_config(
            provider="glm",
            model="glm-4",
            api_key="test-key"
        )
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

        assert isinstance(llm, GLMLLM)
        assert llm.provider_name == "glm"
        assert llm.model == "glm-4"

    def test_glm_llm_requires_api_key(self):
        """Test that GLM LLM raises error without API key."""
        with pytest.raises(ValueError) as exc_info:
            GLMLLM(model="glm-4", api_key=None)

        assert "API key is required" in str(exc_info.value)
        assert "glm" in str(exc_info.value).lower()

    @patch('src.libs.llm.glm_llm.httpx.Client')
    def test_glm_chat_success(self, mock_client_class):
        """Test successful GLM chat completion."""
        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = create_mock_response(
            content="你好！我是GLM模型。",
            model="glm-4"
        )
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Create LLM and make request
        llm = GLMLLM(model="glm-4", api_key="test-key")
        messages = [Message(role="user", content="你好！")]
        response = llm.chat(messages)

        # Verify response
        assert isinstance(response, LLMResponse)
        assert response.content == "你好！我是GLM模型。"
        assert response.model == "glm-4"
        assert response.provider == "glm"

    @patch('src.libs.llm.glm_llm.httpx.Client')
    def test_glm_chat_http_error_includes_provider_info(self, mock_client_class):
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

        # Create LLM and make request
        llm = GLMLLM(model="glm-4", api_key="invalid-key")
        messages = [Message(role="user", content="你好！")]

        # Verify error message is clear
        with pytest.raises(RuntimeError) as exc_info:
            llm.chat(messages)

        error_msg = str(exc_info.value)
        assert "glm" in error_msg.lower()
        assert "401" in error_msg
        assert "glm-4" in error_msg

    def test_glm_chat_validates_messages_not_empty(self):
        """Test that GLM LLM validates messages list is not empty."""
        llm = GLMLLM(model="glm-4", api_key="test-key")

        with pytest.raises(ValueError) as exc_info:
            llm.chat([])

        assert "cannot be empty" in str(exc_info.value)
        assert "glm" in str(exc_info.value).lower()

    def test_glm_chat_validates_message_types(self):
        """Test that GLM LLM validates message types."""
        llm = GLMLLM(model="glm-4", api_key="test-key")

        with pytest.raises(ValueError) as exc_info:
            llm.chat([{"role": "user", "content": "test"}])  # Dict instead of Message

        assert "Message instances" in str(exc_info.value)
        assert "glm" in str(exc_info.value).lower()


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


# Add helper methods to test classes
TestOpenAILLM._create_vector_store_config = _create_vector_store_config
TestOpenAILLM._create_retrieval_config = _create_retrieval_config
TestOpenAILLM._create_rerank_config = _create_rerank_config
TestOpenAILLM._create_evaluation_config = _create_evaluation_config
TestOpenAILLM._create_observability_config = _create_observability_config
TestOpenAILLM._create_dashboard_config = _create_dashboard_config

TestAzureLLM._create_vector_store_config = _create_vector_store_config
TestAzureLLM._create_retrieval_config = _create_retrieval_config
TestAzureLLM._create_rerank_config = _create_rerank_config
TestAzureLLM._create_evaluation_config = _create_evaluation_config
TestAzureLLM._create_observability_config = _create_observability_config
TestAzureLLM._create_dashboard_config = _create_dashboard_config

TestDeepSeekLLM._create_vector_store_config = _create_vector_store_config
TestDeepSeekLLM._create_retrieval_config = _create_retrieval_config
TestDeepSeekLLM._create_rerank_config = _create_rerank_config
TestDeepSeekLLM._create_evaluation_config = _create_evaluation_config
TestDeepSeekLLM._create_observability_config = _create_observability_config
TestDeepSeekLLM._create_dashboard_config = _create_dashboard_config

TestGLMLLM._create_vector_store_config = _create_vector_store_config
TestGLMLLM._create_retrieval_config = _create_retrieval_config
TestGLMLLM._create_rerank_config = _create_rerank_config
TestGLMLLM._create_evaluation_config = _create_evaluation_config
TestGLMLLM._create_observability_config = _create_observability_config
TestGLMLLM._create_dashboard_config = _create_dashboard_config
