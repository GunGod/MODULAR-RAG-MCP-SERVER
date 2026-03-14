"""
Unit tests for LLM Factory and BaseLLM interface (Task B1)

These tests verify that:
1. The BaseLLM interface is correctly defined
2. LLMFactory can create instances based on settings
3. Factory routing logic works correctly
4. Unsupported providers raise appropriate errors
5. FakeLLM can be used for testing

Author: Modular RAG MCP Server Project
License: MIT
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from src.core.settings import LLMConfig, Settings
from src.libs.llm.base_llm import BaseLLM, LLMResponse, Message
from src.libs.llm.fake_llm import FakeLLM
from src.libs.llm.llm_factory import LLMFactory


class TestMessage:
    """Test the Message dataclass."""

    def test_create_message_with_valid_role(self):
        """Test creating a message with valid roles."""
        for role in ["system", "user", "assistant"]:
            msg = Message(role=role, content="Hello")
            assert msg.role == role
            assert msg.content == "Hello"

    def test_create_message_with_invalid_role_raises_error(self):
        """Test that invalid role raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Message(role="invalid_role", content="Hello")
        assert "Invalid role" in str(exc_info.value)
        assert "invalid_role" in str(exc_info.value)


class TestLLMResponse:
    """Test the LLMResponse dataclass."""

    def test_create_response(self):
        """Test creating an LLM response."""
        response = LLMResponse(
            content="Test response",
            model="gpt-4o",
            provider="openai",
            usage={"total_tokens": 100}
        )
        assert response.content == "Test response"
        assert response.model == "gpt-4o"
        assert response.provider == "openai"
        assert response.usage["total_tokens"] == 100

    def test_create_response_with_optional_fields(self):
        """Test creating response without optional fields."""
        response = LLMResponse(
            content="Test",
            model="test-model",
            provider="test"
        )
        assert response.usage is None
        assert response.raw_response is None


class TestBaseLLM:
    """Test the BaseLLM abstract interface."""

    def test_base_llm_cannot_be_instantiated(self):
        """Test that BaseLLM cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLM(model="test")

    def test_base_llm_subclass_must_implement_chat(self):
        """Test that subclasses must implement chat method."""
        class IncompleteLLM(BaseLLM):
            def __init__(self):
                super().__init__(model="incomplete")

            @property
            def provider_name(self) -> str:
                return "incomplete"

            # Missing chat() method

        with pytest.raises(TypeError):
            IncompleteLLM()

    def test_base_llm_subclass_must_implement_provider_name(self):
        """Test that subclasses must implement provider_name property."""
        class IncompleteLLM(BaseLLM):
            def __init__(self):
                super().__init__(model="incomplete")

            def chat(self, messages, **kwargs):
                return LLMResponse(
                    content="test",
                    model=self.model,
                    provider="incomplete"
                )

            # Missing provider_name property

        with pytest.raises(TypeError):
            IncompleteLLM()

    def test_base_llm_initialization(self):
        """Test BaseLLM initialization with parameters."""
        llm = FakeLLM(model="test-model", temperature=0.5, max_tokens=1000)
        assert llm.model == "test-model"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 1000

    def test_base_llm_repr(self):
        """Test string representation of LLM."""
        llm = FakeLLM(model="test-model")
        repr_str = repr(llm)
        assert "FakeLLM" in repr_str
        assert "test-model" in repr_str
        assert "fake" in repr_str


class TestFakeLLM:
    """Test the FakeLLM implementation."""

    def test_fake_llm_chat_returns_response(self):
        """Test that FakeLLM.chat() returns a valid LLMResponse."""
        llm = FakeLLM()
        messages = [Message(role="user", content="Hello, FakeLLM!")]
        response = llm.chat(messages)

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert response.model == "fake-model"
        assert response.provider == "fake"
        assert response.usage is not None

    def test_fake_llm_includes_message_in_response(self):
        """Test that FakeLLM includes user message in response."""
        llm = FakeLLM()
        messages = [Message(role="user", content="Test message for FakeLLM")]
        response = llm.chat(messages)

        assert "Test message for FakeLLM" in response.content
        assert "[FakeLLM Response" in response.content

    def test_fake_llm_tracks_call_count(self):
        """Test that FakeLLM tracks number of chat() calls."""
        llm = FakeLLM()
        assert llm.get_call_count() == 0

        llm.chat([Message(role="user", content="First")])
        assert llm.get_call_count() == 1

        llm.chat([Message(role="user", content="Second")])
        assert llm.get_call_count() == 2

    def test_fake_llm_reset_call_count(self):
        """Test resetting the call counter."""
        llm = FakeLLM()
        llm.chat([Message(role="user", content="Test")])
        assert llm.get_call_count() == 1

        llm.reset_call_count()
        assert llm.get_call_count() == 0

    def test_fake_llm_default_response_when_no_user_message(self):
        """Test FakeLLM returns default response when no user message."""
        llm = FakeLLM()
        messages = [Message(role="system", content="You are a helpful assistant.")]
        response = llm.chat(messages)

        # Check that response contains "fake" and "llm" (case-insensitive)
        content_lower = response.content.lower()
        assert "fake" in content_lower and "llm" in content_lower

    def test_fake_llm_provider_name(self):
        """Test FakeLLM provider_name property."""
        llm = FakeLLM()
        assert llm.provider_name == "fake"


class TestLLMFactory:
    """Test the LLMFactory class."""

    def test_factory_create_with_fake_provider(self):
        """Test creating FakeLLM through factory."""
        settings = self._create_settings(provider="fake")
        llm = LLMFactory.create(settings)

        assert isinstance(llm, FakeLLM)
        assert llm.model == "fake-model"
        assert llm.provider_name == "fake"

    def test_factory_create_with_custom_temperature(self):
        """Test factory passes temperature to LLM."""
        settings = self._create_settings(provider="fake", temperature=0.3)
        llm = LLMFactory.create(settings)

        assert llm.temperature == 0.3

    def test_factory_create_with_custom_max_tokens(self):
        """Test factory passes max_tokens to LLM."""
        settings = self._create_settings(provider="fake", max_tokens=500)
        llm = LLMFactory.create(settings)

        assert llm.max_tokens == 500

    def test_factory_create_with_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises ValueError."""
        settings = self._create_settings(provider="unsupported_provider")

        with pytest.raises(ValueError) as exc_info:
            LLMFactory.create(settings)
        assert "Unsupported LLM provider" in str(exc_info.value)
        assert "unsupported_provider" in str(exc_info.value)

    def test_factory_list_providers(self):
        """Test getting list of available providers."""
        providers = LLMFactory.list_providers()
        assert isinstance(providers, list)
        assert "fake" in providers

    def test_factory_register_provider(self):
        """Test registering a custom provider."""
        # Create a custom LLM class
        class CustomLLM(BaseLLM):
            def __init__(self, model="custom", **kwargs):
                super().__init__(model, **kwargs)

            def chat(self, messages, **kwargs):
                return LLMResponse(
                    content="Custom response",
                    model=self.model,
                    provider="custom"
                )

            @property
            def provider_name(self) -> str:
                return "custom"

        # Register the provider
        LLMFactory.register_provider("custom", CustomLLM)

        # Verify it's in the list
        providers = LLMFactory.list_providers()
        assert "custom" in providers

        # Verify it can be created
        settings = self._create_settings(provider="custom")
        llm = LLMFactory.create(settings)
        assert isinstance(llm, CustomLLM)
        assert llm.provider_name == "custom"

    def test_factory_register_non_llm_class_raises_error(self):
        """Test that registering non-BaseLLM class raises TypeError."""
        class NotAnLLM:
            pass

        with pytest.raises(TypeError) as exc_info:
            LLMFactory.register_provider("invalid", NotAnLLM)
        assert "BaseLLM" in str(exc_info.value)

    def test_factory_routing_logic(self):
        """Test that factory correctly routes to different providers."""
        # Test with fake provider
        settings1 = self._create_settings(provider="fake", model="model-1")
        llm1 = LLMFactory.create(settings1)
        assert isinstance(llm1, FakeLLM)
        assert llm1.model == "model-1"

        # Test with different model but same provider
        settings2 = self._create_settings(provider="fake", model="model-2")
        llm2 = LLMFactory.create(settings2)
        assert isinstance(llm2, FakeLLM)
        assert llm2.model == "model-2"

        # Verify they are different instances
        assert llm1 is not llm2

    def _create_settings(
        self,
        provider: str = "fake",
        model: str = "fake-model",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Settings:
        """Helper method to create test Settings object."""
        return Settings(
            llm=LLMConfig(
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            ),
            embedding=self._create_embedding_config(),
            vector_store=self._create_vector_store_config(),
            retrieval=self._create_retrieval_config(),
            rerank=self._create_rerank_config(),
            evaluation=self._create_evaluation_config(),
            observability=self._create_observability_config(),
            dashboard=self._create_dashboard_config(),
        )

    def _create_embedding_config(self):
        from src.core.settings import EmbeddingConfig
        return EmbeddingConfig(provider="openai", model="text-embedding-3-small")

    def _create_vector_store_config(self):
        from src.core.settings import VectorStoreConfig
        return VectorStoreConfig(backend="chroma")

    def _create_retrieval_config(self):
        from src.core.settings import RetrievalConfig
        return RetrievalConfig()

    def _create_rerank_config(self):
        from src.core.settings import RerankConfig
        return RerankConfig()

    def _create_evaluation_config(self):
        from src.core.settings import EvaluationConfig
        return EvaluationConfig()

    def _create_observability_config(self):
        from src.core.settings import ObservabilityConfig
        return ObservabilityConfig()

    def _create_dashboard_config(self):
        from src.core.settings import DashboardConfig
        return DashboardConfig()


class TestLLMIntegration:
    """Integration tests for LLM usage patterns."""

    def test_conversation_flow_with_fake_llm(self):
        """Test a simple conversation flow using FakeLLM."""
        # Create LLM through factory
        settings = self._create_minimal_settings()
        llm = LLMFactory.create(settings)

        # Simulate a conversation
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="What is the capital of France?")
        ]

        response = llm.chat(messages)

        # Verify response
        assert response.content is not None
        assert response.provider == "fake"
        assert "France" in response.content or "capital" in response.content or "FakeLLM" in response.content

    def test_multiple_conversation_turns(self):
        """Test multiple turns in a conversation."""
        settings = self._create_minimal_settings()
        llm = LLMFactory.create(settings)

        # First turn
        messages1 = [Message(role="user", content="Hello")]
        response1 = llm.chat(messages1)
        assert response1.content is not None

        # Second turn (continuing conversation)
        messages2 = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content=response1.content),
            Message(role="user", content="How are you?")
        ]
        response2 = llm.chat(messages2)
        assert response2.content is not None

        # Verify call count increased
        assert llm.get_call_count() == 2

    def _create_minimal_settings(self) -> Settings:
        """Helper to create minimal settings for testing."""
        from src.core.settings import (
            EmbeddingConfig, VectorStoreConfig, RetrievalConfig,
            RerankConfig, EvaluationConfig, ObservabilityConfig, DashboardConfig
        )
        return Settings(
            llm=LLMConfig(provider="fake", model="fake-model"),
            embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
            vector_store=VectorStoreConfig(backend="chroma"),
            retrieval=RetrievalConfig(),
            rerank=RerankConfig(),
            evaluation=EvaluationConfig(),
            observability=ObservabilityConfig(),
            dashboard=DashboardConfig(),
        )
