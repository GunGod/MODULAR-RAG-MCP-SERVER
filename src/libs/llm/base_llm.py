"""
Base LLM abstract interface for the Modular RAG MCP Server.

This module defines the abstract interface that all LLM providers must implement.
The interface is intentionally simple to support multiple providers (Azure OpenAI,
OpenAI, Ollama, DeepSeek, etc.) with minimal overhead.

Author: Modular RAG MCP Server Project
License: MIT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Message:
    """
    A message in a conversation with an LLM.

    Attributes:
        role: The role of the message sender (system, user, assistant)
        content: The text content of the message
    """
    role: str
    content: str

    def __post_init__(self):
        """Validate role is one of the allowed values."""
        valid_roles = {"system", "user", "assistant"}
        if self.role not in valid_roles:
            raise ValueError(f"Invalid role '{self.role}'. Must be one of {valid_roles}")


@dataclass
class LLMResponse:
    """
    Response from an LLM provider.

    Attributes:
        content: The generated text content
        model: The model name used for generation
        provider: The provider name (e.g., "openai", "azure")
        usage: Optional token usage information
        raw_response: Optional raw response from the provider
    """
    content: str
    model: str
    provider: str
    usage: Optional[dict[str, Any]] = None
    raw_response: Optional[Any] = None


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.

    All LLM implementations (OpenAI, Azure, Ollama, etc.) must inherit from
    this class and implement the chat() method.

    The interface is intentionally simple to minimize provider-specific
    complexity in the core RAG pipeline.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ):
        """
        Initialize the LLM provider.

        Args:
            model: Model name (e.g., "gpt-4o", "llama3:8b")
            api_key: API key for the provider (if required)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
        """
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._provider_config = kwargs

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send a chat request to the LLM and get a response.

        Args:
            messages: List of messages in the conversation
            temperature: Override the default temperature
            max_tokens: Override the default max_tokens
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse containing the generated text and metadata

        Raises:
            RuntimeError: If the API call fails
            ValueError: If parameters are invalid
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Get the provider name (e.g., "openai", "azure", "ollama").

        Returns:
            Provider name as a string
        """
        pass

    def __repr__(self) -> str:
        """String representation of the LLM instance."""
        return f"{self.__class__.__name__}(model='{self.model}', provider='{self.provider_name}')"
