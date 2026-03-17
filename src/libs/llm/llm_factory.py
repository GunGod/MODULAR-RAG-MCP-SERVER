"""
LLM Factory for creating LLM provider instances.

This module provides the factory pattern implementation for creating LLM
instances based on configuration settings. The factory reads the provider
type from settings and returns the appropriate implementation.

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import Optional

from src.core.settings import LLMConfig, Settings
from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.fake_llm import FakeLLM  # For testing/fallback
from src.libs.llm.openai_llm import OpenAILLM
from src.libs.llm.azure_llm import AzureLLM
from src.libs.llm.deepseek_llm import DeepSeekLLM
from src.libs.llm.glm_llm import GLMLLM
from src.observability.logger import get_logger

logger = get_logger(__name__)


class LLMFactory:
    """
    Factory class for creating LLM provider instances.

    The factory determines which LLM implementation to use based on the
    provider field in settings.llm. This allows zero-code switching between
    different LLM providers through configuration.

    Supported providers:
        - "fake": Fake LLM for testing (returns predefined responses)
        - "openai": OpenAI API (implementation in B7.1)
        - "azure": Azure OpenAI (implementation in B7.1)
        - "ollama": Ollama local models (implementation in B7.2)
        - "deepseek": DeepSeek API (implementation in B7.3)
    """

    # Registry of available providers
    _providers = {
        "fake": FakeLLM,
        "openai": OpenAILLM,
        "azure": AzureLLM,
        "deepseek": DeepSeekLLM,
        "glm": GLMLLM,
    }

    @classmethod
    def create(cls, settings: Settings) -> BaseLLM:
        """
        Create an LLM instance based on settings configuration.

        This factory method reads the llm.provider field from settings and
        returns an instance of the corresponding LLM implementation.

        Args:
            settings: Settings object containing LLM configuration

        Returns:
            BaseLLM instance configured according to settings

        Raises:
            ValueError: If the provider is not supported
            RuntimeError: If provider initialization fails

        Example:
            >>> settings = load_settings()
            >>> llm = LLMFactory.create(settings)
            >>> response = llm.chat([Message(role="user", content="Hello")])
        """
        llm_config = settings.llm
        provider = llm_config.provider.lower()

        # Validate provider is supported
        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unsupported LLM provider: '{provider}'. "
                f"Available providers: {available}"
            )

        # Get the provider class
        provider_class = cls._providers[provider]

        try:
            # Create instance with provider-specific config
            llm_instance = cls._create_provider_instance(provider_class, llm_config)
            logger.info(f"Created LLM: {llm_instance}")
            return llm_instance

        except Exception as e:
            logger.error(f"Failed to create LLM provider '{provider}': {e}")
            raise RuntimeError(f"Failed to initialize LLM provider '{provider}': {e}") from e

    @classmethod
    def _create_provider_instance(cls, provider_class: type, config: LLMConfig) -> BaseLLM:
        """
        Create an instance of a specific LLM provider.

        This method extracts the relevant configuration and passes it to the
        provider's constructor.

        Args:
            provider_class: The LLM class to instantiate
            config: LLM configuration from settings

        Returns:
            Instantiated LLM provider
        """
        # Build kwargs from config (excluding fields handled specially)
        provider_kwargs = {
            "model": config.model,
            "api_key": config.api_key,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }

        # Add provider-specific config
        if config.provider == "azure":
            if config.azure_endpoint:
                provider_kwargs["azure_endpoint"] = config.azure_endpoint
            if config.api_version:
                provider_kwargs["api_version"] = config.api_version

        # Create and return instance
        return provider_class(**provider_kwargs)

    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """
        Register a new LLM provider.

        This allows adding new providers at runtime without modifying the
        factory code. Useful for plugins or custom implementations.

        Args:
            name: Provider name (e.g., "custom_llm")
            provider_class: Class implementing BaseLLM

        Example:
            >>> class CustomLLM(BaseLLM):
            ...     def chat(self, messages, **kwargs): ...
            >>> LLMFactory.register_provider("custom", CustomLLM)
        """
        if not issubclass(provider_class, BaseLLM):
            raise TypeError(f"{provider_class} must inherit from BaseLLM")

        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered LLM provider: {name}")

    @classmethod
    def list_providers(cls) -> list[str]:
        """
        Get a list of all registered provider names.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())
