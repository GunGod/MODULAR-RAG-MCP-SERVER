"""
Embedding Factory for creating embedding provider instances.

This module provides the factory pattern implementation for creating embedding
instances based on configuration settings.

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import List

from src.core.settings import EmbeddingConfig, Settings
from src.libs.embedding.base_embedding import BaseEmbedding, EmbeddingResult
from src.libs.embedding.fake_embedding import FakeEmbedding
from src.libs.embedding.openai_embedding import OpenAIEmbedding
from src.libs.embedding.azure_embedding import AzureEmbedding
from src.observability.logger import get_logger

logger = get_logger(__name__)


class EmbeddingFactory:
    """
    Factory class for creating embedding provider instances.

    The factory determines which embedding implementation to use based on the
    provider field in settings.embedding. This allows zero-code switching between
    different embedding providers through configuration.

    Supported providers:
        - "fake": Fake embedding for testing (returns stable vectors)
        - "openai": OpenAI API (implementation in B7.3)
        - "azure": Azure OpenAI (implementation in B7.3)
        - "ollama": Ollama local models (implementation in B7.4)
    """

    # Registry of available providers
    _providers = {
        "fake": FakeEmbedding,
        "openai": OpenAIEmbedding,
        "azure": AzureEmbedding,
        # Additional providers will be registered in B7.x tasks
    }

    @classmethod
    def create(cls, settings: Settings) -> BaseEmbedding:
        """
        Create an embedding instance based on settings configuration.

        This factory method reads the embedding.provider field from settings and
        returns an instance of the corresponding embedding implementation.

        Args:
            settings: Settings object containing embedding configuration

        Returns:
            BaseEmbedding instance configured according to settings

        Raises:
            ValueError: If the provider is not supported
            RuntimeError: If provider initialization fails

        Example:
            >>> settings = load_settings()
            >>> embedding = EmbeddingFactory.create(settings)
            >>> result = embedding.embed(["Hello", "World"])
            >>> vectors = result.embeddings
        """
        embedding_config = settings.embedding
        provider = embedding_config.provider.lower()

        # Validate provider is supported
        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unsupported embedding provider: '{provider}'. "
                f"Available providers: {available}"
            )

        # Get the provider class
        provider_class = cls._providers[provider]

        try:
            # Create instance with provider-specific config
            embedding_instance = cls._create_provider_instance(provider_class, embedding_config)
            logger.info(f"Created embedding: {embedding_instance}")
            return embedding_instance

        except Exception as e:
            logger.error(f"Failed to create embedding provider '{provider}': {e}")
            raise RuntimeError(f"Failed to initialize embedding provider '{provider}': {e}") from e

    @classmethod
    def _create_provider_instance(cls, provider_class: type, config: EmbeddingConfig) -> BaseEmbedding:
        """
        Create an instance of a specific embedding provider.

        This method extracts the relevant configuration and passes it to the
        provider's constructor.

        Args:
            provider_class: The embedding class to instantiate
            config: Embedding configuration from settings

        Returns:
            Instantiated embedding provider
        """
        # Build kwargs from config
        provider_kwargs = {
            "model": config.model,
            "dimension": config.dimension,
            "api_key": config.api_key,
            "batch_size": config.batch_size,
        }

        # Add provider-specific config for Azure
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
        Register a new embedding provider.

        This allows adding new providers at runtime without modifying the
        factory code. Useful for plugins or custom implementations.

        Args:
            name: Provider name (e.g., "custom_embedding")
            provider_class: Class implementing BaseEmbedding

        Example:
            >>> class CustomEmbedding(BaseEmbedding):
            ...     def embed(self, texts, **kwargs): ...
            >>> EmbeddingFactory.register_provider("custom", CustomEmbedding)
        """
        if not issubclass(provider_class, BaseEmbedding):
            raise TypeError(f"{provider_class} must inherit from BaseEmbedding")

        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered embedding provider: {name}")

    @classmethod
    def list_providers(cls) -> List[str]:
        """
        Get a list of all registered provider names.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())
