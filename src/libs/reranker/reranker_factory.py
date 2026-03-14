"""
Reranker Factory for creating reranker provider instances.

This module provides the factory pattern implementation for creating reranker
instances based on configuration settings.

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import List

from src.core.settings import Settings
from src.libs.reranker.base_reranker import BaseReranker
from src.libs.reranker.none_reranker import NoneReranker
from src.observability.logger import get_logger

logger = get_logger(__name__)


class RerankerFactory:
    """
    Factory class for creating reranker provider instances.

    The factory determines which reranker implementation to use based on the
    backend field in settings.reranker. This allows zero-code switching between
    different reranking strategies through configuration.

    Supported providers:
        - "none": No reranking (returns candidates in original order)
        - "cross_encoder": Cross-encoder model (implementation in B9)
        - "llm": LLM-based reranking (implementation in B9)
    """

    # Registry of available providers
    _providers = {
        "none": NoneReranker,
        # Additional providers will be registered in B9 task
    }

    @classmethod
    def create(cls, settings: Settings) -> BaseReranker:
        """
        Create a reranker instance based on settings configuration.

        This factory method reads the reranker backend field from settings and
        returns an instance of the corresponding reranker implementation.

        Args:
            settings: Settings object containing reranker configuration

        Returns:
            BaseReranker instance configured according to settings

        Raises:
            ValueError: If the provider is not supported
            RuntimeError: If provider initialization fails

        Example:
            >>> settings = load_settings()
            >>> reranker = RerankerFactory.create(settings)
            >>> results = reranker.rerank(query="test", candidates=[...])
            >>> print(len(results.candidates))
        """
        reranker_config = settings.rerank
        provider = reranker_config.backend.lower()

        # Validate provider is supported
        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unsupported reranker provider: '{provider}'. "
                f"Available providers: {available}"
            )

        # Get the provider class
        provider_class = cls._providers[provider]

        try:
            # Create instance with provider-specific config
            reranker_instance = cls._create_provider_instance(
                provider_class,
                reranker_config
            )
            logger.info(f"Created reranker: {reranker_instance}")
            return reranker_instance

        except Exception as e:
            logger.error(f"Failed to create reranker provider '{provider}': {e}")
            raise RuntimeError(
                f"Failed to initialize reranker provider '{provider}': {e}"
            ) from e

    @classmethod
    def _create_provider_instance(
        cls,
        provider_class: type,
        config
    ) -> BaseReranker:
        """
        Create an instance of a specific reranker provider.

        This method extracts the relevant configuration and passes it to the
        provider's constructor.

        Args:
            provider_class: The reranker class to instantiate
            config: RerankerConfig configuration from settings

        Returns:
            Instantiated reranker provider
        """
        # Build kwargs from config
        provider_kwargs = {
            "top_k": config.top_m if hasattr(config, 'top_m') else 10,
            "model": config.model if hasattr(config, 'model') else None,
        }

        # Create and return instance
        return provider_class(**provider_kwargs)

    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """
        Register a new reranker provider.

        This allows adding new providers at runtime without modifying the
        factory code. Useful for plugins or custom implementations.

        Args:
            name: Provider name (e.g., "custom_reranker")
            provider_class: Class implementing BaseReranker

        Example:
            >>> class CustomReranker(BaseReranker):
            ...     def rerank(self, query, candidates, **kwargs): ...
            >>> RerankerFactory.register_provider("custom", CustomReranker)
        """
        if not issubclass(provider_class, BaseReranker):
            raise TypeError(f"{provider_class} must inherit from BaseReranker")

        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered reranker provider: {name}")

    @classmethod
    def list_providers(cls) -> List[str]:
        """
        Get a list of all registered provider names.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())
