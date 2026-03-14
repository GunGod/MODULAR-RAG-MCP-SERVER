"""
Evaluator Factory for creating evaluator provider instances.

This module provides the factory pattern implementation for creating evaluator
instances based on configuration settings.

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import List

from src.core.settings import Settings
from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.observability.logger import get_logger

logger = get_logger(__name__)


class EvaluatorFactory:
    """
    Factory class for creating evaluator provider instances.

    The factory determines which evaluator implementation to use based on the
    backends field in settings.evaluation. This allows zero-code switching between
    different evaluation frameworks through configuration.

    Supported providers:
        - "custom": Custom evaluator with lightweight metrics
        - "ragas": Ragas framework (implementation in Phase H)
        - "deepeval": DeepEval framework (implementation in Phase H)
    """

    # Registry of available providers
    _providers = {
        "custom": CustomEvaluator,
        # Additional providers will be registered in Phase H
    }

    @classmethod
    def create(cls, settings: Settings) -> BaseEvaluator:
        """
        Create an evaluator instance based on settings configuration.

        This factory method reads the evaluator backends field from settings and
        returns an instance of the corresponding evaluator implementation.
        If multiple backends are specified, returns the first available one.

        Args:
            settings: Settings object containing evaluator configuration

        Returns:
            BaseEvaluator instance configured according to settings

        Raises:
            ValueError: If no supported provider is found
            RuntimeError: If provider initialization fails

        Example:
            >>> settings = load_settings()
            >>> evaluator = EvaluatorFactory.create(settings)
            >>> result = evaluator.evaluate(queries)
            >>> print(f"Hit Rate: {result.metrics.hit_rate_at_k}")
        """
        evaluation_config = settings.evaluation

        # Get the list of backends (try each one until we find a working one)
        backends = evaluation_config.backends if hasattr(evaluation_config, 'backends') else ["custom"]

        for backend in backends:
            provider = backend.lower()

            # Check if this provider is available
            if provider in cls._providers:
                provider_class = cls._providers[provider]

                try:
                    # Create instance with provider-specific config
                    evaluator_instance = cls._create_provider_instance(
                        provider_class,
                        evaluation_config
                    )
                    logger.info(f"Created evaluator: {evaluator_instance}")
                    return evaluator_instance

                except Exception as e:
                    logger.warning(f"Failed to create evaluator '{provider}': {e}")
                    continue

        # If we get here, no provider worked
        available = ", ".join(cls._providers.keys())
        raise ValueError(
            f"Unable to create any evaluator from backends: {backends}. "
            f"Available providers: {available}"
        )

    @classmethod
    def _create_provider_instance(
        cls,
        provider_class: type,
        config
    ) -> BaseEvaluator:
        """
        Create an instance of a specific evaluator provider.

        This method extracts the relevant configuration and passes it to the
        provider's constructor.

        Args:
            provider_class: The evaluator class to instantiate
            config: EvaluationConfig configuration from settings

        Returns:
            Instantiated evaluator provider
        """
        # Build kwargs from config
        provider_kwargs = {
            "top_k": config.top_k if hasattr(config, 'top_k') else 10,
        }

        # Create and return instance
        return provider_class(**provider_kwargs)

    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """
        Register a new evaluator provider.

        This allows adding new providers at runtime without modifying the
        factory code. Useful for plugins or custom implementations.

        Args:
            name: Provider name (e.g., "custom_evaluator")
            provider_class: Class implementing BaseEvaluator

        Example:
            >>> class CustomEvaluator(BaseEvaluator):
            ...     def evaluate(self, queries, **kwargs): ...
            >>> EvaluatorFactory.register_provider("custom", CustomEvaluator)
        """
        if not issubclass(provider_class, BaseEvaluator):
            raise TypeError(f"{provider_class} must inherit from BaseEvaluator")

        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered evaluator provider: {name}")

    @classmethod
    def list_providers(cls) -> List[str]:
        """
        Get a list of all registered provider names.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())
