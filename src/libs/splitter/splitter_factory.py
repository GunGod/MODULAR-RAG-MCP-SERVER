"""
Splitter Factory for creating splitter provider instances.

This module provides the factory pattern implementation for creating splitter
instances based on configuration settings.

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import List

from src.core.settings import Settings
from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.fake_splitter import FakeSplitter
from src.libs.splitter.recursive_splitter import RecursiveSplitter
from src.observability.logger import get_logger

logger = get_logger(__name__)


class SplitterFactory:
    """
    Factory class for creating splitter provider instances.

    The factory determines which splitting implementation to use based on the
    provider field in settings. This allows zero-code switching between
    different splitting strategies through configuration.

    Supported providers:
        - "fake": Fake splitter for testing (returns fixed-size chunks)
        - "recursive": Recursive character splitting (implementation in B7.5)
        - "semantic": Semantic splitting (implementation in B7.5)
        - "fixed": Fixed-length splitting (implementation in B7.5)
    """

    # Registry of available providers
    _providers = {
        "fake": FakeSplitter,
        "recursive": RecursiveSplitter,
        # Additional providers: semantic, fixed (to be implemented)
    }

    @classmethod
    def create(cls, settings: Settings) -> BaseSplitter:
        """
        Create a splitter instance based on settings configuration.

        This factory method reads the splitter provider field from settings and
        returns an instance of the corresponding splitter implementation.

        Args:
            settings: Settings object containing splitter configuration

        Returns:
            BaseSplitter instance configured according to settings

        Raises:
            ValueError: If the provider is not supported
            RuntimeError: If provider initialization fails

        Example:
            >>> settings = load_settings()
            >>> splitter = SplitterFactory.create(settings)
            >>> chunks = splitter.split_text("Long text...")
            >>> print(len(chunks))
        """
        # For now, we'll use a default configuration since Settings doesn't have splitter config yet
        # This will be updated when splitter configuration is added to Settings
        provider = "fake"  # Default provider for now

        # Validate provider is supported
        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unsupported splitter provider: '{provider}'. "
                f"Available providers: {available}"
            )

        # Get the provider class
        provider_class = cls._providers[provider]

        try:
            # Create instance with default config
            splitter_instance = cls._create_provider_instance(
                provider_class,
                chunk_size=1000,
                chunk_overlap=200
            )
            logger.info(f"Created splitter: {splitter_instance}")
            return splitter_instance

        except Exception as e:
            logger.error(f"Failed to create splitter provider '{provider}': {e}")
            raise RuntimeError(f"Failed to initialize splitter provider '{provider}': {e}") from e

    @classmethod
    def _create_provider_instance(
        cls,
        provider_class: type,
        chunk_size: int,
        chunk_overlap: int
    ) -> BaseSplitter:
        """
        Create an instance of a specific splitter provider.

        This method extracts the relevant configuration and passes it to the
        provider's constructor.

        Args:
            provider_class: The splitter class to instantiate
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks

        Returns:
            Instantiated splitter provider
        """
        # Build kwargs from config
        provider_kwargs = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }

        # Create and return instance
        return provider_class(**provider_kwargs)

    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """
        Register a new splitter provider.

        This allows adding new providers at runtime without modifying the
        factory code. Useful for plugins or custom implementations.

        Args:
            name: Provider name (e.g., "custom_splitter")
            provider_class: Class implementing BaseSplitter

        Example:
            >>> class CustomSplitter(BaseSplitter):
            ...     def split_text(self, text, **kwargs): ...
            >>> SplitterFactory.register_provider("custom", CustomSplitter)
        """
        if not issubclass(provider_class, BaseSplitter):
            raise TypeError(f"{provider_class} must inherit from BaseSplitter")

        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered splitter provider: {name}")

    @classmethod
    def list_providers(cls) -> List[str]:
        """
        Get a list of all registered provider names.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())
