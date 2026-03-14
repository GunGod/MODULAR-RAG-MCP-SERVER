"""
VectorStore Factory for creating vector store provider instances.

This module provides the factory pattern implementation for creating vector
store instances based on configuration settings.

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import List

from src.core.settings import Settings
from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.vector_store.fake_vector_store import FakeVectorStore
from src.observability.logger import get_logger

logger = get_logger(__name__)


class VectorStoreFactory:
    """
    Factory class for creating vector store provider instances.

    The factory determines which vector store implementation to use based on the
    backend field in settings.vector_store. This allows zero-code switching between
    different vector databases through configuration.

    Supported providers:
        - "fake": Fake vector store for testing (in-memory storage)
        - "chroma": Chroma DB (implementation in C1)
        - "qdrant": Qdrant (future implementation)
        - "pinecone": Pinecone (future implementation)
    """

    # Registry of available providers
    _providers = {
        "fake": FakeVectorStore,
        # Additional providers will be registered in C1 task
    }

    @classmethod
    def create(cls, settings: Settings) -> BaseVectorStore:
        """
        Create a vector store instance based on settings configuration.

        This factory method reads the vector_store.backend field from settings and
        returns an instance of the corresponding vector store implementation.

        Args:
            settings: Settings object containing vector store configuration

        Returns:
            BaseVectorStore instance configured according to settings

        Raises:
            ValueError: If the provider is not supported
            RuntimeError: If provider initialization fails

        Example:
            >>> settings = load_settings()
            >>> store = VectorStoreFactory.create(settings)
            >>> results = store.query(vector=[0.1, 0.2], top_k=10)
            >>> print(len(results))
        """
        vector_store_config = settings.vector_store
        provider = vector_store_config.backend.lower()

        # Validate provider is supported
        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unsupported vector store provider: '{provider}'. "
                f"Available providers: {available}"
            )

        # Get the provider class
        provider_class = cls._providers[provider]

        try:
            # Create instance with provider-specific config
            store_instance = cls._create_provider_instance(
                provider_class,
                vector_store_config
            )
            logger.info(f"Created vector store: {store_instance}")
            return store_instance

        except Exception as e:
            logger.error(f"Failed to create vector store provider '{provider}': {e}")
            raise RuntimeError(
                f"Failed to initialize vector store provider '{provider}': {e}"
            ) from e

    @classmethod
    def _create_provider_instance(
        cls,
        provider_class: type,
        config
    ) -> BaseVectorStore:
        """
        Create an instance of a specific vector store provider.

        This method extracts the relevant configuration and passes it to the
        provider's constructor.

        Args:
            provider_class: The vector store class to instantiate
            config: VectorStore configuration from settings

        Returns:
            Instantiated vector store provider
        """
        # Build kwargs from config
        provider_kwargs = {
            "collection_name": "default",  # Default collection
            "persist_path": config.persist_path if hasattr(config, 'persist_path') else None,
        }

        # Create and return instance
        return provider_class(**provider_kwargs)

    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """
        Register a new vector store provider.

        This allows adding new providers at runtime without modifying the
        factory code. Useful for plugins or custom implementations.

        Args:
            name: Provider name (e.g., "custom_store")
            provider_class: Class implementing BaseVectorStore

        Example:
            >>> class CustomStore(BaseVectorStore):
            ...     def upsert(self, records, **kwargs): ...
            >>> VectorStoreFactory.register_provider("custom", CustomStore)
        """
        if not issubclass(provider_class, BaseVectorStore):
            raise TypeError(f"{provider_class} must inherit from BaseVectorStore")

        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered vector store provider: {name}")

    @classmethod
    def list_providers(cls) -> List[str]:
        """
        Get a list of all registered provider names.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())
