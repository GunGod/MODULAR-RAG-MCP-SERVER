"""
Azure OpenAI Embedding provider implementation.

This module implements the BaseEmbedding interface for Azure OpenAI Service.
It reuses the core logic from OpenAIEmbedding and adapts it for Azure's endpoint structure.

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import List, Optional

import httpx

from src.libs.embedding.openai_embedding import OpenAIEmbedding
from src.libs.embedding.base_embedding import EmbeddingResult
from src.observability.logger import get_logger

logger = get_logger(__name__)


class AzureEmbedding(OpenAIEmbedding):
    """
    Azure OpenAI Embedding provider implementation.

    This class extends OpenAIEmbedding to work with Azure OpenAI Service.
    It reuses the core embedding logic from OpenAIEmbedding and adapts the
    endpoint structure for Azure's requirements.

    Azure OpenAI API format:
        https://{resource_name}.openai.azure.com/openai/deployments/{deployment_name}/embeddings?api-version={api_version}

    Supported models:
        - text-embedding-ada-002: 1536 dimensions
        - text-embedding-3-small: 1536 dimensions
        - text-embedding-3-large: 3072 dimensions

    Example:
        >>> embedding = AzureEmbedding(
        ...     model="text-embedding-3-small",
        ...     dimension=1536,
        ...     api_key="...",
        ...     azure_endpoint="https://my-resource.openai.azure.com",
        ...     api_version="2024-02-15-preview"
        ... )
        >>> result = embedding.embed(["Hello, world!"])
        >>> print(len(result.embeddings[0]))  # 1536
    """

    # Azure-specific default
    DEFAULT_API_VERSION = "2024-02-15-preview"

    def __init__(
        self,
        model: str,
        dimension: int,
        api_key: Optional[str] = None,
        batch_size: int = 32,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Azure OpenAI Embedding provider.

        Args:
            model: Deployment name (e.g., "text-embedding-3-small")
            dimension: Dimension of the embedding vectors
            api_key: Azure OpenAI API key
            batch_size: Maximum number of texts to process in one batch
            azure_endpoint: Azure OpenAI endpoint (e.g., "https://my-resource.openai.azure.com")
            api_version: API version (defaults to "2024-02-15-preview")
            **kwargs: Additional provider-specific parameters
        """
        # Azure-specific configuration
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version or self.DEFAULT_API_VERSION

        # Validate Azure configuration
        if not self.azure_endpoint:
            raise ValueError(
                f"[azure] azure_endpoint is required. "
                f"Provide your Azure OpenAI endpoint (e.g., 'https://my-resource.openai.azure.com')."
            )

        if not api_key:
            raise ValueError(
                f"[azure] API key is required. "
                f"Provide it via api_key parameter or AZURE_OPENAI_API_KEY environment variable."
            )

        # Ensure endpoint doesn't have trailing slash
        self.azure_endpoint = self.azure_endpoint.rstrip('/')

        # Build the full embedding URL
        # Format: https://{endpoint}/openai/deployments/{deployment_name}/embeddings?api-version={version}
        self.embedding_url = (
            f"{self.azure_endpoint}/openai/deployments"
            f"/{model}/embeddings"
            f"?api-version={self.api_version}"
        )

        # Initialize parent with custom base_url (not used for Azure but kept for compatibility)
        # We'll override the _embed_batch method to use the Azure-specific URL
        super().__init__(
            model=model,
            dimension=dimension,
            api_key=api_key,
            batch_size=batch_size,
            base_url=self.azure_endpoint,  # This won't be used due to override
            **kwargs
        )

        # Override the provider name
        self._provider_name_override = "azure"

        # Reinitialize HTTP client with Azure headers
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "api-key": api_key,  # Azure also uses api-key header
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

        logger.info(
            f"Initialized Azure OpenAI Embedding: deployment='{model}', "
            f"endpoint='{self.azure_endpoint}', api_version='{self.api_version}', "
            f"dimension={dimension}"
        )

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name_override

    def _embed_batch(
        self,
        texts: List[str],
        **kwargs
    ) -> tuple[List[List[float]], int]:
        """
        Embed a batch of texts using Azure OpenAI endpoint.

        This method overrides the parent implementation to use Azure's
        endpoint structure while reusing the same core logic.

        Args:
            texts: List of texts to embed (max batch_size)
            **kwargs: Additional Azure-specific parameters

        Returns:
            Tuple of (list of embeddings, total tokens used)
        """
        # Prepare request payload (Azure uses same format as OpenAI)
        payload = {
            "input": texts,
            **kwargs  # Pass through additional parameters
        }

        try:
            # Make API request to Azure endpoint
            logger.debug(
                f"[{self.provider_name}] Sending embedding request to Azure: "
                f"deployment='{self.model}', texts={len(texts)}"
            )

            response = self._client.post(
                self.embedding_url,
                json=payload
            )
            response.raise_for_status()

            # Parse response (Azure response format matches OpenAI)
            data = response.json()

            # Extract embeddings
            embeddings = []
            for item in data["data"]:
                embeddings.append(item["embedding"])

            # Get total tokens
            total_tokens = data.get("usage", {}).get("total_tokens", 0)

            return embeddings, total_tokens

        except httpx.HTTPStatusError as e:
            # Handle HTTP errors with clear error messages
            status_code = e.response.status_code
            error_detail = "Unknown error"

            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", str(e))
            except Exception:
                error_detail = str(e)

            error_msg = (
                f"[{self.provider_name}] API request failed with status {status_code}. "
                f"Endpoint: '{self.azure_endpoint}'. "
                f"Deployment: '{self.model}'. "
                f"API Version: '{self.api_version}'. "
                f"Error: {error_detail}"
            )

            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except httpx.RequestError as e:
            # Handle network/connection errors
            error_msg = (
                f"[{self.provider_name}] Network error occurred while communicating with API. "
                f"Endpoint: '{self.azure_endpoint}'. "
                f"Deployment: '{self.model}'. "
                f"Error: {str(e)}"
            )

            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except Exception as e:
            # Handle unexpected errors
            error_msg = (
                f"[{self.provider_name}] Unexpected error during embedding. "
                f"Endpoint: '{self.azure_endpoint}'. "
                f"Deployment: '{self.model}'. "
                f"Error type: {type(e).__name__}. "
                f"Error: {str(e)}"
            )

            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
