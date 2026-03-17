"""
OpenAI Embedding provider implementation.

This module implements the BaseEmbedding interface for OpenAI's embedding API.
It supports batch processing of texts to generate vector embeddings.

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import List, Optional

import httpx

from src.libs.embedding.base_embedding import BaseEmbedding, EmbeddingResult
from src.observability.logger import get_logger

logger = get_logger(__name__)


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI Embedding provider implementation.

    This class implements text embeddings using OpenAI's API.
    It supports models like text-embedding-3-small, text-embedding-3-large, and
    text-embedding-ada-002.

    API Documentation: https://platform.openai.com/docs/guides/embeddings

    Supported models:
        - text-embedding-3-small: 1536 dimensions, faster and cheaper
        - text-embedding-3-large: 3072 dimensions, higher quality
        - text-embedding-ada-002: 1536 dimensions, legacy model

    Example:
        >>> embedding = OpenAIEmbedding(model="text-embedding-3-small", dimension=1536, api_key="sk-...")
        >>> result = embedding.embed(["Hello, world!", "How are you?"])
        >>> print(len(result.embeddings))  # 2
        >>> print(len(result.embeddings[0]))  # 1536
    """

    # OpenAI API endpoint
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    EMBEDDING_ENDPOINT = "/embeddings"

    # Model-specific dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    # Maximum input tokens for different models
    MAX_INPUT_TOKENS = {
        "text-embedding-3-small": 8191,
        "text-embedding-3-large": 8191,
        "text-embedding-ada-002": 8191,
    }

    def __init__(
        self,
        model: str,
        dimension: int,
        api_key: Optional[str] = None,
        batch_size: int = 32,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI Embedding provider.

        Args:
            model: Model name (e.g., "text-embedding-3-small")
            dimension: Dimension of the embedding vectors
            api_key: OpenAI API key
            batch_size: Maximum number of texts to process in one batch
            base_url: Custom base URL (defaults to OpenAI's API endpoint)
            **kwargs: Additional provider-specific parameters
        """
        super().__init__(model, dimension, api_key, batch_size, **kwargs)

        # Use provided base_url or default to OpenAI's endpoint
        self.base_url = base_url or self.DEFAULT_BASE_URL

        # Validate API key
        if not self.api_key:
            raise ValueError(
                f"[{self.provider_name}] API key is required. "
                f"Provide it via api_key parameter or OPENAI_API_KEY environment variable."
            )

        # Initialize HTTP client
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

        logger.info(
            f"Initialized OpenAI Embedding: model='{self.model}', "
            f"dimension={self.dimension}, batch_size={self.batch_size}"
        )

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "openai"

    def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> EmbeddingResult:
        """
        Convert texts to embedding vectors using OpenAI API.

        This method handles batching internally if the number of texts exceeds
        batch_size. All texts are processed in order.

        Args:
            texts: List of text strings to embed
            **kwargs: Additional OpenAI-specific parameters (e.g., encoding_format)

        Returns:
            EmbeddingResult containing embeddings and metadata

        Raises:
            ValueError: If texts list is empty or contains invalid inputs
            RuntimeError: If the API call fails

        Example:
            >>> result = embedding.embed(["Hello", "World"])
            >>> print(len(result.embeddings))  # 2
            >>> print(len(result.embeddings[0]))  # 1536
        """
        # Validate input
        if not texts:
            raise ValueError(
                f"[{self.provider_name}] Texts list cannot be empty. "
                f"Please provide at least one text to embed."
            )

        if not all(isinstance(text, str) for text in texts):
            raise ValueError(
                f"[{self.provider_name}] All items in texts must be strings. "
                f"Got types: {[type(text).__name__ for text in texts]}"
            )

        # Check for empty strings
        if any(text.strip() == "" for text in texts):
            logger.warning(
                f"[{self.provider_name}] Some texts are empty or contain only whitespace. "
                f"Empty texts will still be processed but may result in lower quality embeddings."
            )

        # Process in batches
        all_embeddings = []
        total_tokens = 0

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings, batch_tokens = self._embed_batch(batch, **kwargs)
            all_embeddings.extend(batch_embeddings)
            total_tokens += batch_tokens

        # Build result
        result = EmbeddingResult(
            embeddings=all_embeddings,
            model=self.model,
            provider=self.provider_name,
            dimension=self.dimension,
            usage={
                "total_tokens": total_tokens,
            }
        )

        logger.debug(
            f"[{self.provider_name}] Embedded {len(texts)} texts: "
            f"{total_tokens} total tokens"
        )

        return result

    def _embed_batch(
        self,
        texts: List[str],
        **kwargs
    ) -> tuple[List[List[float]], int]:
        """
        Embed a batch of texts.

        Args:
            texts: List of texts to embed (max batch_size)
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            Tuple of (list of embeddings, total tokens used)

        Raises:
            RuntimeError: If the API call fails
        """
        # Prepare request payload
        payload = {
            "model": self.model,
            "input": texts,
            **kwargs  # Pass through additional parameters (e.g., encoding_format="float")
        }

        try:
            # Make API request
            logger.debug(
                f"[{self.provider_name}] Sending embedding request: "
                f"model='{self.model}', texts={len(texts)}"
            )

            response = self._client.post(
                self.EMBEDDING_ENDPOINT,
                json=payload
            )
            response.raise_for_status()

            # Parse response
            data = response.json()

            # Extract embeddings
            # OpenAI returns embeddings in the same order as input
            embeddings = []
            for item in data["data"]:
                # Sort by index to ensure correct order
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
                f"Model: '{self.model}'. "
                f"Error: {error_detail}"
            )

            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except httpx.RequestError as e:
            # Handle network/connection errors
            error_msg = (
                f"[{self.provider_name}] Network error occurred while communicating with API. "
                f"Model: '{self.model}'. "
                f"Error: {str(e)}"
            )

            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except Exception as e:
            # Handle unexpected errors
            error_msg = (
                f"[{self.provider_name}] Unexpected error during embedding. "
                f"Model: '{self.model}'. "
                f"Error type: {type(e).__name__}. "
                f"Error: {str(e)}"
            )

            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def __del__(self):
        """Clean up HTTP client when instance is destroyed."""
        if hasattr(self, '_client'):
            self._client.close()
