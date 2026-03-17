"""
Ollama Embedding provider implementation.

This module implements the BaseEmbedding interface for Ollama, which allows
running open-source embedding models locally. Ollama provides an OpenAI-compatible
API for embeddings.

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import List, Optional

import httpx

from src.libs.embedding.base_embedding import BaseEmbedding, EmbeddingResult
from src.observability.logger import get_logger

logger = get_logger(__name__)


class OllamaEmbedding(BaseEmbedding):
    """
    Ollama Embedding provider implementation.

    This class implements text embeddings using Ollama's local API.
    Ollama allows running open-source embedding models (nomic-embed-text, mxbai-embed-large, etc.)
    on your local machine with an OpenAI-compatible API.

    API Documentation: https://github.com/ollama/ollama/blob/main/docs/api.md

    Common models:
        - nomic-embed-text: 768 dimensions, fast and efficient
        - mxbai-embed-large: 1024 dimensions, high quality
        - all-minilm: 384 dimensions, lightweight

    Example:
        >>> embedding = OllamaEmbedding(model="nomic-embed-text", dimension=768)
        >>> result = embedding.embed(["Hello, world!"])
        >>> print(len(result.embeddings[0]))  # 768
    """

    # Ollama API endpoint
    DEFAULT_BASE_URL = "http://localhost:11434"
    EMBEDDING_ENDPOINT = "/v1/embeddings"  # OpenAI-compatible endpoint
    # Legacy endpoint (for older Ollama versions)
    EMBEDDING_ENDPOINT_LEGACY = "/api/embed"

    # Model-specific dimensions (from Ollama library)
    MODEL_DIMENSIONS = {
        "nomic-embed-text": 768,
        "nomic-embed-text-v1.5": 768,
        "mxbai-embed-large": 1024,
        "mxbai-embed-large-v1": 1024,
        "all-minilm": 384,
        "all-minilm-l6-v2": 384,
    }

    def __init__(
        self,
        model: str,
        dimension: int,
        api_key: Optional[str] = None,  # Not required for Ollama but kept for interface compatibility
        batch_size: int = 32,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Ollama Embedding provider.

        Args:
            model: Model name (e.g., "nomic-embed-text", "mxbai-embed-large")
            dimension: Dimension of the embedding vectors (determined by the model)
            api_key: Not required for Ollama (kept for interface compatibility)
            batch_size: Maximum number of texts to process in one batch
            base_url: Custom base URL (defaults to http://localhost:11434)
            **kwargs: Additional provider-specific parameters
        """
        super().__init__(model, dimension, api_key, batch_size, **kwargs)

        # Use provided base_url or default to localhost
        self.base_url = base_url or self.DEFAULT_BASE_URL

        # Validate dimension matches model (if known)
        if model in self.MODEL_DIMENSIONS:
            expected_dim = self.MODEL_DIMENSIONS[model]
            if dimension != expected_dim:
                logger.warning(
                    f"[{self.provider_name}] Dimension {dimension} doesn't match expected "
                    f"dimension {expected_dim} for model '{model}'. "
                    f"This may cause issues with vector operations."
                )

        # Initialize HTTP client (no auth required for Ollama)
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Content-Type": "application/json",
            },
            timeout=120.0,  # Ollama may need more time on slower machines
        )

        logger.info(
            f"Initialized Ollama Embedding: model='{self.model}', "
            f"dimension={self.dimension}, base_url={self.base_url}"
        )

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "ollama"

    def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> EmbeddingResult:
        """
        Convert texts to embedding vectors using Ollama API.

        This method handles batching internally if the number of texts exceeds
        batch_size. All texts are processed in order.

        Args:
            texts: List of text strings to embed
            **kwargs: Additional Ollama-specific parameters

        Returns:
            EmbeddingResult containing embeddings and metadata

        Raises:
            ValueError: If texts list is empty or contains invalid inputs
            RuntimeError: If the API call fails

        Example:
            >>> result = embedding.embed(["Hello", "World"])
            >>> print(len(result.embeddings))  # 2
            >>> print(len(result.embeddings[0]))  # 768 (for nomic-embed-text)
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
            } if total_tokens > 0 else None
        )

        logger.debug(
            f"[{self.provider_name}] Embedded {len(texts)} texts: "
            f"dimension={self.dimension}"
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
            **kwargs: Additional Ollama-specific parameters

        Returns:
            Tuple of (list of embeddings, total tokens used)

        Raises:
            RuntimeError: If the API call fails
        """
        # Prepare request payload (Ollama uses OpenAI-compatible format)
        payload = {
            "model": self.model,
            "input": texts,
            **kwargs  # Pass through additional parameters
        }

        try:
            # Try OpenAI-compatible endpoint first
            logger.debug(
                f"[{self.provider_name}] Sending embedding request: "
                f"model='{self.model}', texts={len(texts)}"
            )

            response = self._client.post(
                self.EMBEDDING_ENDPOINT,
                json=payload
            )

            # If OpenAI-compatible endpoint fails, try legacy endpoint
            if response.status_code == 404:
                logger.debug(f"[{self.provider_name}] OpenAI-compatible endpoint not found, trying legacy endpoint")
                # Convert to legacy format
                legacy_payload = {
                    "model": self.model,
                    "input": texts[0] if len(texts) == 1 else texts,  # Ollama legacy only supports single text
                }
                response = self._client.post(
                    self.EMBEDDING_ENDPOINT_LEGACY,
                    json=legacy_payload
                )

            response.raise_for_status()

            # Parse response
            data = response.json()

            # Handle both OpenAI-compatible and legacy response formats
            if "data" in data:
                # OpenAI-compatible format
                embeddings = []
                for item in data["data"]:
                    # Sort by index to ensure correct order
                    embeddings.append(item["embedding"])
                total_tokens = data.get("usage", {}).get("total_tokens", 0)
            else:
                # Legacy Ollama format
                embeddings = [data.get("embedding", [])]
                total_tokens = data.get("prompt_eval_count", 0)

            return embeddings, total_tokens

        except httpx.HTTPStatusError as e:
            # Handle HTTP errors with clear error messages
            status_code = e.response.status_code
            error_detail = "Unknown error"

            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", str(e))
            except Exception:
                error_detail = str(e)

            error_msg = (
                f"[{self.provider_name}] API request failed with status {status_code}. "
                f"Model: '{self.model}'. "
                f"Base URL: '{self.base_url}'. "
                f"Error: {error_detail}"
            )

            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except httpx.RequestError as e:
            # Handle network/connection errors
            error_msg = (
                f"[{self.provider_name}] Unable to connect to Ollama server. "
                f"Model: '{self.model}'. "
                f"Base URL: '{self.base_url}'. "
                f"Please ensure Ollama is running and accessible. "
                f"You can start Ollama with: 'ollama serve'. "
                f"You can pull the model with: 'ollama pull {self.model}'. "
                f"Error: {str(e)}"
            )

            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except Exception as e:
            # Handle unexpected errors
            error_msg = (
                f"[{self.provider_name}] Unexpected error during embedding. "
                f"Model: '{self.model}'. "
                f"Base URL: '{self.base_url}'. "
                f"Error type: {type(e).__name__}. "
                f"Error: {str(e)}"
            )

            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def __del__(self):
        """Clean up HTTP client when instance is destroyed."""
        if hasattr(self, '_client'):
            self._client.close()
