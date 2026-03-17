"""
Ollama LLM provider implementation.

This module implements the BaseLLM interface for Ollama, which allows
running open-source LLMs locally. Ollama provides an OpenAI-compatible API.

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import Optional

import httpx

from src.libs.llm.base_llm import BaseLLM, LLMResponse, Message
from src.observability.logger import get_logger

logger = get_logger(__name__)


class OllamaLLM(BaseLLM):
    """
    Ollama LLM provider implementation.

    This class implements chat completions using Ollama's local API.
    Ollama allows running open-source models (llama3, mistral, codellama, etc.)
    on your local machine with an OpenAI-compatible API.

    API Documentation: https://github.com/ollama/ollama/blob/main/docs/api.md

    Common models:
        - llama3: Meta's Llama 3 model (8B, 70B)
        - mistral: Mistral 7B
        - codellama: Code Llama
        - phi3: Microsoft Phi-3
        - qwen2: Alibaba Qwen2

    Example:
        >>> llm = OllamaLLM(model="llama3")
        >>> messages = [Message(role="user", content="Hello!")]
        >>> response = llm.chat(messages)
        >>> print(response.content)
    """

    # Ollama API endpoint
    DEFAULT_BASE_URL = "http://localhost:11434"
    CHAT_COMPLETION_ENDPOINT = "/v1/chat/completions"  # OpenAI-compatible endpoint
    # Legacy endpoint (for older Ollama versions)
    CHAT_COMPLETION_ENDPOINT_LEGACY = "/api/chat"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,  # Not required for Ollama but kept for interface compatibility
        temperature: float = 0.7,
        max_tokens: int = 2048,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Ollama LLM provider.

        Args:
            model: Model name (e.g., "llama3", "mistral", "codellama")
            api_key: Not required for Ollama (kept for interface compatibility)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            base_url: Custom base URL (defaults to http://localhost:11434)
            **kwargs: Additional provider-specific parameters
        """
        super().__init__(model, api_key, temperature, max_tokens, **kwargs)

        # Use provided base_url or default to localhost
        self.base_url = base_url or self.DEFAULT_BASE_URL

        # Initialize HTTP client (no auth required for Ollama)
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Content-Type": "application/json",
            },
            timeout=120.0,  # Ollama may need more time on slower machines
        )

        logger.info(f"Initialized Ollama LLM with model '{self.model}' at {self.base_url}")

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "ollama"

    def chat(
        self,
        messages: list[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send a chat completion request to Ollama API.

        Args:
            messages: List of messages in the conversation
            temperature: Override the default temperature
            max_tokens: Override the default max_tokens (maps to num_predict in Ollama)
            **kwargs: Additional Ollama-specific parameters

        Returns:
            LLMResponse containing the generated text and metadata

        Raises:
            ValueError: If messages list is empty or invalid
            RuntimeError: If the API call fails

        Example:
            >>> messages = [
            ...     Message(role="system", content="You are a helpful assistant."),
            ...     Message(role="user", content="What is the capital of France?")
            ... ]
            >>> response = llm.chat(messages)
            >>> print(response.content)
        """
        # Validate input
        if not messages:
            raise ValueError(
                f"[{self.provider_name}] Messages list cannot be empty. "
                f"Please provide at least one message."
            )

        if not all(isinstance(msg, Message) for msg in messages):
            raise ValueError(
                f"[{self.provider_name}] All items in messages must be Message instances. "
                f"Got types: {[type(msg).__name__ for msg in messages]}"
            )

        # Prepare request payload (Ollama uses OpenAI-compatible format)
        payload = {
            "model": self.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": temperature if temperature is not None else self.temperature,
            # Ollama uses 'num_predict' instead of 'max_tokens'
            "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": False,  # We don't support streaming yet
            **kwargs  # Pass through any additional Ollama parameters
        }

        # Remove None values from payload
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            # Try OpenAI-compatible endpoint first
            logger.debug(
                f"Sending chat completion request to {self.provider_name}: "
                f"model='{self.model}', messages={len(messages)}"
            )

            response = self._client.post(
                self.CHAT_COMPLETION_ENDPOINT,
                json=payload
            )

            # If OpenAI-compatible endpoint fails, try legacy endpoint
            if response.status_code == 404:
                logger.debug(f"OpenAI-compatible endpoint not found, trying legacy endpoint")
                # Convert to legacy format
                legacy_payload = {
                    "model": self.model,
                    "messages": payload["messages"],
                    "options": {
                        "temperature": payload.get("temperature"),
                        "num_predict": payload.get("num_predict"),
                    },
                    "stream": False,
                }
                response = self._client.post(
                    self.CHAT_COMPLETION_ENDPOINT_LEGACY,
                    json=legacy_payload
                )

            response.raise_for_status()

            # Parse response
            data = response.json()

            # Handle both OpenAI-compatible and legacy response formats
            if "choices" in data:
                # OpenAI-compatible format
                choice = data["choices"][0]
                message = choice["message"]
                model_name = data.get("model", self.model)
                usage = data.get("usage", {})
            else:
                # Legacy Ollama format
                message = data.get("message", {})
                model_name = data.get("model", self.model)
                # Ollama legacy doesn't return usage info
                usage = {}

            # Build LLMResponse
            llm_response = LLMResponse(
                content=message.get("content", ""),
                model=model_name,
                provider=self.provider_name,
                usage={
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                },
                raw_response=data
            )

            logger.debug(
                f"Received response from {self.provider_name}: "
                f"{llm_response.usage.get('total_tokens', 'N/A')} total tokens"
            )

            return llm_response

        except httpx.HTTPStatusError as e:
            # Handle HTTP errors with clear error messages
            status_code = e.response.status_code
            error_detail = "Unknown error"

            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", str(e))
            except Exception:
                error_detail = str(e)

            # Sanitize error detail to avoid leaking sensitive config
            error_detail = self._sanitize_error_message(error_detail)

            error_msg = (
                f"[{self.provider_name}] API request failed with status {status_code}. "
                f"Model: '{self.model}'. "
                f"Base URL: '{self.base_url}'. "
                f"Error: {error_detail}"
            )

            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except httpx.RequestError as e:
            # Handle network/connection errors (very common with Ollama)
            error_msg = (
                f"[{self.provider_name}] Unable to connect to Ollama server. "
                f"Model: '{self.model}'. "
                f"Base URL: '{self.base_url}'. "
                f"Please ensure Ollama is running and accessible. "
                f"You can start Ollama with: 'ollama serve'. "
                f"Error: {str(e)}"
            )

            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except Exception as e:
            # Handle unexpected errors
            error_msg = (
                f"[{self.provider_name}] Unexpected error during chat completion. "
                f"Model: '{self.model}'. "
                f"Base URL: '{self.base_url}'. "
                f"Error type: {type(e).__name__}. "
                f"Error: {str(e)}"
            )

            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _sanitize_error_message(self, error_msg: str) -> str:
        """
        Sanitize error message to avoid leaking sensitive configuration.

        This method removes or masks potentially sensitive information like
        API keys, tokens, or internal paths from error messages.

        Args:
            error_msg: Raw error message

        Returns:
            Sanitized error message
        """
        # Remove potential API keys (though Ollama doesn't use them)
        import re

        # Mask common patterns
        patterns_to_mask = [
            (r'Bearer\s+[A-Za-z0-9\._-]+', 'Bearer [REDACTED]'),
            (r'api[_-]?key["\']?\s*[:=]\s*["\']?[A-Za-z0-9\._-]+', 'api_key=[REDACTED]'),
            (r'token["\']?\s*[:=]\s*["\']?[A-Za-z0-9\._-]+', 'token=[REDACTED]'),
        ]

        sanitized = error_msg
        for pattern, replacement in patterns_to_mask:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        return sanitized

    def __del__(self):
        """Clean up HTTP client when instance is destroyed."""
        if hasattr(self, '_client'):
            self._client.close()
