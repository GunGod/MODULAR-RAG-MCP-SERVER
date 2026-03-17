"""
DeepSeek LLM provider implementation.

This module implements the BaseLLM interface for DeepSeek's API.
DeepSeek provides OpenAI-compatible API endpoints for their models.

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import Optional

import httpx

from src.libs.llm.base_llm import BaseLLM, LLMResponse, Message
from src.observability.logger import get_logger

logger = get_logger(__name__)


class DeepSeekLLM(BaseLLM):
    """
    DeepSeek LLM provider implementation.

    This class implements chat completions using DeepSeek's API.
    DeepSeek's API is compatible with OpenAI's API format.

    Supported models:
        - deepseek-chat: DeepSeek's main chat model
        - deepseek-coder: Specialized for code generation

    API Documentation: https://platform.deepseek.com/api-docs/

    Example:
        >>> llm = DeepSeekLLM(model="deepseek-chat", api_key="sk-...")
        >>> messages = [Message(role="user", content="Hello!")]
        >>> response = llm.chat(messages)
        >>> print(response.content)
    """

    # DeepSeek API endpoint
    DEFAULT_BASE_URL = "https://api.deepseek.com"
    CHAT_COMPLETION_ENDPOINT = "/v1/chat/completions"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize DeepSeek LLM provider.

        Args:
            model: Model name (e.g., "deepseek-chat", "deepseek-coder")
            api_key: DeepSeek API key
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate (DeepSeek supports up to 64K)
            base_url: Custom base URL (defaults to DeepSeek's API endpoint)
            **kwargs: Additional provider-specific parameters
        """
        super().__init__(model, api_key, temperature, max_tokens, **kwargs)

        # Use provided base_url or default to DeepSeek's endpoint
        self.base_url = base_url or self.DEFAULT_BASE_URL

        # Validate API key
        if not self.api_key:
            raise ValueError(
                f"[{self.provider_name}] API key is required. "
                f"Provide it via api_key parameter or DEEPSEEK_API_KEY environment variable."
            )

        # Initialize HTTP client
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,  # DeepSeek may need more time for complex reasoning
        )

        logger.info(f"Initialized DeepSeek LLM with model '{self.model}'")

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "deepseek"

    def chat(
        self,
        messages: list[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send a chat completion request to DeepSeek API.

        Args:
            messages: List of messages in the conversation
            temperature: Override the default temperature
            max_tokens: Override the default max_tokens
            **kwargs: Additional DeepSeek-specific parameters

        Returns:
            LLMResponse containing the generated text and metadata

        Raises:
            ValueError: If messages list is empty or invalid
            RuntimeError: If the API call fails

        Example:
            >>> messages = [
            ...     Message(role="system", content="You are a helpful assistant."),
            ...     Message(role="user", content="Explain quantum computing")
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

        # Prepare request payload (DeepSeek uses OpenAI-compatible format)
        payload = {
            "model": self.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            **kwargs  # Pass through any additional DeepSeek parameters
        }

        # Remove None values from payload
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            # Make API request
            logger.debug(
                f"Sending chat completion request to {self.provider_name}: "
                f"model='{self.model}', messages={len(messages)}"
            )

            response = self._client.post(
                self.CHAT_COMPLETION_ENDPOINT,
                json=payload
            )
            response.raise_for_status()

            # Parse response (DeepSeek response format matches OpenAI)
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]

            # Build LLMResponse
            llm_response = LLMResponse(
                content=message["content"],
                model=data["model"],
                provider=self.provider_name,
                usage={
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
                    "completion_tokens": data.get("usage", {}).get("completion_tokens"),
                    "total_tokens": data.get("usage", {}).get("total_tokens"),
                    "prompt_cache_hit_tokens": data.get("usage", {}).get("prompt_cache_hit_tokens"),
                    "prompt_cache_miss_tokens": data.get("usage", {}).get("prompt_cache_miss_tokens"),
                },
                raw_response=data
            )

            logger.debug(
                f"Received response from {self.provider_name}: "
                f"{llm_response.usage.get('total_tokens')} total tokens"
            )

            return llm_response

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
                f"[{self.provider_name}] Unexpected error during chat completion. "
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
