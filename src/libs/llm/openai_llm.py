"""
OpenAI LLM provider implementation.

This module implements the BaseLLM interface for OpenAI's API.
It supports chat completions using the OpenAI-compatible API format.

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import Optional

import httpx

from src.core.settings import Settings
from src.libs.llm.base_llm import BaseLLM, LLMResponse, Message
from src.observability.logger import get_logger

logger = get_logger(__name__)


class OpenAILLM(BaseLLM):
    """
    OpenAI LLM provider implementation.

    This class implements chat completions using OpenAI's API.
    It supports models like GPT-4o, GPT-4-turbo, and GPT-3.5-turbo.

    Example:
        >>> llm = OpenAILLM(model="gpt-4o", api_key="sk-...")
        >>> messages = [Message(role="user", content="Hello!")]
        >>> response = llm.chat(messages)
        >>> print(response.content)
    """

    # OpenAI API endpoint
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    CHAT_COMPLETION_ENDPOINT = "/chat/completions"

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
        Initialize OpenAI LLM provider.

        Args:
            model: Model name (e.g., "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            base_url: Custom base URL (defaults to OpenAI's API endpoint)
            **kwargs: Additional provider-specific parameters
        """
        super().__init__(model, api_key, temperature, max_tokens, **kwargs)

        # Use provided base_url or default to OpenAI's endpoint
        self.base_url = base_url or self.DEFAULT_BASE_URL

        # Validate API key
        if not self.api_key:
            raise ValueError(
                f"OpenAI API key is required. "
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

        logger.info(f"Initialized OpenAI LLM with model '{self.model}'")

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "openai"

    def chat(
        self,
        messages: list[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send a chat completion request to OpenAI API.

        Args:
            messages: List of messages in the conversation
            temperature: Override the default temperature
            max_tokens: Override the default max_tokens
            **kwargs: Additional OpenAI-specific parameters (e.g., top_p, presence_penalty)

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

        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            **kwargs  # Pass through any additional OpenAI parameters
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

            # Parse response
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
