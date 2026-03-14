"""
Fake LLM implementation for testing purposes.

This module provides a mock LLM implementation that returns predefined responses
without making any API calls. It's used for testing the factory routing logic
and for development when actual LLM APIs are not available.

Author: Modular RAG MCP Server Project
License: MIT
"""

from typing import Optional

from src.libs.llm.base_llm import BaseLLM, LLMResponse, Message


class FakeLLM(BaseLLM):
    """
    Fake LLM implementation for testing.

    This LLM does not make any actual API calls. Instead, it returns
    predefined responses based on the input messages. It's useful for:
    - Testing the LLMFactory routing logic
    - Development without API credentials
    - Unit tests that don't require external dependencies
    """

    def __init__(
        self,
        model: str = "fake-model",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ):
        """
        Initialize the Fake LLM.

        Args:
            model: Model name (default: "fake-model")
            api_key: Not used, kept for interface compatibility
            temperature: Not used, kept for interface compatibility
            max_tokens: Not used, kept for interface compatibility
            **kwargs: Additional ignored parameters
        """
        super().__init__(model, api_key, temperature, max_tokens, **kwargs)
        self._call_count = 0

    def chat(
        self,
        messages: list[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a fake response.

        The response is based on the last user message. If no user message
        is present, returns a default response.

        Args:
            messages: List of messages in the conversation
            temperature: Ignored (kept for interface compatibility)
            max_tokens: Ignored (kept for interface compatibility)
            **kwargs: Additional ignored parameters

        Returns:
            LLMResponse with fake content
        """
        self._call_count += 1

        # Extract the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.role == "user":
                user_message = msg.content
                break

        # Generate fake response
        if user_message:
            response_content = f"[FakeLLM Response #{self._call_count}] I received your message: '{user_message[:50]}...'"
        else:
            response_content = f"[FakeLLM Response #{self._call_count}] Hello! I'm a fake LLM for testing."

        return LLMResponse(
            content=response_content,
            model=self.model,
            provider=self.provider_name,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "fake"

    def get_call_count(self) -> int:
        """
        Get the number of times chat() has been called.

        Useful for testing to verify the LLM was invoked.

        Returns:
            Number of chat() calls
        """
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the call counter to zero."""
        self._call_count = 0
