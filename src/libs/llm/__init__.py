"""
LLM provider implementations for Modular RAG MCP Server.

This package provides a pluggable LLM layer with support for multiple providers:
- OpenAI (GPT-4o, GPT-4-turbo, GPT-3.5-turbo)
- Azure OpenAI (GPT models hosted on Azure)
- Azure OpenAI Vision (multimodal image understanding)
- DeepSeek (deepseek-chat, deepseek-coder)
- Ollama (local models, coming in B7.2)
- Fake (for testing)

Author: Modular RAG MCP Server Project
License: MIT
"""

from src.libs.llm.base_llm import BaseLLM, LLMResponse, Message
from src.libs.llm.base_vision_llm import (
    BaseVisionLLM,
    VisionResponse,
    VisionMessage,
    ImageContent,
    ImageType,
)
from src.libs.llm.fake_llm import FakeLLM
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.openai_llm import OpenAILLM
from src.libs.llm.azure_llm import AzureLLM
from src.libs.llm.azure_vision_llm import AzureVisionLLM
from src.libs.llm.deepseek_llm import DeepSeekLLM
from src.libs.llm.glm_llm import GLMLLM
from src.libs.llm.ollama_llm import OllamaLLM

__all__ = [
    # Base interfaces
    "BaseLLM",
    "LLMResponse",
    "Message",
    # Vision LLM interfaces
    "BaseVisionLLM",
    "VisionResponse",
    "VisionMessage",
    "ImageContent",
    "ImageType",
    # Factory
    "LLMFactory",
    # Providers
    "FakeLLM",
    "OpenAILLM",
    "AzureLLM",
    "AzureVisionLLM",
    "DeepSeekLLM",
    "GLMLLM",
    "OllamaLLM",
]
