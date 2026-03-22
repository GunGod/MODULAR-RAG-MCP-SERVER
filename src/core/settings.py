"""
Configuration loading and validation for the Modular RAG MCP Server.

This module provides the Settings dataclass and functions to load and validate
configuration from config/settings.yaml. All required fields are validated at
startup with clear error messages.

Author: Modular RAG MCP Server Project
License: MIT
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from yaml import YAMLError

from src.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str  # azure | openai | ollama | deepseek
    model: str
    azure_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    api_version: Optional[str] = None  # For Azure OpenAI
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class VisionLLMConfig:
    """Vision LLM provider configuration for multimodal image understanding."""
    provider: str  # azure | dashscope | openai
    model: str
    api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None  # For Azure OpenAI Vision
    api_version: Optional[str] = None  # For Azure OpenAI
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout_seconds: int = 30  # Timeout for image processing


@dataclass
class EmbeddingConfig:
    """Embedding provider configuration."""
    provider: str  # openai | azure | ollama
    model: str
    api_key: Optional[str] = None
    batch_size: int = 32
    dimension: int = 1536  # Default for OpenAI text-embedding-3-small
    azure_endpoint: Optional[str] = None  # For Azure OpenAI
    api_version: Optional[str] = None  # For Azure OpenAI


@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    backend: str  # chroma | qdrant | pinecone
    persist_path: str = "./data/db/chroma"
    collection_name: str = "default"


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    sparse_backend: str = "bm25"  # bm25 | elasticsearch
    fusion_algorithm: str = "rrf"  # rrf | weighted_sum
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_final: int = 10


@dataclass
class RerankConfig:
    """Reranking configuration."""
    backend: str = "none"  # none | cross_encoder | llm
    model: Optional[str] = None
    top_m: int = 30
    timeout_seconds: int = 10


@dataclass
class SplitterConfig:
    """Text splitter configuration for document chunking."""
    provider: str = "recursive"  # recursive | fake | semantic | fixed
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: Optional[list[str]] = None  # Custom separators (optional)

    def __post_init__(self):
        if self.separators is None:
            # Default separators for RecursiveSplitter
            self.separators = ["\n\n", "\n", " ", ""]


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    backends: list[str] = None  # ["ragas", "custom"]
    golden_test_set: str = "./tests/fixtures/golden_test_set.json"

    def __post_init__(self):
        if self.backends is None:
            self.backends = []


@dataclass
class ObservabilityConfig:
    """Observability configuration."""
    enabled: bool = True
    log_file: str = "./logs/traces.jsonl"
    log_level: str = "INFO"


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    enabled: bool = True
    port: int = 8501
    traces_dir: str = "./logs"
    auto_refresh: bool = True
    refresh_interval: int = 5


@dataclass
class Settings:
    """
    Main settings dataclass containing all configuration sections.

    This class only holds the configuration structure and performs minimal
    validation. No network/IO operations should be performed during initialization.
    """
    llm: LLMConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    rerank: RerankConfig
    splitter: SplitterConfig
    vision_llm: VisionLLMConfig
    evaluation: EvaluationConfig
    observability: ObservabilityConfig
    dashboard: DashboardConfig


def validate_settings(settings: Settings) -> None:
    """
    Validate that all required fields are present in the Settings object.

    Args:
        settings: Settings object to validate

    Raises:
        ValueError: If a required field is missing or invalid
    """
    errors = []

    # Validate LLM config
    if not settings.llm.provider:
        errors.append("llm.provider is required")
    if not settings.llm.model:
        errors.append("llm.model is required")
    if settings.llm.provider == "azure" and not settings.llm.azure_endpoint:
        errors.append("llm.azure_endpoint is required when provider is 'azure'")

    # Validate Embedding config
    if not settings.embedding.provider:
        errors.append("embedding.provider is required")
    if not settings.embedding.model:
        errors.append("embedding.model is required")

    # Validate VectorStore config
    if not settings.vector_store.backend:
        errors.append("vector_store.backend is required")

    # Validate Retrieval config
    if settings.retrieval.top_k_dense <= 0:
        errors.append("retrieval.top_k_dense must be positive")
    if settings.retrieval.top_k_sparse <= 0:
        errors.append("retrieval.top_k_sparse must be positive")
    if settings.retrieval.top_k_final <= 0:
        errors.append("retrieval.top_k_final must be positive")

    # Validate Rerank config
    if settings.rerank.backend not in ["none", "cross_encoder", "llm"]:
        errors.append(f"rerank.backend must be one of: none, cross_encoder, llm")
    if settings.rerank.backend == "cross_encoder" and not settings.rerank.model:
        errors.append("rerank.model is required when backend is 'cross_encoder'")

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Configuration validation passed")


def load_settings(path: str = "config/settings.yaml") -> Settings:
    """
    Load settings from a YAML configuration file.

    Args:
        path: Path to the settings.yaml file (default: "config/settings.yaml")

    Returns:
        Settings object with loaded configuration

    Raises:
        FileNotFoundError: If the config file doesn't exist
        YAMLError: If the YAML is malformed
        ValueError: If required fields are missing
    """
    config_path = Path(path)

    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Current working directory: {os.getcwd()}\n"
            f"Please ensure config/settings.yaml exists."
        )

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
    except YAMLError as e:
        logger.error(f"Failed to parse YAML configuration: {e}")
        raise ValueError(f"Invalid YAML in {config_path}: {e}") from e

    if config_data is None:
        logger.error(f"Configuration file is empty: {config_path}")
        raise ValueError(f"Configuration file is empty: {config_path}")

    try:
        # Parse each configuration section
        llm_config = _parse_llm_config(config_data.get("llm", {}))
        embedding_config = _parse_embedding_config(config_data.get("embedding", {}))
        vector_store_config = _parse_vector_store_config(config_data.get("vector_store", {}))
        retrieval_config = _parse_retrieval_config(config_data.get("retrieval", {}))
        rerank_config = _parse_rerank_config(config_data.get("rerank", {}))
        splitter_config = _parse_splitter_config(config_data.get("splitter", {}))
        vision_llm_config = _parse_vision_llm_config(config_data.get("vision_llm", {}))
        evaluation_config = _parse_evaluation_config(config_data.get("evaluation", {}))
        observability_config = _parse_observability_config(config_data.get("observability", {}))
        dashboard_config = _parse_dashboard_config(config_data.get("dashboard", {}))

        settings = Settings(
            llm=llm_config,
            embedding=embedding_config,
            vector_store=vector_store_config,
            retrieval=retrieval_config,
            rerank=rerank_config,
            splitter=splitter_config,
            vision_llm=vision_llm_config,
            evaluation=evaluation_config,
            observability=observability_config,
            dashboard=dashboard_config,
        )

        # Validate the settings
        validate_settings(settings)

        logger.info(f"Successfully loaded configuration from {config_path}")
        return settings

    except KeyError as e:
        logger.error(f"Missing required configuration key: {e}")
        raise ValueError(f"Missing required configuration: {e}") from e
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def _parse_llm_config(data: dict[str, Any]) -> LLMConfig:
    """Parse LLM configuration section."""
    return LLMConfig(
        provider=data.get("provider", "openai"),
        model=data.get("model", "gpt-4o"),
        azure_endpoint=data.get("azure_endpoint"),
        api_key=data.get("api_key"),
        temperature=data.get("temperature", 0.7),
        max_tokens=data.get("max_tokens", 2048),
    )


def _parse_vision_llm_config(data: dict[str, Any]) -> VisionLLMConfig:
    """Parse Vision LLM configuration section."""
    return VisionLLMConfig(
        provider=data.get("provider", "azure"),
        model=data.get("model", "gpt-4o"),
        api_key=data.get("api_key"),
        azure_endpoint=data.get("azure_endpoint"),
        api_version=data.get("api_version"),
        temperature=data.get("temperature", 0.7),
        max_tokens=data.get("max_tokens", 2048),
        timeout_seconds=data.get("timeout_seconds", 30),
    )


def _parse_embedding_config(data: dict[str, Any]) -> EmbeddingConfig:
    """Parse Embedding configuration section."""
    return EmbeddingConfig(
        provider=data.get("provider", "openai"),
        model=data.get("model", "text-embedding-3-small"),
        api_key=data.get("api_key"),
        batch_size=data.get("batch_size", 32),
        dimension=data.get("dimension", 1536),
    )


def _parse_vector_store_config(data: dict[str, Any]) -> VectorStoreConfig:
    """Parse VectorStore configuration section."""
    return VectorStoreConfig(
        backend=data.get("backend", "chroma"),
        persist_path=data.get("persist_path", "./data/db/chroma"),
        collection_name=data.get("collection_name", "default"),
    )


def _parse_retrieval_config(data: dict[str, Any]) -> RetrievalConfig:
    """Parse Retrieval configuration section."""
    return RetrievalConfig(
        sparse_backend=data.get("sparse_backend", "bm25"),
        fusion_algorithm=data.get("fusion_algorithm", "rrf"),
        top_k_dense=data.get("top_k_dense", 20),
        top_k_sparse=data.get("top_k_sparse", 20),
        top_k_final=data.get("top_k_final", 10),
    )


def _parse_rerank_config(data: dict[str, Any]) -> RerankConfig:
    """Parse Rerank configuration section."""
    return RerankConfig(
        backend=data.get("backend", "none"),
        model=data.get("model"),
        top_m=data.get("top_m", 30),
        timeout_seconds=data.get("timeout_seconds", 10),
    )


def _parse_splitter_config(data: dict[str, Any]) -> SplitterConfig:
    """Parse Splitter configuration section."""
    return SplitterConfig(
        provider=data.get("provider", "recursive"),
        chunk_size=data.get("chunk_size", 1000),
        chunk_overlap=data.get("chunk_overlap", 200),
        separators=data.get("separators"),
    )


def _parse_evaluation_config(data: dict[str, Any]) -> EvaluationConfig:
    """Parse Evaluation configuration section."""
    return EvaluationConfig(
        backends=data.get("backends", []),
        golden_test_set=data.get("golden_test_set", "./tests/fixtures/golden_test_set.json"),
    )


def _parse_observability_config(data: dict[str, Any]) -> ObservabilityConfig:
    """Parse Observability configuration section."""
    return ObservabilityConfig(
        enabled=data.get("enabled", True),
        log_file=data.get("log_file", "./logs/traces.jsonl"),
        log_level=data.get("log_level", "INFO"),
    )


def _parse_dashboard_config(data: dict[str, Any]) -> DashboardConfig:
    """Parse Dashboard configuration section."""
    return DashboardConfig(
        enabled=data.get("enabled", True),
        port=data.get("port", 8501),
        traces_dir=data.get("traces_dir", "./logs"),
        auto_refresh=data.get("auto_refresh", True),
        refresh_interval=data.get("refresh_interval", 5),
    )
