"""
Unit tests for configuration loading and validation (Task A3)

These tests verify that the Settings loader correctly:
1. Loads configuration from config/settings.yaml
2. Validates required fields
3. Provides clear error messages for missing/invalid configuration
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.core.settings import (
    Settings,
    EvaluationConfig,
    LLMConfig,
    load_settings,
    validate_settings,
    _parse_llm_config,
    _parse_embedding_config,
    _parse_vector_store_config,
    _parse_retrieval_config,
    _parse_rerank_config,
    _parse_evaluation_config,
    _parse_observability_config,
    _parse_dashboard_config,
)


class TestLoadSettings:
    """Test the load_settings function."""

    def test_load_settings_from_default_path(self):
        """Test loading settings from the default config path."""
        # This test assumes config/settings.yaml exists
        settings = load_settings("config/settings.yaml")
        assert isinstance(settings, Settings)
        assert settings.llm.provider in ["azure", "openai", "ollama", "deepseek"]
        assert settings.embedding.provider in ["openai", "azure", "ollama"]
        assert settings.vector_store.backend == "chroma"

    def test_load_settings_from_nonexistent_path_raises_error(self):
        """Test that loading from a nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_settings("config/nonexistent.yaml")
        assert "Configuration file not found" in str(exc_info.value)

    def test_load_settings_from_invalid_yaml_raises_error(self, tmp_path):
        """Test that loading invalid YAML raises ValueError."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: content: [")

        with pytest.raises(ValueError) as exc_info:
            load_settings(str(invalid_yaml))
        assert "Invalid YAML" in str(exc_info.value)

    def test_load_settings_from_empty_file_raises_error(self, tmp_path):
        """Test that loading an empty config file raises ValueError."""
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("")

        with pytest.raises(ValueError) as exc_info:
            load_settings(str(empty_yaml))
        assert "Configuration file is empty" in str(exc_info.value)

    def test_load_settings_with_valid_config(self, tmp_path):
        """Test loading a valid minimal configuration."""
        config_data = {
            "llm": {"provider": "openai", "model": "gpt-4o"},
            "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
            "vector_store": {"backend": "chroma"},
            "retrieval": {"top_k_dense": 20, "top_k_sparse": 20, "top_k_final": 10},
            "rerank": {"backend": "none"},
            "evaluation": {"backends": []},
            "observability": {"enabled": True},
            "dashboard": {"enabled": True},
        }

        config_file = tmp_path / "valid_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        settings = load_settings(str(config_file))
        assert settings.llm.provider == "openai"
        assert settings.llm.model == "gpt-4o"
        assert settings.embedding.provider == "openai"


class TestValidateSettings:
    """Test the validate_settings function."""

    def test_validate_settings_with_all_required_fields(self):
        """Test validation passes with all required fields present."""
        settings = Settings(
            llm=LLMConfig(provider="openai", model="gpt-4o"),
            embedding=_parse_embedding_config({"provider": "openai", "model": "text-embedding-3-small"}),
            vector_store=_parse_vector_store_config({"backend": "chroma"}),
            retrieval=_parse_retrieval_config({}),
            rerank=_parse_rerank_config({}),
            evaluation=_parse_evaluation_config({}),
            observability=_parse_observability_config({}),
            dashboard=_parse_dashboard_config({}),
        )
        # Should not raise any exception
        validate_settings(settings)

    def test_validate_settings_missing_llm_provider(self):
        """Test validation fails when llm.provider is missing."""
        settings = Settings(
            llm=LLMConfig(provider="", model="gpt-4o"),
            embedding=_parse_embedding_config({"provider": "openai", "model": "text-embedding-3-small"}),
            vector_store=_parse_vector_store_config({"backend": "chroma"}),
            retrieval=_parse_retrieval_config({}),
            rerank=_parse_rerank_config({}),
            evaluation=_parse_evaluation_config({}),
            observability=_parse_observability_config({}),
            dashboard=_parse_dashboard_config({}),
        )
        with pytest.raises(ValueError) as exc_info:
            validate_settings(settings)
        assert "llm.provider is required" in str(exc_info.value)

    def test_validate_settings_missing_llm_model(self):
        """Test validation fails when llm.model is missing."""
        settings = Settings(
            llm=LLMConfig(provider="openai", model=""),
            embedding=_parse_embedding_config({"provider": "openai", "model": "text-embedding-3-small"}),
            vector_store=_parse_vector_store_config({"backend": "chroma"}),
            retrieval=_parse_retrieval_config({}),
            rerank=_parse_rerank_config({}),
            evaluation=_parse_evaluation_config({}),
            observability=_parse_observability_config({}),
            dashboard=_parse_dashboard_config({}),
        )
        with pytest.raises(ValueError) as exc_info:
            validate_settings(settings)
        assert "llm.model is required" in str(exc_info.value)

    def test_validate_settings_azure_requires_endpoint(self):
        """Test validation fails when Azure provider lacks endpoint."""
        settings = Settings(
            llm=LLMConfig(provider="azure", model="gpt-4o", azure_endpoint=None),
            embedding=_parse_embedding_config({"provider": "openai", "model": "text-embedding-3-small"}),
            vector_store=_parse_vector_store_config({"backend": "chroma"}),
            retrieval=_parse_retrieval_config({}),
            rerank=_parse_rerank_config({}),
            evaluation=_parse_evaluation_config({}),
            observability=_parse_observability_config({}),
            dashboard=_parse_dashboard_config({}),
        )
        with pytest.raises(ValueError) as exc_info:
            validate_settings(settings)
        assert "llm.azure_endpoint is required" in str(exc_info.value)

    def test_validate_settings_missing_embedding_provider(self):
        """Test validation fails when embedding.provider is missing."""
        settings = Settings(
            llm=LLMConfig(provider="openai", model="gpt-4o"),
            embedding=_parse_embedding_config({"provider": "", "model": "text-embedding-3-small"}),
            vector_store=_parse_vector_store_config({"backend": "chroma"}),
            retrieval=_parse_retrieval_config({}),
            rerank=_parse_rerank_config({}),
            evaluation=_parse_evaluation_config({}),
            observability=_parse_observability_config({}),
            dashboard=_parse_dashboard_config({}),
        )
        with pytest.raises(ValueError) as exc_info:
            validate_settings(settings)
        assert "embedding.provider is required" in str(exc_info.value)

    def test_validate_settings_missing_embedding_model(self):
        """Test validation fails when embedding.model is missing."""
        settings = Settings(
            llm=LLMConfig(provider="openai", model="gpt-4o"),
            embedding=_parse_embedding_config({"provider": "openai", "model": ""}),
            vector_store=_parse_vector_store_config({"backend": "chroma"}),
            retrieval=_parse_retrieval_config({}),
            rerank=_parse_rerank_config({}),
            evaluation=_parse_evaluation_config({}),
            observability=_parse_observability_config({}),
            dashboard=_parse_dashboard_config({}),
        )
        with pytest.raises(ValueError) as exc_info:
            validate_settings(settings)
        assert "embedding.model is required" in str(exc_info.value)

    def test_validate_settings_invalid_rerank_backend(self):
        """Test validation fails with invalid rerank backend."""
        settings = Settings(
            llm=LLMConfig(provider="openai", model="gpt-4o"),
            embedding=_parse_embedding_config({"provider": "openai", "model": "text-embedding-3-small"}),
            vector_store=_parse_vector_store_config({"backend": "chroma"}),
            retrieval=_parse_retrieval_config({}),
            rerank=_parse_rerank_config({"backend": "invalid_backend"}),
            evaluation=_parse_evaluation_config({}),
            observability=_parse_observability_config({}),
            dashboard=_parse_dashboard_config({}),
        )
        with pytest.raises(ValueError) as exc_info:
            validate_settings(settings)
        assert "rerank.backend must be one of" in str(exc_info.value)

    def test_validate_settings_cross_encoder_requires_model(self):
        """Test validation fails when cross_encoder backend lacks model."""
        settings = Settings(
            llm=LLMConfig(provider="openai", model="gpt-4o"),
            embedding=_parse_embedding_config({"provider": "openai", "model": "text-embedding-3-small"}),
            vector_store=_parse_vector_store_config({"backend": "chroma"}),
            retrieval=_parse_retrieval_config({}),
            rerank=_parse_rerank_config({"backend": "cross_encoder", "model": None}),
            evaluation=_parse_evaluation_config({}),
            observability=_parse_observability_config({}),
            dashboard=_parse_dashboard_config({}),
        )
        with pytest.raises(ValueError) as exc_info:
            validate_settings(settings)
        assert "rerank.model is required" in str(exc_info.value)


class TestConfigParsers:
    """Test the individual configuration parser functions."""

    def test_parse_llm_config(self):
        """Test LLM config parsing."""
        data = {"provider": "azure", "model": "gpt-4o", "azure_endpoint": "https://example.com"}
        config = _parse_llm_config(data)
        assert config.provider == "azure"
        assert config.model == "gpt-4o"
        assert config.azure_endpoint == "https://example.com"

    def test_parse_llm_config_with_defaults(self):
        """Test LLM config parsing with defaults."""
        config = _parse_llm_config({})
        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048

    def test_parse_embedding_config(self):
        """Test Embedding config parsing."""
        data = {"provider": "openai", "model": "text-embedding-3-small", "batch_size": 64}
        config = _parse_embedding_config(data)
        assert config.provider == "openai"
        assert config.model == "text-embedding-3-small"
        assert config.batch_size == 64

    def test_parse_embedding_config_with_defaults(self):
        """Test Embedding config parsing with defaults."""
        config = _parse_embedding_config({})
        assert config.provider == "openai"
        assert config.model == "text-embedding-3-small"
        assert config.batch_size == 32
        assert config.dimension == 1536

    def test_parse_vector_store_config(self):
        """Test VectorStore config parsing."""
        data = {"backend": "chroma", "persist_path": "/tmp/chroma"}
        config = _parse_vector_store_config(data)
        assert config.backend == "chroma"
        assert config.persist_path == "/tmp/chroma"

    def test_parse_retrieval_config(self):
        """Test Retrieval config parsing."""
        data = {"top_k_dense": 30, "top_k_sparse": 30, "top_k_final": 15}
        config = _parse_retrieval_config(data)
        assert config.top_k_dense == 30
        assert config.top_k_sparse == 30
        assert config.top_k_final == 15

    def test_parse_rerank_config(self):
        """Test Rerank config parsing."""
        data = {"backend": "cross_encoder", "model": "ms-marco"}
        config = _parse_rerank_config(data)
        assert config.backend == "cross_encoder"
        assert config.model == "ms-marco"

    def test_parse_evaluation_config(self):
        """Test Evaluation config parsing."""
        data = {"backends": ["ragas", "custom"]}
        config = _parse_evaluation_config(data)
        assert config.backends == ["ragas", "custom"]

    def test_parse_observability_config(self):
        """Test Observability config parsing."""
        data = {"enabled": True, "log_file": "/tmp/traces.jsonl"}
        config = _parse_observability_config(data)
        assert config.enabled is True
        assert config.log_file == "/tmp/traces.jsonl"

    def test_parse_dashboard_config(self):
        """Test Dashboard config parsing."""
        data = {"port": 8502, "auto_refresh": False}
        config = _parse_dashboard_config(data)
        assert config.port == 8502
        assert config.auto_refresh is False
