"""
Unit tests for LLM Reranker provider (Task B7.7)

These tests verify that:
1. RerankerFactory can create LLM Reranker provider
2. LLMReranker loads prompt template correctly
3. LLMReranker parses LLM responses correctly
4. LLMReranker handles errors and provides fallback
5. Custom prompts can be injected for testing

Author: Modular RAG MCP Server Project
License: MIT
"""

from unittest.mock import Mock, patch, MagicMock
from typing import List

import pytest

from src.core.settings import LLMConfig, RerankConfig, Settings
from src.libs.llm.base_llm import Message, LLMResponse
from src.libs.reranker.base_reranker import RerankCandidate, RerankResult
from src.libs.reranker.reranker_factory import RerankerFactory
from src.libs.reranker.llm_reranker import LLMReranker


# ===== Test Fixtures =====

def create_reranker_config(
    backend: str = "llm",
    top_m: int = 10,
    model: str = "gpt-4o",
    **kwargs
) -> RerankConfig:
    """Helper to create RerankConfig for testing."""
    return RerankConfig(
        backend=backend,
        top_m=top_m,
        model=model,
        **kwargs
    )


def create_mock_llm_response(ranked_ids: List[str]) -> str:
    """Helper to create a mock LLM response."""
    import json
    return json.dumps(ranked_ids)


# ===== LLM Reranker Tests =====

class TestLLMReranker:
    """Test LLM Reranker provider."""

    def test_factory_creates_llm_reranker(self):
        """Test that RerankerFactory can create LLM Reranker (Acceptance Criteria 1)."""
        config = create_reranker_config(
            backend="llm",
            top_m=5,
            model="gpt-4o"
        )
        settings = Settings(
            llm=self._create_llm_config(),
            embedding=self._create_embedding_config(),
            vector_store=self._create_vector_store_config(),
            retrieval=self._create_retrieval_config(),
            rerank=config,
            evaluation=self._create_evaluation_config(),
            observability=self._create_observability_config(),
            dashboard=self._create_dashboard_config(),
        )

        reranker = RerankerFactory.create(settings)

        assert isinstance(reranker, LLMReranker)
        assert reranker.provider_name == "llm"
        assert reranker.top_k == 5
        assert reranker.model == "gpt-4o"

    def test_llm_reranker_initialization(self):
        """Test LLMReranker initialization with parameters."""
        reranker = LLMReranker(
            top_k=5,
            model="gpt-4o",
            prompt_path="config/prompts/rerank.txt"
        )

        assert reranker.top_k == 5
        assert reranker.model == "gpt-4o"
        assert reranker.prompt_path == "config/prompts/rerank.txt"
        assert reranker.provider_name == "llm"

    def test_llm_reranker_loads_prompt_from_file(self):
        """Test that LLMReranker loads prompt template from file."""
        reranker = LLMReranker(
            top_k=10,
            model="gpt-4o"
        )

        # Should load from config/prompts/rerank.txt
        assert reranker._prompt_template is not None
        assert "{query}" in reranker._prompt_template
        assert "{chunks}" in reranker._prompt_template

    def test_llm_reranker_custom_prompt_override(self):
        """Test that custom prompt overrides file prompt."""
        custom_prompt = "Custom prompt for testing with {query} and {chunks}"
        reranker = LLMReranker(
            top_k=10,
            model="gpt-4o",
            custom_prompt=custom_prompt
        )

        assert reranker._prompt_template == custom_prompt

    def test_llm_reranker_validates_empty_candidates(self):
        """Test that LLMReranker validates candidates list is not empty."""
        reranker = LLMReranker(top_k=10, model="gpt-4o")
        query = "test query"
        candidates = []

        with pytest.raises(ValueError) as exc_info:
            reranker.rerank(query=query, candidates=candidates, settings=self._create_settings())

        assert "cannot be empty" in str(exc_info.value).lower()
        assert "llm" in str(exc_info.value).lower()

    def test_llm_reranker_requires_settings(self):
        """Test that LLMReranker requires settings in kwargs."""
        reranker = LLMReranker(top_k=10, model="gpt-4o")
        query = "test query"
        candidates = [
            RerankCandidate(id="chunk1", content="Content 1", score=0.8)
        ]

        with pytest.raises(ValueError) as exc_info:
            reranker.rerank(query=query, candidates=candidates)

        assert "settings object is required" in str(exc_info.value).lower()

    @patch('src.libs.reranker.llm_reranker.LLMFactory')
    def test_llm_reranker_successful_reranking(self, mock_llm_factory):
        """Test successful reranking with LLM."""
        # Setup mock LLM
        mock_response_content = '["chunk3", "chunk1", "chunk2"]'
        mock_llm_response = LLMResponse(
            content=mock_response_content,
            model="gpt-4o",
            provider="fake"
        )
        mock_llm = Mock()
        mock_llm.chat.return_value = mock_llm_response
        mock_llm_factory.create.return_value = mock_llm

        # Create candidates
        candidates = [
            RerankCandidate(id="chunk1", content="Content about ML", score=0.8),
            RerankCandidate(id="chunk2", content="Content about DL", score=0.7),
            RerankCandidate(id="chunk3", content="Content about ML basics", score=0.6),
        ]

        # Rerank
        reranker = LLMReranker(top_k=10, model="gpt-4o")
        result = reranker.rerank(
            query="machine learning basics",
            candidates=candidates,
            settings=self._create_settings()
        )

        # Verify result
        assert isinstance(result, RerankResult)
        assert len(result.candidates) == 3
        assert result.method == "llm"
        assert result.fallback_triggered is False

        # Verify ranking (chunk3 should be first)
        assert result.candidates[0].id == "chunk3"
        assert result.candidates[0].score > result.candidates[1].score

    @patch('src.libs.reranker.llm_reranker.LLMFactory')
    def test_llm_reranker_trims_to_top_k(self, mock_llm_factory):
        """Test that reranker respects top_k parameter."""
        # Setup mock LLM
        mock_response_content = '["chunk3", "chunk1", "chunk2"]'
        mock_llm_response = LLMResponse(
            content=mock_response_content,
            model="gpt-4o",
            provider="fake"
        )
        mock_llm = Mock()
        mock_llm.chat.return_value = mock_llm_response
        mock_llm_factory.create.return_value = mock_llm

        # Create 5 candidates but only return top 3
        candidates = [
            RerankCandidate(id=f"chunk{i}", content=f"Content {i}", score=0.5)
            for i in range(5)
        ]

        # Rerank with top_k=3
        reranker = LLMReranker(top_k=3, model="gpt-4o")
        result = reranker.rerank(
            query="test query",
            candidates=candidates,
            settings=self._create_settings()
        )

        # Should only return 3 candidates
        assert len(result.candidates) == 3
        assert len(result.scores) == 3

    @patch('src.libs.reranker.llm_reranker.LLMFactory')
    def test_llm_reranker_parses_json_from_markdown(self, mock_llm_factory):
        """Test that LLMReranker extracts JSON from markdown code blocks."""
        # Setup mock LLM with markdown response
        mock_response_content = '''Here are the ranked chunks:

```json
["chunk3", "chunk1", "chunk2"]
```

The ranking is based on relevance.'''
        mock_llm_response = LLMResponse(
            content=mock_response_content,
            model="gpt-4o",
            provider="fake"
        )
        mock_llm = Mock()
        mock_llm.chat.return_value = mock_llm_response
        mock_llm_factory.create.return_value = mock_llm

        candidates = [
            RerankCandidate(id="chunk1", content="Content 1", score=0.8),
            RerankCandidate(id="chunk2", content="Content 2", score=0.7),
            RerankCandidate(id="chunk3", content="Content 3", score=0.6),
        ]

        # Rerank
        reranker = LLMReranker(top_k=10, model="gpt-4o")
        result = reranker.rerank(
            query="test",
            candidates=candidates,
            settings=self._create_settings()
        )

        # Should successfully parse and rank
        assert len(result.candidates) == 3
        assert result.candidates[0].id == "chunk3"
        assert result.fallback_triggered is False

    @patch('src.libs.reranker.llm_reranker.LLMFactory')
    def test_llm_reranker_fallback_on_invalid_json(self, mock_llm_factory):
        """Test that LLMReranker returns fallback on JSON parse error."""
        # Setup mock LLM with invalid response
        mock_response_content = "This is not valid JSON"
        mock_llm_response = LLMResponse(
            content=mock_response_content,
            model="gpt-4o",
            provider="fake"
        )
        mock_llm = Mock()
        mock_llm.chat.return_value = mock_llm_response
        mock_llm_factory.create.return_value = mock_llm

        candidates = [
            RerankCandidate(id="chunk1", content="Content 1", score=0.8),
            RerankCandidate(id="chunk2", content="Content 2", score=0.7),
        ]

        # Rerank
        reranker = LLMReranker(top_k=10, model="gpt-4o")
        result = reranker.rerank(
            query="test",
            candidates=candidates,
            settings=self._create_settings()
        )

        # Should return fallback (original order)
        assert len(result.candidates) == 2
        assert result.fallback_triggered is True
        # Scores should be original scores
        assert result.scores == [0.8, 0.7]

    @patch('src.libs.reranker.llm_reranker.LLMFactory')
    def test_llm_reranker_fallback_on_llm_error(self, mock_llm_factory):
        """Test that LLMReranker returns fallback on LLM error."""
        # Setup mock LLM that raises exception
        mock_llm = Mock()
        mock_llm.chat.side_effect = Exception("LLM API error")
        mock_llm_factory.create.return_value = mock_llm

        candidates = [
            RerankCandidate(id="chunk1", content="Content 1", score=0.8),
            RerankCandidate(id="chunk2", content="Content 2", score=0.7),
        ]

        # Rerank
        reranker = LLMReranker(top_k=10, model="gpt-4o")
        result = reranker.rerank(
            query="test",
            candidates=candidates,
            settings=self._create_settings()
        )

        # Should return fallback (original order)
        assert len(result.candidates) == 2
        assert result.fallback_triggered is True

    @patch('src.libs.reranker.llm_reranker.LLMFactory')
    def test_llm_reranker_handles_missing_candidates_in_response(self, mock_llm_factory):
        """Test that LLMReranker handles candidates not in LLM response."""
        # Setup mock LLM - returns only 2 of 3 candidates
        mock_response_content = '["chunk3", "chunk1"]'
        mock_llm_response = LLMResponse(
            content=mock_response_content,
            model="gpt-4o",
            provider="fake"
        )
        mock_llm = Mock()
        mock_llm.chat.return_value = mock_llm_response
        mock_llm_factory.create.return_value = mock_llm

        candidates = [
            RerankCandidate(id="chunk1", content="Content 1", score=0.8),
            RerankCandidate(id="chunk2", content="Content 2", score=0.7),
            RerankCandidate(id="chunk3", content="Content 3", score=0.6),
        ]

        # Rerank
        reranker = LLMReranker(top_k=10, model="gpt-4o")
        result = reranker.rerank(
            query="test",
            candidates=candidates,
            settings=self._create_settings()
        )

        # Should include chunk2 at the end
        assert len(result.candidates) == 3
        assert result.candidates[0].id == "chunk3"
        assert result.candidates[1].id == "chunk1"
        # chunk2 should be last (not in LLM response)
        assert result.candidates[2].id == "chunk2"

    @patch('src.libs.reranker.llm_reranker.LLMFactory')
    def test_llm_reranker_constructs_prompt_correctly(self, mock_llm_factory):
        """Test that LLMReranker constructs prompt with query and candidates."""
        # Setup mock LLM
        mock_response_content = '["chunk1"]'
        mock_llm_response = LLMResponse(
            content=mock_response_content,
            model="gpt-4o",
            provider="fake"
        )
        mock_llm = Mock()
        mock_llm.chat.return_value = mock_llm_response
        mock_llm_factory.create.return_value = mock_llm

        candidates = [
            RerankCandidate(
                id="chunk1",
                content="Content about machine learning",
                score=0.8,
                metadata={"source": "doc.pdf"}
            ),
        ]

        # Rerank
        reranker = LLMReranker(top_k=10, model="gpt-4o")
        reranker.rerank(
            query="machine learning",
            candidates=candidates,
            settings=self._create_settings()
        )

        # Verify LLM was called
        mock_llm.chat.assert_called_once()
        call_args = mock_llm.chat.call_args

        # Check that prompt contains query and candidate info
        messages = call_args[0][0]  # Get the messages argument
        assert len(messages) == 1
        prompt = messages[0].content

        # Verify prompt contains query
        assert "machine learning" in prompt

        # Verify prompt contains candidate info
        assert "chunk1" in prompt
        assert "Content about machine learning" in prompt

    @patch('src.libs.reranker.llm_reranker.LLMFactory')
    def test_llm_reranker_error_message_for_invalid_schema(self, mock_llm_factory):
        """Test that LLMReranker provides clear error for invalid schema."""
        # Setup mock LLM with non-list response
        mock_response_content = '{"not": "a list"}'
        mock_llm_response = LLMResponse(
            content=mock_response_content,
            model="gpt-4o",
            provider="fake"
        )
        mock_llm = Mock()
        mock_llm.chat.return_value = mock_llm_response
        mock_llm_factory.create.return_value = mock_llm

        candidates = [
            RerankCandidate(id="chunk1", content="Content", score=0.8)
        ]

        # Rerank
        reranker = LLMReranker(top_k=10, model="gpt-4o")
        result = reranker.rerank(
            query="test",
            candidates=candidates,
            settings=self._create_settings()
        )

        # Should return fallback
        assert result.fallback_triggered is True


# ===== Helper Methods for Creating Settings =====

@staticmethod
def _create_llm_config():
    return LLMConfig(provider="fake", model="fake-model")

@staticmethod
def _create_embedding_config():
    from src.core.settings import EmbeddingConfig
    return EmbeddingConfig(provider="fake", model="fake-model", dimension=768)

@staticmethod
def _create_vector_store_config():
    from src.core.settings import VectorStoreConfig
    return VectorStoreConfig(backend="chroma")

@staticmethod
def _create_retrieval_config():
    from src.core.settings import RetrievalConfig
    return RetrievalConfig()

@staticmethod
def _create_rerank_config():
    return RerankConfig(backend="none")

@staticmethod
def _create_evaluation_config():
    from src.core.settings import EvaluationConfig
    return EvaluationConfig()

@staticmethod
def _create_observability_config():
    from src.core.settings import ObservabilityConfig
    return ObservabilityConfig()

@staticmethod
def _create_dashboard_config():
    from src.core.settings import DashboardConfig
    return DashboardConfig()


@staticmethod
def _create_settings():
    """Create a complete Settings object."""
    return Settings(
        llm=TestLLMReranker._create_llm_config(),
        embedding=TestLLMReranker._create_embedding_config(),
        vector_store=TestLLMReranker._create_vector_store_config(),
        retrieval=TestLLMReranker._create_retrieval_config(),
        rerank=TestLLMReranker._create_rerank_config(),
        evaluation=TestLLMReranker._create_evaluation_config(),
        observability=TestLLMReranker._create_observability_config(),
        dashboard=TestLLMReranker._create_dashboard_config(),
    )


# Add helper methods to test class
TestLLMReranker._create_llm_config = _create_llm_config
TestLLMReranker._create_embedding_config = _create_embedding_config
TestLLMReranker._create_vector_store_config = _create_vector_store_config
TestLLMReranker._create_retrieval_config = _create_retrieval_config
TestLLMReranker._create_rerank_config = _create_rerank_config
TestLLMReranker._create_evaluation_config = _create_evaluation_config
TestLLMReranker._create_observability_config = _create_observability_config
TestLLMReranker._create_dashboard_config = _create_dashboard_config
TestLLMReranker._create_settings = _create_settings
