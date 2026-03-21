"""
LLM Reranker provider implementation.

This module implements the BaseReranker interface using an LLM to rerank
retrieved candidates. It reads a prompt template from config/prompts/rerank.txt
and uses an LLM to determine the optimal ordering based on query relevance.

Author: Modular RAG MCP Server Project
License: MIT
"""

import json
import os
import re
from typing import List, Optional

from src.core.settings import Settings
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.base_llm import Message
from src.libs.reranker.base_reranker import (
    BaseReranker,
    RerankCandidate,
    RerankResult,
)
from src.observability.logger import get_logger

logger = get_logger(__name__)


class LLMReranker(BaseReranker):
    """
    LLM-based reranker provider implementation.

    This class uses a language model to intelligently rerank retrieved candidates
    based on their relevance to the query. It constructs a prompt from a template
    file and parses the LLM's response to extract the ranked IDs.

    Key features:
    - Reads prompt template from config/prompts/rerank.txt
    - Supports custom prompt injection (useful for testing)
    - Returns structured output (ranked list of IDs)
    - Fallback mechanism on failure
    - Clear error messages for schema validation

    Example:
        >>> reranker = LLMReranker(top_k=5, model="gpt-4o")
        >>> candidates = [
        ...     RerankCandidate(id="chunk1", content="...", score=0.8),
        ...     RerankCandidate(id="chunk2", content="...", score=0.7),
        ... ]
        >>> result = reranker.rerank(query="machine learning", candidates=candidates)
        >>> print(result.candidates[0].id)  # Most relevant after reranking
        >>> print(result.fallback_triggered)  # False if successful, True if failed
    """

    # Default prompt template (used if file not found)
    DEFAULT_PROMPT = """You are a relevance ranking expert. Your task is to rerank the retrieved document chunks based on their relevance to the query.

## Query
{query}

## Retrieved Chunks
{chunks}

## Instructions

1. Analyze each chunk's relevance to the query
2. Consider semantic similarity, factual correctness, and completeness
3. Rank the chunks from most relevant (1) to least relevant
4. Return ONLY a JSON list with the ranked chunk IDs

## Output Format

Return ONLY a JSON array like this:
```json
["chunk_id_3", "chunk_id_1", "chunk_id_4", "chunk_id_2"]
```

Where the first element is the most relevant chunk.

Do NOT include any explanations, only return the JSON array.
"""

    def __init__(
        self,
        top_k: int = 10,
        model: Optional[str] = None,
        prompt_path: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize LLM Reranker.

        Args:
            top_k: Maximum number of candidates to return after reranking
            model: Model name to use for reranking (from LLM providers)
            prompt_path: Path to prompt template file (default: config/prompts/rerank.txt)
            custom_prompt: Custom prompt to use instead of file (useful for testing)
            **kwargs: Additional provider-specific parameters
        """
        super().__init__(top_k, model, **kwargs)

        # Set default prompt path
        if prompt_path is None:
            prompt_path = "config/prompts/rerank.txt"

        self.prompt_path = prompt_path
        self.custom_prompt = custom_prompt

        # Load prompt template
        self._prompt_template = self._load_prompt_template()

        logger.debug(
            f"Initialized LLMReranker: top_k={top_k}, model={model}, "
            f"prompt_path={prompt_path}, custom_prompt={bool(custom_prompt)}"
        )

    def _load_prompt_template(self) -> str:
        """
        Load prompt template from file or use custom prompt.

        Returns:
            Prompt template string
        """
        # Use custom prompt if provided
        if self.custom_prompt:
            return self.custom_prompt

        # Try to load from file
        if os.path.exists(self.prompt_path):
            try:
                with open(self.prompt_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Skip if file only has TODO
                    if "TODO:" in content or not content.strip():
                        logger.warning(
                            f"[{self.provider_name}] Prompt file contains only TODO, "
                            f"using default prompt"
                        )
                        return self.DEFAULT_PROMPT
                    return content
            except Exception as e:
                logger.error(f"[{self.provider_name}] Failed to load prompt: {e}")
                return self.DEFAULT_PROMPT
        else:
            logger.warning(
                f"[{self.provider_name}] Prompt file not found: {self.prompt_path}, "
                f"using default prompt"
            )
            return self.DEFAULT_PROMPT

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "llm"

    def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        **kwargs
    ) -> RerankResult:
        """
        Rerank candidates using LLM-based ranking.

        This method constructs a prompt with the query and candidates, sends it
        to the LLM, and parses the response to extract the ranked ordering.

        Args:
            query: The search query
            candidates: List of candidates to rerank
            **kwargs: Additional parameters:
                - settings: Settings object (required for LLMFactory)

        Returns:
            RerankResult containing reranked candidates

        Raises:
            ValueError: If candidates list is empty or response doesn't match schema
            RuntimeError: If the reranking operation fails
        """
        if not candidates:
            raise ValueError(
                f"[{self.provider_name}] Candidates list cannot be empty."
            )

        # Extract settings from kwargs
        settings = kwargs.get('settings')
        if not settings:
            raise ValueError(
                f"[{self.provider_name}] Settings object is required in kwargs. "
                f"Pass settings=settings when calling rerank()"
            )

        try:
            # Construct prompt with query and candidates
            prompt = self._construct_prompt(query, candidates)

            # Create LLM and get response
            llm = LLMFactory.create(settings)
            messages = [
                Message(role="user", content=prompt)
            ]

            response = llm.chat(messages).content

            # Parse LLM response to extract ranked IDs
            ranked_ids = self._parse_llm_response(response)

            # Reorder candidates according to LLM ranking
            id_to_candidate = {c.id: c for c in candidates}
            ranked_candidates = []
            scores = []

            for i, candidate_id in enumerate(ranked_ids):
                if candidate_id in id_to_candidate:
                    candidate = id_to_candidate[candidate_id]
                    # Update score based on rank (higher rank = higher score)
                    # Score formula: 1.0 - (rank / total)
                    new_score = 1.0 - (i / len(ranked_ids))
                    # Create new candidate with updated score
                    updated_candidate = RerankCandidate(
                        id=candidate.id,
                        content=candidate.content,
                        score=new_score,
                        metadata=candidate.metadata
                    )
                    ranked_candidates.append(updated_candidate)
                    scores.append(new_score)

            # Handle any candidates not in LLM response (append to end)
            remaining_candidates = [
                c for c in candidates if c.id not in ranked_ids
            ]
            for candidate in remaining_candidates:
                ranked_candidates.append(candidate)
                scores.append(candidate.score)

            # Trim to top_k
            final_candidates = ranked_candidates[:self.top_k]
            final_scores = scores[:self.top_k]

            logger.debug(
                f"[{self.provider_name}] Successfully reranked {len(candidates)} "
                f"candidates to {len(final_candidates)} results"
            )

            return RerankResult(
                candidates=final_candidates,
                scores=final_scores,
                method="llm",
                fallback_triggered=False
            )

        except Exception as e:
            # On failure, return candidates in original order with fallback signal
            error_msg = (
                f"[{self.provider_name}] Failed to rerank with LLM: {str(e)}. "
                f"Returning candidates in original order with fallback."
            )
            logger.warning(error_msg)

            # Return original order with fallback triggered
            trimmed_candidates = candidates[:self.top_k]
            scores = [c.score for c in trimmed_candidates]

            return RerankResult(
                candidates=trimmed_candidates,
                scores=scores,
                method="llm",
                fallback_triggered=True
            )

    def _construct_prompt(self, query: str, candidates: List[RerankCandidate]) -> str:
        """
        Construct prompt from template with query and candidates.

        Args:
            query: Search query
            candidates: List of candidates

        Returns:
            Formatted prompt string
        """
        # Format chunks for the prompt
        chunks_text = ""
        for i, candidate in enumerate(candidates):
            chunks_text += f"\n### Chunk {i + 1}: {candidate.id}\n"
            chunks_text += f"Content: {candidate.content}\n"
            if candidate.metadata:
                chunks_text += f"Metadata: {json.dumps(candidate.metadata, ensure_ascii=False)}\n"
            chunks_text += f"Original Score: {candidate.score:.3f}\n"

        # Replace placeholders in prompt template
        prompt = self._prompt_template.replace("{query}", query)
        prompt = prompt.replace("{chunks}", chunks_text.strip())

        return prompt

    def _parse_llm_response(self, response: str) -> List[str]:
        """
        Parse LLM response to extract ranked IDs.

        Args:
            response: LLM response text

        Returns:
            List of ranked chunk IDs

        Raises:
            ValueError: If response doesn't match expected schema
        """
        # Clean up response text
        response = response.strip()

        # Try to extract JSON array
        try:
            # Method 1: Try to find JSON array in markdown code blocks
            # Pattern to match ```json...``` or ```...```
            json_pattern = r'```(?:json)?\s*\n*(\[.*?\])\n*```'
            json_match = re.search(json_pattern, response, re.DOTALL)

            if json_match:
                response = json_match.group(1)
            else:
                # Method 2: Try to find JSON array directly with brackets
                # Pattern to match [ ... ]
                array_pattern = r'\[.*?\]'
                array_match = re.search(array_pattern, response, re.DOTALL)

                if array_match:
                    response = array_match.group(0)
                else:
                    # Method 3: Remove markdown-style code blocks (old logic)
                    if "```json" in response:
                        parts = response.split("```json")
                        if len(parts) > 1:
                            response = parts[1]
                    elif response.startswith("```"):
                        parts = response.split("```", 1)
                        if len(parts) > 1:
                            response = parts[1]

                    # Remove closing code block
                    if "```" in response:
                        response = response.split("```")[0]

            response = response.strip()

            # Parse JSON array
            ranked_ids = json.loads(response)

            # Validate it's a list of strings
            if not isinstance(ranked_ids, list):
                raise ValueError("Response is not a list")

            # Convert all items to strings
            ranked_ids = [str(item) for item in ranked_ids]

            return ranked_ids

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse LLM response as JSON: {response}. "
                f"Error: {str(e)}. "
                f"Response must be a JSON array of IDs."
            ) from e
        except Exception as e:
            raise ValueError(
                f"Failed to parse LLM response: {str(e)}. "
                f"Response must be a JSON array of IDs."
            ) from e
