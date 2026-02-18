"""Wrapper for integrating hypothesis engine with latent reasoning.

This module provides a wrapper that checks the hypothesis engine before
invoking the LLM-based latent reasoning. If the hypothesis engine can
answer with low entropy, it bypasses the LLM.
"""

from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.memory_types import Hypothesis, SuperpositionalState
from nanobot.memory.hypothesis_engine import HypothesisEngine


class MemoryAwareReasoner:
    """Wrapper that tries hypothesis engine before LLM-based reasoning.

    This class implements a two-stage reasoning approach:
    1. First, query the relational cache via HypothesisEngine
    2. If entropy is low, use cache-based hypotheses
    3. If entropy is high, fall back to LLM-based reasoning

    Attributes:
        workspace: Path to workspace directory
        entropy_threshold: Threshold for LLM invocation
        hypothesis_engine: HypothesisEngine instance
    """

    def __init__(
        self,
        workspace: Path | None = None,
        memory_config: dict[str, Any] | None = None
    ):
        """Initialize memory-aware reasoner.

        Args:
            workspace: Path to workspace directory (if None, uses default)
            memory_config: Optional memory configuration
        """
        self.memory_config = memory_config or {}
        self.entropy_threshold = float(
            self.memory_config.get("clarify_entropy_threshold", 0.8)
        )

        # Only initialize if workspace is provided
        self.hypothesis_engine = None
        if workspace:
            try:
                self.hypothesis_engine = HypothesisEngine(
                    workspace,
                    entropy_threshold=self.entropy_threshold
                )
                logger.info("HypothesisEngine initialized for memory-aware reasoning")
            except Exception as e:
                logger.warning(f"Failed to initialize HypothesisEngine: {e}")

    def check_memory_first(
        self,
        user_message: str,
        max_hypotheses: int = 3
    ) -> tuple[bool, SuperpositionalState | None]:
        """Check if hypothesis engine can answer the query.

        Args:
            user_message: User's input message
            max_hypotheses: Maximum hypotheses to generate

        Returns:
            Tuple of (can_answer, state) where:
            - can_answer: True if entropy is below threshold
            - state: SuperpositionalState if can answer, None otherwise
        """
        if not self.hypothesis_engine:
            return False, None

        try:
            # Query the hypothesis engine
            result = self.hypothesis_engine.generate_hypotheses(
                user_message,
                max_hypotheses=max_hypotheses
            )

            # Check if we have hypotheses and low entropy
            if result["hypotheses"] and not result["requires_llm"]:
                # Convert hypothesis engine format to SuperpositionalState
                hypotheses = []
                for hyp in result["hypotheses"]:
                    hypothesis = Hypothesis(
                        intent=hyp["intent"],
                        confidence=hyp["confidence"],
                        reasoning=hyp["reasoning"]
                    )
                    hypotheses.append(hypothesis)

                state = SuperpositionalState(
                    hypotheses=hypotheses,
                    entropy=result["entropy"],
                    strategic_direction=self._get_strategic_direction(result)
                )

                logger.info(
                    f"Memory cache answered query with entropy={result['entropy']:.3f}, "
                    f"bypassing LLM"
                )

                return True, state

            # High entropy or no hypotheses - need LLM
            logger.debug(
                f"Memory cache entropy={result['entropy']:.3f} >= threshold={self.entropy_threshold}, "
                "invoking LLM"
            )
            return False, None

        except Exception as e:
            logger.warning(f"Error checking memory cache: {e}")
            return False, None

    def _get_strategic_direction(self, result: dict[str, Any]) -> str:
        """Generate strategic direction from hypothesis engine result.

        Args:
            result: Result from hypothesis engine

        Returns:
            Strategic direction string
        """
        if not result["hypotheses"]:
            return "No cached information available"

        top_hyp = result["hypotheses"][0]
        query_type = result.get("query_type", "unknown")

        if query_type == "comparison":
            return f"Direct comparison from cache: {top_hyp['result']}"
        elif query_type == "attribute":
            return f"Direct attribute lookup from cache: {top_hyp['result']}"
        elif query_type == "relationship":
            return f"Relationship found in cache: {top_hyp['result']}"
        else:
            return f"Cache match found: {top_hyp['result']}"


def wrap_latent_reasoner_with_memory(
    reasoner,
    workspace: Path | None = None,
    memory_config: dict[str, Any] | None = None
):
    """Wrap an existing LatentReasoner with memory-aware reasoning.

    This function modifies a LatentReasoner instance to check the hypothesis
    engine before invoking the LLM.

    Args:
        reasoner: LatentReasoner instance to wrap
        workspace: Path to workspace directory
        memory_config: Optional memory configuration

    Returns:
        Modified reasoner with memory-aware reasoning
    """
    memory_reasoner = MemoryAwareReasoner(workspace, memory_config)

    # Save original reason method
    original_reason = reasoner.reason

    # Define new reason method that checks memory first
    async def memory_aware_reason(user_message: str, context_summary: str):
        # Try memory cache first
        can_answer, state = memory_reasoner.check_memory_first(user_message)

        if can_answer and state:
            return state

        # Fall back to original LLM-based reasoning
        return await original_reason(user_message, context_summary)

    # Replace reason method
    reasoner.reason = memory_aware_reason

    return reasoner
