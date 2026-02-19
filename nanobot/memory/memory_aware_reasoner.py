"""Wrapper for integrating hypothesis engine with latent reasoning.

This module provides a wrapper that checks the hypothesis engine before
invoking the LLM-based latent reasoning. If the hypothesis engine can
answer with low entropy, it bypasses the LLM.
"""

from pathlib import Path
from typing import Any, Optional

from loguru import logger

from nanobot.agent.memory_types import Hypothesis, SuperpositionalState
from nanobot.memory.hypothesis_engine import HypothesisEngine


class MemoryAwareReasoner:
    """Wrapper that tries hypothesis engine before LLM-based reasoning.

    This class implements a two-stage reasoning approach:
    1. First, query the relational cache via HypothesisEngine (v1) or MemoryFirstReasonerV2 (v2)
    2. If entropy is low, use cache-based hypotheses
    3. If entropy is high, fall back to LLM-based reasoning

    Attributes:
        workspace: Path to workspace directory
        entropy_threshold: Threshold for LLM invocation
        hypothesis_engine: HypothesisEngine instance (v1)
        use_memory_v2: Flag to enable v2 memory-first reasoning
        reasoner_v2: MemoryFirstReasonerV2 instance (v2)
        cache_v2: RelationalCacheV2 instance (v2)
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
        self.use_memory_v2 = self.memory_config.get("use_memory_v2", False)

        # Only initialize if workspace is provided
        self.hypothesis_engine = None
        self.reasoner_v2 = None
        self.cache_v2 = None
        self.deterministic_agent = None
        
        if workspace:
            if self.use_memory_v2:
                # Initialize v2 components
                try:
                    from nanobot.memory.relational_cache_v2 import RelationalCacheV2
                    from nanobot.memory.memory_first_reasoner_v2 import MemoryFirstReasonerV2
                    from nanobot.memory.deterministic_agent import DeterministicReasoningAgent
                    
                    self.cache_v2 = RelationalCacheV2()
                    self.reasoner_v2 = MemoryFirstReasonerV2(cache=self.cache_v2)
                    self.deterministic_agent = DeterministicReasoningAgent(cache=self.cache_v2)
                    logger.info("MemoryFirstReasonerV2 initialized for memory-aware reasoning (v2)")
                except Exception as e:
                    logger.warning(f"Failed to initialize v2 reasoner: {e}")
            else:
                # Initialize v1 components
                try:
                    self.hypothesis_engine = HypothesisEngine(
                        workspace,
                        entropy_threshold=self.entropy_threshold
                    )
                    logger.info("HypothesisEngine initialized for memory-aware reasoning (v1)")
                except Exception as e:
                    logger.warning(f"Failed to initialize HypothesisEngine: {e}")

    def check_memory_first(
        self,
        user_message: str,
        max_hypotheses: int = 3
    ) -> tuple[bool, Optional[SuperpositionalState]]:
        """Check if hypothesis engine can answer the query.

        Args:
            user_message: User's input message
            max_hypotheses: Maximum hypotheses to generate

        Returns:
            Tuple of (can_answer, state) where:
            - can_answer: True if entropy is below threshold or v2 can answer deterministically
            - state: SuperpositionalState if can answer, None otherwise
        """
        if self.use_memory_v2:
            return self._check_memory_v2(user_message)
        
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
    
    def _check_memory_v2(self, user_message: str) -> tuple[bool, Optional[SuperpositionalState]]:
        """Check if v2 memory-first reasoner can answer the query.
        
        Args:
            user_message: User's input message
            
        Returns:
            Tuple of (can_answer, state) where can_answer is True if v2 has a deterministic answer
        """
        if not self.reasoner_v2:
            return False, None
        
        try:
            from nanobot.memory.types_v2 import TruthValue, RelationType
            
            # Parse the query to determine type and extract entities
            query_lower = user_message.lower()
            
            # Check for superlative queries
            if "tallest" in query_lower:
                result = self.reasoner_v2.query_superlative("tallest")
                if result.value == TruthValue.TRUE:
                    hypothesis = Hypothesis(
                        intent="superlative_tallest",
                        confidence=0.95,
                        reasoning=result.message
                    )
                    state = SuperpositionalState(
                        hypotheses=[hypothesis],
                        entropy=0.1,
                        strategic_direction=f"Deterministic v2 answer: {result.message}"
                    )
                    logger.info(f"V2 cache hit for superlative query: {result.message}")
                    return True, state
                elif result.value == TruthValue.UNKNOWN:
                    # Log UNKNOWN outcome
                    logger.info(f"V2 superlative query returned UNKNOWN: {result.message}")
                    return False, None
            
            elif "shortest" in query_lower:
                result = self.reasoner_v2.query_superlative("shortest")
                if result.value == TruthValue.TRUE:
                    hypothesis = Hypothesis(
                        intent="superlative_shortest",
                        confidence=0.95,
                        reasoning=result.message
                    )
                    state = SuperpositionalState(
                        hypotheses=[hypothesis],
                        entropy=0.1,
                        strategic_direction=f"Deterministic v2 answer: {result.message}"
                    )
                    logger.info(f"V2 cache hit for superlative query: {result.message}")
                    return True, state
                elif result.value == TruthValue.UNKNOWN:
                    logger.info(f"V2 superlative query returned UNKNOWN: {result.message}")
                    return False, None
            
            # Check for pairwise comparison queries
            # Extract entity names from the query
            entities = list(self.cache_v2.entities) if self.cache_v2 else []
            mentioned_entities = [e for e in entities if e.lower() in query_lower]
            
            if len(mentioned_entities) >= 2:
                # Determine relation type from query
                relation_type = RelationType.TALLER_THAN
                if "shorter" in query_lower:
                    relation_type = RelationType.SHORTER_THAN
                
                a, b = mentioned_entities[0], mentioned_entities[1]
                result = self.reasoner_v2.query_pairwise(a, b, relation_type)
                
                if result.value in [TruthValue.TRUE, TruthValue.FALSE]:
                    hypothesis = Hypothesis(
                        intent=f"pairwise_{relation_type.value.lower()}",
                        confidence=0.95,
                        reasoning=result.message
                    )
                    state = SuperpositionalState(
                        hypotheses=[hypothesis],
                        entropy=0.1,
                        strategic_direction=f"Deterministic v2 answer: {result.message}"
                    )
                    logger.info(f"V2 cache hit for pairwise query: {result.message}")
                    return True, state
                elif result.value == TruthValue.UNKNOWN:
                    logger.info(f"V2 pairwise query returned UNKNOWN: {result.message} (source: {result.source})")
                    return False, None
            
            # Query not supported by v2 or no answer
            logger.debug("V2 cache miss: query not supported or no information")
            return False, None
            
        except Exception as e:
            logger.warning(f"Error checking v2 memory cache: {e}")
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
        from nanobot.runtime.state import state
        
        # Try memory cache first
        can_answer, state_result = memory_reasoner.check_memory_first(user_message)

        if can_answer and state_result:
            return state_result
        
        # Check if LLM fallback is allowed
        llm_enabled = state.llm_enabled
        enable_llm_fallback = memory_config.get("enable_llm_fallback", True) if memory_config else True
        
        # CLI flag overrides config
        if not llm_enabled:
            logger.warning(
                f"LLM disabled globally - returning UNKNOWN for query: {user_message[:50]} ..."
            )
            from nanobot.agent.memory_types import Hypothesis, SuperpositionalState
            # Return deterministic UNKNOWN response
            return SuperpositionalState(
                hypotheses=[
                    Hypothesis(
                        intent="UNKNOWN - LLM disabled, memory has no answer",
                        confidence=1.0,
                        reasoning="LLM calls are disabled and memory cannot answer this query"
                    )
                ],
                entropy=0.0  # Deterministic UNKNOWN
            )
        
        # Config can also disable LLM fallback
        if not enable_llm_fallback:
            logger.warning(
                f"LLM fallback disabled by config - returning UNKNOWN for query: {user_message[:50]} ..."
            )
            from nanobot.agent.memory_types import Hypothesis, SuperpositionalState
            return SuperpositionalState(
                hypotheses=[
                    Hypothesis(
                        intent="UNKNOWN - LLM fallback disabled, memory has no answer",
                        confidence=1.0,
                        reasoning="LLM fallback is disabled by configuration and memory cannot answer this query"
                    )
                ],
                entropy=0.0
            )

        # Fall back to original LLM-based reasoning
        logger.debug(f"Falling back to LLM for query: {user_message[:50]} ...")
        return await original_reason(user_message, context_summary)

    # Replace reason method
    reasoner.reason = memory_aware_reason

    return reasoner
