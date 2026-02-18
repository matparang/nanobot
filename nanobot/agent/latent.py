"""Async latent reasoning step for ambiguity handling."""

import asyncio
import json
import math
from json import JSONDecodeError
from typing import Any

from loguru import logger
from pydantic import ValidationError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from nanobot.agent.memory_types import Hypothesis, SuperpositionalState
from nanobot.middleware.circuit_breaker import CircuitBreaker, CircuitOpenError
from nanobot.providers.base import LLMProvider
from nanobot.runtime.state import state


# Track whether we've logged the disabled message this session
_latent_disabled_logged = False


def latent_enabled() -> bool:
    """
    Centralized gate function for latent reasoning.
    
    Returns False if:
    - Baseline mode is active (highest priority)
    - state.latent_reasoning_enabled is False
    
    This is the single source of truth for whether latent reasoning
    should execute. All latent reasoning entry points must check this gate.
    
    Returns:
        True if latent reasoning should execute, False otherwise.
    """
    global _latent_disabled_logged
    
    # Baseline mode always disables latent reasoning
    if state.baseline_active:
        if not _latent_disabled_logged:
            logger.info("Latent reasoning disabled: baseline mode active")
            _latent_disabled_logged = True
        return False
    
    # Check the runtime toggle
    if not state.latent_reasoning_enabled:
        if not _latent_disabled_logged:
            logger.info("Latent reasoning disabled for this session")
            _latent_disabled_logged = True
        return False
    
    return True


class LatentReasoner:
    """Performs a short hidden reasoning pass before tool execution."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        timeout_seconds: int = 10,
        memory_config: dict[str, Any] | None = None,
    ):
        self.provider = provider
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.memory_config = memory_config or {}
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=int(self.memory_config.get("latent_circuit_failure_threshold", 3)),
            recovery_timeout=int(self.memory_config.get("latent_circuit_recovery_timeout", 60)),
        )

    async def _call_llm_with_backoff(
        self, system_prompt: str, prompt: str, temperature: float = 0.1
    ) -> str:
        max_attempts = max(int(self.memory_config.get("latent_retry_attempts", 3)), 1)
        retry_min_wait = max(float(self.memory_config.get("latent_retry_min_wait", 1.0)), 0.0)
        retry_max_wait = max(float(self.memory_config.get("latent_retry_max_wait", 5.0)), 0.0)
        retry_multiplier = max(float(self.memory_config.get("latent_retry_multiplier", 1.0)), 0.0)

        retrying = AsyncRetrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(
                multiplier=retry_multiplier, min=retry_min_wait, max=retry_max_wait
            ),
            retry=retry_if_exception_type((asyncio.TimeoutError, ConnectionError, OSError)),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                response = await self.provider.chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.model,
                    temperature=temperature,
                )
                return (response.content or "").strip()
        return ""

    async def reason(self, user_message: str, context_summary: str) -> SuperpositionalState:
        """
        Perform latent reasoning to generate hypotheses about user intent.
        
        Returns a fallback SuperpositionalState if latent reasoning is disabled
        or if an error occurs.
        """
        # Check the centralized gate first
        if not latent_enabled():
            return self._fallback_state()
        
        fallback_state = self._fallback_state()
        try:
            return await self.circuit_breaker.call(
                self._reason_internal,
                user_message=user_message,
                context_summary=context_summary,
            )
        except CircuitOpenError:
            return SuperpositionalState(
                hypotheses=[],
                entropy=0.0,
                strategic_direction="Circuit open - standard processing",
            )
        except Exception:
            return fallback_state

    async def _reason_internal(self, user_message: str, context_summary: str) -> SuperpositionalState:
        fallback_state = self._fallback_state()

        system_prompt = (
            "You are the subconscious reasoning engine of an AI agent. "
            "Analyze the user input and context. Generate 2-3 intent hypotheses, "
            "their confidence, entropy of ambiguity, and a strategic direction. "
            "Return only JSON for SuperpositionalState."
        )
        prompt = (
            f"Context: {context_summary}\n"
            f"User Input: {user_message}\n\n"
            "Tasks:\n"
            "1. Identify 2-3 potential distinct intents.\n"
            "2. Assign confidence to each.\n"
            "3. If confidence is split evenly, set high entropy near 1.0.\n"
            "4. If one hypothesis dominates, set low entropy near 0.0.\n"
        )

        try:
            initial_hypotheses = await asyncio.wait_for(
                self._generate_initial_hypotheses(system_prompt, prompt),
                timeout=self.timeout_seconds,
            )
            if not initial_hypotheses:
                return fallback_state

            max_depth = max(int(self.memory_config.get("latent_max_depth", 1)), 1)
            beam_width = max(int(self.memory_config.get("beam_width", 3)), 1)
            clarify_threshold = float(self.memory_config.get("clarify_entropy_threshold", 0.8))
            hypotheses = sorted(initial_hypotheses, key=self._score_hypothesis, reverse=True)[
                :beam_width
            ]
            entropy = self._calculate_entropy(hypotheses)
            for current_depth in range(max_depth):
                logger.info(
                    "latent.depth={} latent.entropy={:.3f}", current_depth, entropy
                )
                if entropy < clarify_threshold:
                    break
                if int(self.memory_config.get("monte_carlo_samples", 0)) > 0:
                    expanded = await self._monte_carlo_expand(
                        hypotheses, user_message, context_summary
                    )
                    hypotheses.extend(expanded)
                before_prune = len(hypotheses)
                hypotheses = sorted(hypotheses, key=self._score_hypothesis, reverse=True)[
                    :beam_width
                ]
                logger.info(
                    "latent.beam.pruned={} latent.beam.width={}",
                    max(before_prune - len(hypotheses), 0),
                    beam_width,
                )
                entropy = self._calculate_entropy(hypotheses)

            best_hypothesis = max(hypotheses, key=self._score_hypothesis)
            return SuperpositionalState(
                hypotheses=hypotheses,
                entropy=entropy,
                strategic_direction=best_hypothesis.reasoning
                or "Proceed with standard processing.",
            )
        except (asyncio.TimeoutError, JSONDecodeError, ValidationError) as exc:
            logger.debug(f"Latent reasoning error, propagating exception: {exc}")
            raise
        except Exception as exc:
            logger.warning(f"Unexpected latent reasoning error: {exc}")
            raise

    async def _generate_initial_hypotheses(
        self, system_prompt: str, prompt: str
    ) -> list[Hypothesis]:
        payload = await self._call_llm_with_backoff(system_prompt, prompt)
        return self._parse_hypotheses(payload)

    @staticmethod
    def _fallback_state() -> SuperpositionalState:
        return SuperpositionalState(
            hypotheses=[],
            entropy=0.0,
            strategic_direction="Proceed with standard processing due to reasoning timeout/error.",
        )

    def _parse_hypotheses(self, payload: str) -> list[Hypothesis]:
        """
        Parse hypotheses from LLM response with robust error handling.
        
        Handles:
        - Empty responses
        - Malformed JSON
        - Missing fields
        - Unexpected data types
        """
        if not payload or not payload.strip():
            logger.warning("Latent reasoning: empty response from model")
            return []
        
        # Strip markdown code blocks
        if payload.startswith("```"):
            payload = payload.removeprefix("```json").removeprefix("```").strip()
            if payload.endswith("```"):
                payload = payload[:-3].strip()
        
        try:
            parsed = json.loads(payload)
        except JSONDecodeError as e:
            logger.warning(
                f"Latent reasoning: JSON parse error - {e}. "
                f"Response (truncated): {payload[:200]}..."
            )
            return []
        
        # Handle different response structures
        if not isinstance(parsed, dict):
            logger.warning(
                f"Latent reasoning: expected dict, got {type(parsed).__name__}. "
                f"Response (truncated): {str(parsed)[:200]}..."
            )
            return []
        
        if "hypotheses" in parsed:
            hypotheses_data = parsed["hypotheses"]
        elif all(k in parsed for k in ("intent", "confidence")):
            hypotheses_data = [parsed]
        else:
            logger.warning(
                f"Latent reasoning: no hypotheses found in response. "
                f"Keys: {list(parsed.keys())}"
            )
            return []
        
        if not isinstance(hypotheses_data, list):
            logger.warning(
                f"Latent reasoning: hypotheses should be list, got {type(hypotheses_data).__name__}"
            )
            return []
        
        hypotheses = []
        for item in hypotheses_data:
            if not isinstance(item, dict):
                continue
            
            try:
                confidence = float(item.get("confidence", 0.0))
                hypotheses.append(
                    Hypothesis(
                        intent=str(item.get("intent", "unknown")),
                        confidence=max(confidence, 0.0),
                        reasoning=str(item.get("reasoning", "latent inference")),
                    )
                )
            except (ValueError, TypeError, ValidationError) as e:
                logger.debug(f"Latent reasoning: skipping invalid hypothesis - {e}")
                continue
        
        if not hypotheses:
            logger.warning("Latent reasoning: no valid hypotheses parsed from response")
        
        return hypotheses

    def _score_hypothesis(self, hypothesis: Hypothesis) -> float:
        """Centralized hypothesis scoring hook for consistent beam pruning."""
        return max(hypothesis.confidence, 0.0)

    def _calculate_entropy(self, hypotheses: list[Hypothesis]) -> float:
        total_confidence = sum(max(h.confidence, 0.0) for h in hypotheses) or 1.0
        probabilities = [max(h.confidence, 0.0) / total_confidence for h in hypotheses]
        return -sum(p * math.log2(p) for p in probabilities if p > 0)

    @staticmethod
    def estimate_chi_cost(hypothesis_count: int, entropy: float) -> float:
        """Estimate chi spend for deep reasoning based on explored hypotheses and ambiguity."""
        return max(1.0, 0.5 + (0.1 * max(hypothesis_count, 1)) + max(entropy, 0.0))

    async def _monte_carlo_expand(
        self,
        hypotheses: list[Hypothesis],
        user_message: str,
        context_summary: str,
    ) -> list[Hypothesis]:
        samples = max(int(self.memory_config.get("monte_carlo_samples", 0)), 0)
        if samples <= 0 or not hypotheses:
            return []

        seed_hypotheses = json.dumps(
            [
                {"intent": h.intent, "confidence": h.confidence, "reasoning": h.reasoning}
                for h in hypotheses
            ],
            ensure_ascii=False,
        )
        system_prompt = (
            "You are a latent reasoning sampler. Expand possible user intents from seed hypotheses. "
            "Return only JSON as an object with a hypotheses array."
        )
        prompt = (
            f"Context: {context_summary}\n"
            f"User Input: {user_message}\n"
            f"Seed Hypotheses: {seed_hypotheses}\n\n"
            "Generate 1-2 additional plausible hypotheses with confidence and reasoning."
        )

        expanded: list[Hypothesis] = []
        logger.info("latent.montecarlo.samples={}", samples)
        for _ in range(samples):
            payload = await self._call_llm_with_backoff(system_prompt, prompt, temperature=0.6)
            expanded.extend(self._parse_hypotheses(payload))
        return expanded
