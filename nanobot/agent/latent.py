"""Async latent reasoning step for ambiguity handling."""

import asyncio
import json
import random
import time
from collections import Counter
from json import JSONDecodeError

from loguru import logger
from pydantic import ValidationError

from nanobot.agent.memory_types import Hypothesis, SuperpositionalState
from nanobot.providers.base import LLMProvider


class LatentReasoner:
    """Performs a short hidden reasoning pass before tool execution."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        timeout_seconds: int = 10,
        entropy_threshold: float = 0.8,
        max_depth: int = 1,
        monte_carlo_samples: int = 1,
        monte_carlo_top_k: int = 3,
    ):
        self.provider = provider
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.entropy_threshold = entropy_threshold
        self.max_depth = max(1, max_depth)
        self.monte_carlo_samples = max(1, monte_carlo_samples)
        self.monte_carlo_top_k = max(1, monte_carlo_top_k)

    async def reason(self, user_message: str, context_summary: str) -> SuperpositionalState:
        fallback_state = SuperpositionalState(
            hypotheses=[],
            entropy=0.0,
            strategic_direction="Proceed with standard processing due to reasoning timeout/error.",
        )
        best_state = fallback_state
        for depth in range(1, self.max_depth + 1):
            logger.info("Latent reasoning depth {}/{}", depth, self.max_depth)
            state = await self._reason_once(
                user_message=user_message,
                context_summary=context_summary,
                depth=depth,
                fallback_state=fallback_state,
            )
            best_state = state
            logger.debug("Latent reasoning depth {} entropy={:.4f}", depth, state.entropy)
            if state.entropy <= self.entropy_threshold:
                logger.info(
                    "Latent reasoning stopped at depth {} due to entropy threshold {:.4f}",
                    depth,
                    self.entropy_threshold,
                )
                break
        return self._apply_monte_carlo(best_state)

    def _apply_monte_carlo(self, state: SuperpositionalState) -> SuperpositionalState:
        """Sample and reduce hypotheses based on confidence distribution."""
        hypotheses = state.hypotheses or []
        if not hypotheses:
            return state
        if self.monte_carlo_samples <= 1:
            return state

        weights = [max(h.confidence, 0.0) for h in hypotheses]
        if sum(weights) <= 0:
            weights = [1.0] * len(hypotheses)
        samples = random.choices(hypotheses, weights=weights, k=self.monte_carlo_samples)
        sampled_intents = Counter(h.intent for h in samples)
        retained_hypotheses: list[Hypothesis] = []
        for intent, count in sampled_intents.most_common(self.monte_carlo_top_k):
            match = next((h for h in hypotheses if h.intent == intent), None)
            if not match:
                continue
            retained_hypotheses.append(
                Hypothesis(
                    intent=match.intent,
                    confidence=count / self.monte_carlo_samples,
                    reasoning=match.reasoning,
                    required_tools=match.required_tools,
                )
            )
        if not retained_hypotheses:
            return state
        return SuperpositionalState(
            hypotheses=retained_hypotheses,
            entropy=state.entropy,
            strategic_direction=state.strategic_direction,
        )

    async def _reason_once(
        self,
        user_message: str,
        context_summary: str,
        depth: int,
        fallback_state: SuperpositionalState,
    ) -> SuperpositionalState:
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
            f"5. Current refinement depth: {depth}.\n"
        )

        try:
            start = time.monotonic()
            response = await asyncio.wait_for(
                self.provider.chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.model,
                    temperature=0.1,
                ),
                timeout=self.timeout_seconds,
            )
            elapsed = time.monotonic() - start
            if elapsed > self.timeout_seconds:
                logger.warning(
                    "Latent reasoning exceeded configured timeout: {:.2f}s > {}s",
                    elapsed,
                    self.timeout_seconds,
                )
            payload = (response.content or "").strip()
            if payload.startswith("```"):
                payload = payload.removeprefix("```json").removeprefix("```").strip()
                if payload.endswith("```"):
                    payload = payload[:-3].strip()
            return SuperpositionalState.model_validate(json.loads(payload))
        except (asyncio.TimeoutError, JSONDecodeError, ValidationError) as exc:
            logger.debug(f"Latent reasoning fallback triggered: {exc}")
            return fallback_state
        except Exception as exc:
            logger.warning(f"Unexpected latent reasoning error: {exc}")
            return fallback_state
