"""Fast System 1 reasoning engine with pattern caching."""

import asyncio
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from hashlib import sha256
from json import JSONDecodeError
from typing import Any

from nanobot.agent.memory_types import Hypothesis
from nanobot.providers.base import LLMProvider
from nanobot.runtime.state import state


@dataclass
class LightReasonerConfig:
    confidence_threshold: float = 0.75
    max_response_time_ms: int = 500


@dataclass
class LightReasonerResult:
    hypothesis: Hypothesis
    confidence: float
    chi_cost: float
    latency_ms: float
    pattern_hit: bool
    requires_escalation: bool
    reasoning_trace: list[str] = field(default_factory=list)


class PatternCache:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[str, Hypothesis] = OrderedDict()

    @staticmethod
    def compute_key(user_message: str, context_summary: str) -> str:
        raw = f"{user_message.strip().lower()}::{context_summary[:250].strip().lower()}"
        return sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Hypothesis | None:
        item = self._cache.get(key)
        if item is not None:
            self._cache.move_to_end(key)
        return item

    def put(self, key: str, hypothesis: Hypothesis) -> None:
        self._cache[key] = hypothesis
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)


class LightReasonerEngine:
    """System 1: Fast, heuristic reasoning with cache and confidence gating."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        config: LightReasonerConfig | None = None,
    ):
        self.provider = provider
        self.model = model
        self.config = config or LightReasonerConfig()
        self.pattern_cache = PatternCache(max_size=1000)

    async def reason(self, user_message: str, context_summary: str) -> LightReasonerResult:
        start = time.perf_counter()
        if not state.light_reasoner_enabled:
            return self._escalation_result("LightReasoner disabled", start)

        cache_key = self.pattern_cache.compute_key(user_message, context_summary)
        cached = self.pattern_cache.get(cache_key)
        if cached and cached.confidence >= self.config.confidence_threshold:
            latency_ms = (time.perf_counter() - start) * 1000
            return LightReasonerResult(
                hypothesis=cached,
                confidence=cached.confidence,
                chi_cost=0.01,
                latency_ms=latency_ms,
                pattern_hit=True,
                requires_escalation=False,
                reasoning_trace=["cache_hit"],
            )

        hypothesis = await self._quick_inference(user_message, context_summary)
        confidence = self._estimate_confidence(hypothesis, user_message)
        requires_escalation = confidence < self.config.confidence_threshold
        if not requires_escalation:
            self.pattern_cache.put(cache_key, hypothesis)

        latency_ms = (time.perf_counter() - start) * 1000
        return LightReasonerResult(
            hypothesis=hypothesis,
            confidence=confidence,
            chi_cost=0.1,
            latency_ms=latency_ms,
            pattern_hit=False,
            requires_escalation=requires_escalation,
            reasoning_trace=["quick_inference", f"confidence={confidence:.2f}"],
        )

    async def _quick_inference(self, user_message: str, context_summary: str) -> Hypothesis:
        try:
            response = await asyncio.wait_for(
                self.provider.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Fast intent classifier. Return JSON with keys: "
                                "intent, confidence, reasoning."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Context: {context_summary[:200]}\nQuery: {user_message[:300]}",
                        },
                    ],
                    model=self.model,
                    max_tokens=150,
                    temperature=0.1,
                ),
                timeout=max(self.config.max_response_time_ms, 100) / 1000,
            )
            return self._parse_hypothesis(response.content or "")
        except (asyncio.TimeoutError, JSONDecodeError, ValueError):
            return Hypothesis(
                intent="unknown",
                confidence=0.0,
                reasoning="light_reasoner_fallback",
            )

    @staticmethod
    def _parse_hypothesis(payload: str) -> Hypothesis:
        parsed: dict[str, Any] = json.loads(payload)
        confidence = float(parsed.get("confidence", 0.0))
        return Hypothesis(
            intent=str(parsed.get("intent", "unknown")),
            confidence=max(min(confidence, 1.0), 0.0),
            reasoning=str(parsed.get("reasoning", "fast heuristic reasoning")),
        )

    @staticmethod
    def _estimate_confidence(hypothesis: Hypothesis, user_message: str) -> float:
        base = max(min(hypothesis.confidence, 1.0), 0.0)
        if len(user_message.split()) <= 5:
            base = min(base + 0.05, 1.0)
        return base

    def _escalation_result(self, reason: str, start: float) -> LightReasonerResult:
        latency_ms = (time.perf_counter() - start) * 1000
        return LightReasonerResult(
            hypothesis=Hypothesis(intent="unknown", confidence=0.0, reasoning=reason),
            confidence=0.0,
            chi_cost=0.0,
            latency_ms=latency_ms,
            pattern_hit=False,
            requires_escalation=True,
            reasoning_trace=[reason],
        )
