"""Dual-layer reasoning orchestrator."""

import time
from dataclasses import dataclass, field
from typing import Any

from nanobot.agent.latent import LatentReasoner
from nanobot.agent.light_reasoner import LightReasonerEngine, LightReasonerResult
from nanobot.agent.memory_types import SuperpositionalState
from nanobot.providers.base import LLMProvider
from nanobot.runtime.chi_tracker import ChiTracker, ReasoningMetrics
from nanobot.runtime.state import state


@dataclass
class DualReasoningResult:
    final_state: SuperpositionalState
    system_used: str
    system1_result: LightReasonerResult | None
    system2_state: SuperpositionalState | None
    total_chi_cost: float
    escalated: bool
    reasoning_trace: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


class DualReasoningOrchestrator:
    """Coordinates fast System 1 and deep System 2 reasoning."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        chi_tracker: ChiTracker,
        memory_config: dict[str, Any] | None = None,
    ):
        self.light_reasoner = LightReasonerEngine(provider=provider, model=model)
        self.latent_reasoner = LatentReasoner(
            provider=provider,
            model=model,
            timeout_seconds=int((memory_config or {}).get("latent_timeout_seconds", 10)),
            memory_config=memory_config,
        )
        self.chi_tracker = chi_tracker

    async def reason(self, user_message: str, context_summary: str) -> DualReasoningResult:
        started = time.perf_counter()
        reasoning_trace: list[str] = []
        system1_result: LightReasonerResult | None = None
        system2_state: SuperpositionalState | None = None
        escalated = False
        system_used = "system2"

        budget = self.chi_tracker.get_budget_status()
        use_system1 = (
            state.dual_layer_enabled
            and state.light_reasoner_enabled
            and budget["system1_remaining"] > 0
        )
        use_system2 = state.latent_reasoning_enabled and budget["system2_remaining"] > 0

        if use_system1:
            reasoning_trace.append("→ System 1 (LightReasoner)")
            system1_result = await self.light_reasoner.reason(user_message, context_summary)
            reasoning_trace.extend(system1_result.reasoning_trace)
            if not system1_result.requires_escalation:
                system_used = "system1"
                final_state = SuperpositionalState(
                    hypotheses=[system1_result.hypothesis],
                    entropy=1.0 - system1_result.confidence,
                    strategic_direction=system1_result.hypothesis.reasoning,
                )
            else:
                escalated = True
        else:
            escalated = True

        if escalated and use_system2:
            reasoning_trace.append("→ System 2 (LatentReasoner + ALS)")
            system2_state = await self.latent_reasoner.reason(user_message, context_summary)
            final_state = system2_state
            system_used = "hybrid" if system1_result else "system2"

        if escalated and not use_system2:
            fallback_hypothesis = system1_result.hypothesis if system1_result else None
            final_state = SuperpositionalState(
                hypotheses=[fallback_hypothesis] if fallback_hypothesis else [],
                entropy=1.0 if fallback_hypothesis is None else 1.0 - fallback_hypothesis.confidence,
                strategic_direction="Proceed with standard processing.",
            )
            system_used = "system1" if fallback_hypothesis else "system2"

        system2_chi = (
            self.latent_reasoner.estimate_chi_cost(
                hypothesis_count=len(system2_state.hypotheses),
                entropy=system2_state.entropy,
            )
            if system2_state
            else 0.0
        )
        total_chi = (system1_result.chi_cost if system1_result else 0.0) + system2_chi
        latency_ms = (time.perf_counter() - started) * 1000
        if state.chi_tracking_enabled:
            self.chi_tracker.record_reasoning(
                ReasoningMetrics(
                    system_used=system_used,
                    chi_cost=total_chi,
                    latency_ms=latency_ms,
                    confidence=system1_result.confidence if system1_result else 0.0,
                    escalated=escalated,
                    hypothesis_count=len(final_state.hypotheses),
                    entropy=final_state.entropy,
                    pattern_cache_hit=bool(system1_result.pattern_hit) if system1_result else False,
                )
            )

        return DualReasoningResult(
            final_state=final_state,
            system_used=system_used,
            system1_result=system1_result,
            system2_state=system2_state,
            total_chi_cost=total_chi,
            escalated=escalated,
            reasoning_trace=reasoning_trace,
            latency_ms=latency_ms,
        )
