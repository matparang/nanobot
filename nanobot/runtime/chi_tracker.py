"""Chi cost tracking for dual-layer reasoning."""

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class ReasoningMetrics:
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    system_used: str = "system2"
    chi_cost: float = 0.0
    latency_ms: float = 0.0
    confidence: float = 0.0
    success_score: float = 0.0
    escalated: bool = False
    hypothesis_count: int = 0
    entropy: float = 0.0
    pattern_cache_hit: bool = False


@dataclass
class ChiAllocation:
    system1_budget: float = 100.0
    system2_budget: float = 500.0
    current_system1: float = 100.0
    current_system2: float = 500.0
    total_spent: float = 0.0


class ChiTracker:
    def __init__(self):
        self.allocation = ChiAllocation()
        self.session_metrics: list[ReasoningMetrics] = []
        self.moe_weights = {"system1": 0.3, "system2": 0.7}

    def get_budget_status(self) -> dict[str, float]:
        return {
            "system1_remaining": max(self.allocation.current_system1, 0.0),
            "system2_remaining": max(self.allocation.current_system2, 0.0),
            "total_spent": self.allocation.total_spent,
        }

    def record_reasoning(self, metrics: ReasoningMetrics) -> None:
        if metrics.system_used == "system1":
            self.allocation.current_system1 -= metrics.chi_cost
        else:
            self.allocation.current_system2 -= metrics.chi_cost
        self.allocation.total_spent += metrics.chi_cost
        self.session_metrics.append(metrics)

    def compute_absorption(self, success_score: float, chi_cost: float) -> float:
        weight = self.moe_weights.get("system1", 0.5)
        return (success_score * weight) / max(chi_cost, 0.01)

    def _avg_efficiency(self, metrics: list[ReasoningMetrics]) -> float:
        if not metrics:
            return 0.0
        return sum((m.success_score or m.confidence) / max(m.chi_cost, 0.01) for m in metrics) / len(metrics)

    def update_moe_weights(self) -> dict[str, float]:
        s1_eff = self._avg_efficiency([m for m in self.session_metrics if m.system_used == "system1"])
        s2_eff = self._avg_efficiency(
            [m for m in self.session_metrics if m.system_used in {"system2", "hybrid"}]
        )
        total = s1_eff + s2_eff
        if total > 0:
            self.moe_weights = {"system1": s1_eff / total, "system2": s2_eff / total}
        return self.moe_weights
