"""SangPenjaga v5 Flow/MoE metrics hooks."""

from nanobot.runtime.chi_tracker import ChiTracker


def compute_absorption_metrics(tracker: ChiTracker, success_score: float, chi_cost: float) -> dict[str, float]:
    return {
        "absorption": tracker.compute_absorption(success_score=success_score, chi_cost=chi_cost),
        "system1_weight": tracker.moe_weights.get("system1", 0.0),
        "system2_weight": tracker.moe_weights.get("system2", 0.0),
    }


def update_routing_weights(tracker: ChiTracker) -> dict[str, float]:
    return tracker.update_moe_weights()
