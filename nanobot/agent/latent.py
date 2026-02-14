"""Latent reasoning module for QL-Bot."""

import json

from loguru import logger
from nanobot.agent.memory_types import LatentGraph
from nanobot.providers.base import LLMProvider


class LatentReasoner:
    """Runs a lightweight latent reasoning pass before tool execution."""

    def __init__(self, llm_provider: LLMProvider, model: str, temperature: float = 0.7):
        self.llm = llm_provider
        self.model = model
        self.temperature = temperature

    async def reason(self, context_str: str) -> tuple[LatentGraph, str]:
        """
        Returns the latent graph and the collapsed strategy text.
        """
        system_prompt = (
            "You are QL-Bot. You exist in a probabilistic superposition of strategies. "
            "Return ONLY valid JSON for a LatentGraph with keys: "
            "nodes, edges, probabilities, selected_hypothesis_index."
        )
        user_prompt = (
            f"Context:\n{context_str}\n\n"
            "Generate 3 competing hypotheses in graph form and pick one "
            "selected_hypothesis_index."
        )

        try:
            response = await self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
                temperature=self.temperature,
            )
            raw = (response.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            graph_data = json.loads(raw)
            graph = LatentGraph.model_validate(graph_data)
            collapsed_strategy = f"Selected Hypothesis {graph.selected_hypothesis_index}"
            return graph, collapsed_strategy
        except Exception as e:
            logger.warning(f"Latent reasoning fallback triggered: {e}")
            return LatentGraph(), "Linear Fallback"
