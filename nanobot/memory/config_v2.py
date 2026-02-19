"""Configuration constants for v2 relational reasoning.

This module centralizes default policies, thresholds, and messages for
deterministic memory-first reasoning.
"""

# Confidence threshold for relation extraction
CONF_THRESHOLD = 0.9

# Ranking policy: "tiers" for partial order levels
RANKING_POLICY = "tiers"

# Superlative policy: return UNKNOWN if not uniquely determined
SUPERLATIVE_POLICY = "unknown_if_not_unique"

# Standard messages for UNKNOWN results
UNKNOWN_PAIRWISE_MESSAGE = "UNKNOWN: No known relation"
UNKNOWN_SUPERLATIVE_MESSAGE = "UNKNOWN: Insufficient information"
UNKNOWN_INCONSISTENT_MESSAGE = "UNKNOWN: Inconsistent relations"
UNKNOWN_SELF_COMPARISON_MESSAGE = "UNKNOWN: Self-comparison not allowed"
UNKNOWN_ENTITY_MESSAGE = "UNKNOWN: Entity not in knowledge base"

# Memory-first enforcement: no LLM fallback
MEMORY_FIRST_ENFORCED = True

# Logging defaults
LOG_LEVEL = "INFO"
LOGGER_NAME = "nanobot.memory.v2"
