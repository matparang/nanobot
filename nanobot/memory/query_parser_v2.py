"""Deterministic natural-language query parser for relational reasoning.

Pure regex-based — zero LLM usage. Parses NL queries into structured
ParsedQuery objects that can be routed to the appropriate reasoning engine.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from nanobot.memory.config_v2 import LOGGER_NAME
from nanobot.memory.types_v2 import RelationType

logger = logging.getLogger(LOGGER_NAME)


class QueryType(Enum):
    """Supported query categories."""
    PAIRWISE = "PAIRWISE"
    SUPERLATIVE = "SUPERLATIVE"
    ALL_SATISFYING = "ALL_SATISFYING"
    COUNT = "COUNT"
    RANK = "RANK"
    REACHABLE = "REACHABLE"
    ADD_FACT = "ADD_FACT"
    UNKNOWN = "UNKNOWN"


@dataclass
class ParsedQuery:
    """Structured representation of a parsed NL query.

    Attributes:
        query_type: Detected QueryType
        entities: Entity names extracted from the query
        relation_type: Detected RelationType (if any)
        superlative_kind: Superlative keyword (e.g., "tallest") if applicable
        raw_text: Original query string
    """
    query_type: QueryType
    entities: list[str] = field(default_factory=list)
    relation_type: Optional[RelationType] = None
    superlative_kind: Optional[str] = None
    raw_text: str = ""


# Mapping from NL phrases to RelationType
RELATION_KEYWORDS: dict[str, RelationType] = {
    "taller than": RelationType.TALLER_THAN,
    "shorter than": RelationType.SHORTER_THAN,
    "faster than": RelationType.FASTER_THAN,
    "slower than": RelationType.SLOWER_THAN,
    "greater than": RelationType.GREATER_THAN,
    "less than": RelationType.LESS_THAN,
    "depends on": RelationType.DEPENDS_ON,
    "dependency of": RelationType.DEPENDENCY_OF,
    "impacts": RelationType.IMPACTS,
    "impacted by": RelationType.IMPACTED_BY,
    "contains": RelationType.CONTAINS,
    "contained in": RelationType.CONTAINED_IN,
    "supplies": RelationType.SUPPLIES,
    "supplied by": RelationType.SUPPLIED_BY,
    "related to": RelationType.RELATED_TO,
}

# Mapping from superlative keyword to (kind, RelationType)
SUPERLATIVE_KEYWORDS: dict[str, tuple[str, RelationType]] = {
    "tallest": ("tallest", RelationType.TALLER_THAN),
    "shortest": ("shortest", RelationType.SHORTER_THAN),
    "fastest": ("fastest", RelationType.FASTER_THAN),
    "slowest": ("slowest", RelationType.SLOWER_THAN),
    "greatest": ("greatest", RelationType.GREATER_THAN),
    "least": ("least", RelationType.LESS_THAN),
}

# Build a combined pattern for relation keywords (longest match first)
_RELATION_PATTERN = re.compile(
    r"(" + "|".join(re.escape(k) for k in sorted(RELATION_KEYWORDS, key=len, reverse=True)) + r")",
    re.IGNORECASE,
)


def _extract_relation(text: str) -> Optional[RelationType]:
    """Return the first RelationType matched in *text*, or None."""
    m = _RELATION_PATTERN.search(text)
    if m:
        return RELATION_KEYWORDS[m.group(1).lower()]
    return None


def _extract_entities_around(text: str, relation_phrase: str) -> list[str]:
    """Extract entity tokens on either side of *relation_phrase* in *text*."""
    parts = re.split(re.escape(relation_phrase), text, maxsplit=1, flags=re.IGNORECASE)
    entities: list[str] = []
    if len(parts) == 2:
        # Left side: last token (strip filler words)
        left = re.sub(
            r"(?i)\b(is|are|was|does|do|who|what|the|a|an)\b", " ", parts[0]
        ).strip()
        right = re.sub(
            r"(?i)\b(is|are|was|does|do|who|what|the|a|an)\b", " ", parts[1]
        ).strip()
        # Remove leading "Is " / question marks
        left = re.sub(r"[?!.]+$", "", left).strip()
        right = re.sub(r"[?!.]+$", "", right).strip()
        # Take last word of left, first word of right
        left_tokens = left.split()
        right_tokens = right.split()
        if left_tokens:
            entities.append(left_tokens[-1])
        if right_tokens:
            entities.append(right_tokens[0])
    return entities


class DeterministicQueryParser:
    """Pure regex-based NL → ParsedQuery parser.

    Tries patterns in priority order; falls back to keyword scanning.
    """

    # ------------------------------------------------------------------
    # Compiled patterns (class-level)
    # ------------------------------------------------------------------

    # "How many X does/does A impact?" or "How many entities impact B?"
    _COUNT_PATTERN = re.compile(
        r"how\s+many",
        re.IGNORECASE,
    )

    # "Who is the tallest?" / "Who is tallest?" / "What is the tallest?"
    _SUPERLATIVE_PATTERN = re.compile(
        r"(?:who|what)\s+is\s+(?:the\s+)?(" +
        "|".join(re.escape(k) for k in SUPERLATIVE_KEYWORDS) +
        r")\b",
        re.IGNORECASE,
    )

    # "What does X impact?" / "What does X depend on?"
    _REACHABLE_PATTERN = re.compile(
        r"(?:what|who)\s+(?:does|do)\s+(\S+)\s+(" +
        "|".join(re.escape(k) for k in sorted(RELATION_KEYWORDS, key=len, reverse=True)) +
        r")\b",
        re.IGNORECASE,
    )

    # "Who is taller than Carol?" / "Who is faster than X?"
    _ALL_SATISFYING_PATTERN = re.compile(
        r"(?:who|what)\s+(?:is|are)\s+(" +
        "|".join(re.escape(k) for k in sorted(RELATION_KEYWORDS, key=len, reverse=True)) +
        r")\s+(\S+)\b",
        re.IGNORECASE,
    )

    # "Is Alice taller than Bob?" (pairwise interrogative)
    _PAIRWISE_PATTERN = re.compile(
        r"^(?:is|are)\s+(\S+)\s+(" +
        "|".join(re.escape(k) for k in sorted(RELATION_KEYWORDS, key=len, reverse=True)) +
        r")\s+(\S+)\b",
        re.IGNORECASE,
    )

    # "Alice is taller than Bob" (declarative / add-fact)
    _ADD_FACT_PATTERN = re.compile(
        r"^(\S+)\s+(?:is\s+)?(" +
        "|".join(re.escape(k) for k in sorted(RELATION_KEYWORDS, key=len, reverse=True)) +
        r")\s+(\S+)\b",
        re.IGNORECASE,
    )

    # "Rank all by height" / "Rank all taller_than" / just "Rank all"
    _RANK_PATTERN = re.compile(
        r"(?:rank|list)\s+all",
        re.IGNORECASE,
    )

    def parse(self, query: str) -> ParsedQuery:
        """Parse *query* into a ParsedQuery.

        Tries patterns in priority order:
        1. Count
        2. Superlative
        3. Reachable
        4. All-satisfying
        5. Pairwise (interrogative)
        6. Add-fact (declarative)
        7. Rank
        8. Keyword scan fallback

        Args:
            query: Natural language query string

        Returns:
            ParsedQuery with detected type and extracted fields
        """
        text = query.strip()

        # 1. Count
        if self._COUNT_PATTERN.search(text):
            relation_type = _extract_relation(text)
            entities = self._scan_entities(text, relation_type)
            return ParsedQuery(
                query_type=QueryType.COUNT,
                entities=entities,
                relation_type=relation_type,
                raw_text=query,
            )

        # 2. Superlative
        m = self._SUPERLATIVE_PATTERN.search(text)
        if m:
            kind = m.group(1).lower()
            _, relation_type = SUPERLATIVE_KEYWORDS[kind]
            return ParsedQuery(
                query_type=QueryType.SUPERLATIVE,
                relation_type=relation_type,
                superlative_kind=kind,
                raw_text=query,
            )

        # 3. Reachable — "What does X impact?"
        m = self._REACHABLE_PATTERN.search(text)
        if m:
            entity = m.group(1)
            rel_phrase = m.group(2).lower()
            relation_type = RELATION_KEYWORDS.get(rel_phrase)
            return ParsedQuery(
                query_type=QueryType.REACHABLE,
                entities=[entity],
                relation_type=relation_type,
                raw_text=query,
            )

        # 4. All-satisfying — "Who is taller than Carol?"
        m = self._ALL_SATISFYING_PATTERN.search(text)
        if m:
            rel_phrase = m.group(1).lower()
            target = m.group(2).rstrip("?!.")
            relation_type = RELATION_KEYWORDS.get(rel_phrase)
            return ParsedQuery(
                query_type=QueryType.ALL_SATISFYING,
                entities=[target],
                relation_type=relation_type,
                raw_text=query,
            )

        # 5. Pairwise interrogative — "Is Alice taller than Bob?"
        m = self._PAIRWISE_PATTERN.search(text)
        if m:
            entity_a = m.group(1)
            rel_phrase = m.group(2).lower()
            entity_b = m.group(3).rstrip("?!.")
            relation_type = RELATION_KEYWORDS.get(rel_phrase)
            return ParsedQuery(
                query_type=QueryType.PAIRWISE,
                entities=[entity_a, entity_b],
                relation_type=relation_type,
                raw_text=query,
            )

        # 6. Rank
        if self._RANK_PATTERN.search(text):
            relation_type = _extract_relation(text)
            return ParsedQuery(
                query_type=QueryType.RANK,
                relation_type=relation_type,
                raw_text=query,
            )

        # 7. Add-fact declarative — "Alice is taller than Bob"
        m = self._ADD_FACT_PATTERN.search(text)
        if m:
            entity_a = m.group(1)
            rel_phrase = m.group(2).lower()
            entity_b = m.group(3).rstrip("?!.")
            relation_type = RELATION_KEYWORDS.get(rel_phrase)
            return ParsedQuery(
                query_type=QueryType.ADD_FACT,
                entities=[entity_a, entity_b],
                relation_type=relation_type,
                raw_text=query,
            )

        # 8. Keyword scan fallback
        relation_type = _extract_relation(text)
        if relation_type:
            entities = self._scan_entities(text, relation_type)
            if len(entities) == 2:
                return ParsedQuery(
                    query_type=QueryType.PAIRWISE,
                    entities=entities,
                    relation_type=relation_type,
                    raw_text=query,
                )
            if len(entities) == 1:
                return ParsedQuery(
                    query_type=QueryType.REACHABLE,
                    entities=entities,
                    relation_type=relation_type,
                    raw_text=query,
                )

        logger.debug(f"Could not parse query: '{query}'")
        return ParsedQuery(query_type=QueryType.UNKNOWN, raw_text=query)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scan_entities(
        self,
        text: str,
        relation_type: Optional[RelationType],
    ) -> list[str]:
        """Scan text for entity tokens around a relation keyword."""
        if relation_type is None:
            return []
        # Find the phrase for this relation type
        for phrase, rt in RELATION_KEYWORDS.items():
            if rt == relation_type:
                entities = _extract_entities_around(text, phrase)
                if entities:
                    return entities
        return []
