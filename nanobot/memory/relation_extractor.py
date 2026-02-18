"""Deterministic regex-based relation extraction from natural language text.

This module provides the RelationExtractionEngine class that extracts entity
relationships from text without LLM invocation. It serves as the Root memory
ingestion layer in the Soul Kernel architecture.
"""

import re
from typing import Any

from loguru import logger


class RelationExtractionEngine:
    """Extract entity relationships from natural language using regex patterns.

    This engine provides deterministic, sub-millisecond extraction of comparative
    relationships (taller/shorter, older/younger, etc.) from text. It automatically
    generates inverse relations for complete graph coverage.

    Attributes:
        patterns: Compiled regex patterns for relation extraction
    """

    def __init__(self):
        """Initialize relation extraction engine with compiled regex patterns."""
        # Define relation patterns with their types
        # Format: (pattern, relation_type, inverse_type)
        # Pattern captures entities with optional articles like "the", "a", "an"
        relation_defs = [
            # Height relations
            (r"(?:the\s+)?(\w+)\s+(?:is\s+)?taller\s+than\s+(?:the\s+)?(\w+)", "taller_than", "shorter_than"),
            (r"(?:the\s+)?(\w+)\s+(?:is\s+)?shorter\s+than\s+(?:the\s+)?(\w+)", "shorter_than", "taller_than"),
            
            # Age relations
            (r"(?:the\s+)?(\w+)\s+(?:is\s+)?older\s+than\s+(?:the\s+)?(\w+)", "older_than", "younger_than"),
            (r"(?:the\s+)?(\w+)\s+(?:is\s+)?younger\s+than\s+(?:the\s+)?(\w+)", "younger_than", "older_than"),
            
            # Size relations
            (r"(?:the\s+)?(\w+)\s+(?:is\s+)?bigger\s+than\s+(?:the\s+)?(\w+)", "bigger_than", "smaller_than"),
            (r"(?:the\s+)?(\w+)\s+(?:is\s+)?smaller\s+than\s+(?:the\s+)?(\w+)", "smaller_than", "bigger_than"),
            (r"(?:the\s+)?(\w+)\s+(?:is\s+)?larger\s+than\s+(?:the\s+)?(\w+)", "bigger_than", "smaller_than"),
            
            # Speed relations
            (r"(?:the\s+)?(\w+)\s+(?:is\s+)?faster\s+than\s+(?:the\s+)?(\w+)", "faster_than", "slower_than"),
            (r"(?:the\s+)?(\w+)\s+(?:is\s+)?slower\s+than\s+(?:the\s+)?(\w+)", "slower_than", "faster_than"),
            
            # Weight relations
            (r"(?:the\s+)?(\w+)\s+(?:is\s+)?heavier\s+than\s+(?:the\s+)?(\w+)", "heavier_than", "lighter_than"),
            (r"(?:the\s+)?(\w+)\s+(?:is\s+)?lighter\s+than\s+(?:the\s+)?(\w+)", "lighter_than", "heavier_than"),
            
            # Strength relations
            (r"(?:the\s+)?(\w+)\s+(?:is\s+)?stronger\s+than\s+(?:the\s+)?(\w+)", "stronger_than", "weaker_than"),
            (r"(?:the\s+)?(\w+)\s+(?:is\s+)?weaker\s+than\s+(?:the\s+)?(\w+)", "weaker_than", "stronger_than"),
        ]

        # Compile patterns for performance
        self.patterns = []
        for pattern_str, rel_type, inv_type in relation_defs:
            compiled = re.compile(pattern_str, re.IGNORECASE)
            self.patterns.append((compiled, rel_type, inv_type))

        logger.debug(f"RelationExtractionEngine initialized with {len(self.patterns)} patterns")

    def extract(self, text: str) -> list[dict[str, Any]]:
        """Extract entity relationships from text.

        Args:
            text: Natural language text to extract relations from

        Returns:
            List of relation dictionaries with keys: source, target, relation_type, properties
        """
        if not text:
            return []

        relations = []
        
        for pattern, relation_type, inverse_type in self.patterns:
            matches = pattern.finditer(text)
            
            for match in matches:
                source = match.group(1)
                target = match.group(2)
                
                # Skip if entities are the same
                if source.lower() == target.lower():
                    continue
                
                # Add primary relation
                relations.append({
                    "source": source,
                    "target": target,
                    "relation_type": relation_type,
                    "properties": {
                        "extracted_from": text[:100],  # Store context snippet
                        "confidence": 1.0,  # Deterministic extraction
                    }
                })
                
                # Add inverse relation for complete graph coverage
                relations.append({
                    "source": target,
                    "target": source,
                    "relation_type": inverse_type,
                    "properties": {
                        "extracted_from": text[:100],
                        "confidence": 1.0,
                        "inverse_of": relation_type,
                    }
                })

        if relations:
            logger.debug(f"Extracted {len(relations)} relations (including inverses) from text")

        return relations

    def extract_and_ingest(
        self,
        text: str,
        cache,
        session_store=None,
        session_id: str | None = None
    ) -> int:
        """Extract relations and ingest them into cache and session store.

        Args:
            text: Natural language text to extract relations from
            cache: RelationalCache instance to store relations
            session_store: Optional SessionStore instance
            session_id: Optional session ID for session store

        Returns:
            Count of relations extracted (including inverses)
        """
        relations = self.extract(text)
        
        if not relations:
            return 0

        for relation in relations:
            # Add to relational cache
            try:
                cache.add_relationship(
                    source=relation["source"],
                    target=relation["target"],
                    relation_type=relation["relation_type"],
                    properties=relation.get("properties", {})
                )
            except Exception as e:
                logger.warning(
                    f"Failed to add relationship to cache: {relation['source']} "
                    f"{relation['relation_type']} {relation['target']}: {e}"
                )

            # Add to session store if provided
            if session_store and session_id:
                try:
                    session_store.append_event(
                        session_id,
                        "entity_relation",
                        {
                            "source": relation["source"],
                            "target": relation["target"],
                            "relation_type": relation["relation_type"],
                            "properties": relation.get("properties", {})
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to append entity_relation event to session: {e}"
                    )

        logger.info(
            f"Ingested {len(relations)} relations into cache "
            f"{'and session store' if session_store and session_id else ''}"
        )

        return len(relations)
