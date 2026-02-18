"""Memory-aware hypothesis engine for generating reasoning hypotheses from relational cache.

This module implements a graph-based reasoning system that generates hypotheses
directly from the relational cache without invoking the LLM. It traverses the
relationship graph, computes confidence scores, and calculates entropy to
determine if LLM invocation is needed.
"""

import math
import re
from collections import deque
from pathlib import Path
from typing import Any

from nanobot.memory.relational_cache import RelationalCache


class HypothesisEngine:
    """Generates reasoning hypotheses from relational cache using graph traversal.

    The engine analyzes queries, traverses the relationship graph, and generates
    hypotheses with confidence scores and entropy measurements. When entropy is
    below threshold, these hypotheses can be used directly without LLM invocation.

    Attributes:
        cache: RelationalCache instance
        entropy_threshold: Threshold for LLM invocation (default: 0.8)
    """

    def __init__(self, workspace: Path, entropy_threshold: float = 0.8):
        """Initialize hypothesis engine.

        Args:
            workspace: Path to workspace directory
            entropy_threshold: Entropy threshold above which LLM is invoked
        """
        self.cache = RelationalCache(workspace)
        self.entropy_threshold = entropy_threshold

    def generate_hypotheses(
        self,
        query: str,
        max_depth: int = 3,
        max_hypotheses: int = 3
    ) -> dict[str, Any]:
        """Generate hypotheses for a query from the relational cache.

        Args:
            query: Natural language query
            max_depth: Maximum graph traversal depth
            max_hypotheses: Maximum number of hypotheses to generate

        Returns:
            Dictionary with hypotheses, entropy, and metadata
        """
        # Parse query to extract entities and relationships
        parsed = self._parse_query(query)

        # Generate hypotheses based on query type
        if parsed["type"] == "comparison":
            hypotheses = self._generate_comparison_hypotheses(
                parsed["entities"],
                parsed["attribute"],
                max_hypotheses
            )
        elif parsed["type"] == "relationship":
            hypotheses = self._generate_relationship_hypotheses(
                parsed["source"],
                parsed["target"],
                parsed["relation_type"],
                max_depth,
                max_hypotheses
            )
        elif parsed["type"] == "attribute":
            hypotheses = self._generate_attribute_hypotheses(
                parsed["entity"],
                parsed["attribute"],
                max_hypotheses
            )
        else:
            # General query - try to infer from patterns
            hypotheses = self._generate_general_hypotheses(
                query,
                max_hypotheses
            )

        # Compute entropy based on hypotheses confidence
        entropy = self._compute_entropy(hypotheses)

        return {
            "hypotheses": hypotheses,
            "entropy": entropy,
            "requires_llm": entropy >= self.entropy_threshold,
            "query_type": parsed["type"],
            "entities_found": len(self.cache.get_entities()),
            "relationships_available": len(self._get_all_relationships())
        }

    def _parse_query(self, query: str) -> dict[str, Any]:
        """Parse query to identify type and extract entities/attributes.

        Args:
            query: Natural language query

        Returns:
            Dictionary with query type and extracted information
        """
        query_lower = query.lower()
        entities = self.cache.get_entities()
        entity_names = list(entities.keys())

        # Check for comparison queries (who is taller/shorter, etc.)
        comparison_patterns = [
            (r"who is (taller|shorter|bigger|smaller|faster|slower)", "height"),
            (r"which is (taller|shorter|bigger|smaller)", "height"),
            (r"compare (.*?) and (.*?)(?:\s+by\s+(.*?))?", "comparison"),
        ]

        for pattern, attr_type in comparison_patterns:
            match = re.search(pattern, query_lower)
            if match:
                # Find mentioned entities
                mentioned_entities = [e for e in entity_names if e.lower() in query_lower]
                if len(mentioned_entities) >= 2:
                    return {
                        "type": "comparison",
                        "entities": mentioned_entities[:2],
                        "attribute": attr_type
                    }

        # Check for relationship queries
        relationship_patterns = [
            r"is (.*?) (.*?) than (.*?)[?\.]",
            r"does (.*?) (.*?) (.*?)[?\.]",
            r"(.*?) is (.*?) to (.*?)[?\.]",
        ]

        for pattern in relationship_patterns:
            match = re.search(pattern, query_lower)
            if match:
                groups = match.groups()
                # Try to find entities in the groups
                source = next((e for e in entity_names if e.lower() in groups[0].lower()), None)
                target = next((e for e in entity_names if groups[2] and e.lower() in groups[2].lower()), None)
                if source and target:
                    return {
                        "type": "relationship",
                        "source": source,
                        "target": target,
                        "relation_type": groups[1] if len(groups) > 1 else "related"
                    }

        # Check for attribute queries
        attribute_patterns = [
            r"what is (.*?)['']s (.*?)[?\.]",
            r"how (.*?) is (.*?)[?\.]",
            r"(.*?) (height|age|weight|size)",
        ]

        for pattern in attribute_patterns:
            match = re.search(pattern, query_lower)
            if match:
                groups = match.groups()
                entity = next((e for e in entity_names if e.lower() in " ".join(groups).lower()), None)
                if entity:
                    # Try to identify attribute
                    attr = None
                    if "height" in query_lower or "tall" in query_lower:
                        attr = "height"
                    elif "age" in query_lower or "old" in query_lower:
                        attr = "age"
                    elif "weight" in query_lower or "heavy" in query_lower:
                        attr = "weight"

                    return {
                        "type": "attribute",
                        "entity": entity,
                        "attribute": attr
                    }

        # Default to general query
        mentioned_entities = [e for e in entity_names if e.lower() in query_lower]
        return {
            "type": "general",
            "entities": mentioned_entities
        }

    def _generate_comparison_hypotheses(
        self,
        entities: list[str],
        attribute: str,
        max_hypotheses: int
    ) -> list[dict[str, Any]]:
        """Generate hypotheses for comparison queries.

        Args:
            entities: List of entity names to compare
            attribute: Attribute to compare (e.g., 'height')
            max_hypotheses: Maximum number of hypotheses

        Returns:
            List of hypothesis dictionaries
        """
        hypotheses = []
        entity_data = self.cache.get_entities()

        if len(entities) < 2:
            return hypotheses

        # Get attribute values for each entity
        values = {}
        for entity in entities:
            if entity in entity_data:
                attrs = entity_data[entity].get("attributes", {})
                if attribute in attrs:
                    values[entity] = attrs[attribute]

        # If we have values for both, we can make a confident hypothesis
        if len(values) == 2:
            entity1, entity2 = entities[0], entities[1]
            val1, val2 = values.get(entity1), values.get(entity2)

            if val1 is not None and val2 is not None:
                if val1 > val2:
                    result = f"{entity1} is {attribute} than {entity2}"
                    comparison = "greater"
                elif val1 < val2:
                    result = f"{entity2} is {attribute} than {entity1}"
                    comparison = "less"
                else:
                    result = f"{entity1} and {entity2} have equal {attribute}"
                    comparison = "equal"

                hypotheses.append({
                    "intent": f"compare_{attribute}",
                    "confidence": 0.95,  # High confidence from direct data
                    "reasoning": f"Direct {attribute} comparison: {entity1}={val1}, {entity2}={val2}",
                    "result": result,
                    "evidence": {
                        "type": "direct_attribute",
                        "entities": {entity1: val1, entity2: val2},
                        "comparison": comparison
                    }
                })
        else:
            # Try to infer from relationships
            relationships = self.cache.get_entity_relationships(entities[0])
            for rel in relationships:
                if rel["target"] in entities and attribute in rel["type"]:
                    hypotheses.append({
                        "intent": f"infer_comparison_{attribute}",
                        "confidence": 0.7,  # Lower confidence from inference
                        "reasoning": f"Inferred from relationship: {rel['type']}",
                        "result": f"Relationship suggests comparison based on {attribute}",
                        "evidence": {
                            "type": "relationship_inference",
                            "relationship": rel
                        }
                    })

        return hypotheses[:max_hypotheses]

    def _generate_relationship_hypotheses(
        self,
        source: str,
        target: str,
        relation_type: str | None,
        max_depth: int,
        max_hypotheses: int
    ) -> list[dict[str, Any]]:
        """Generate hypotheses for relationship queries using graph traversal.

        Args:
            source: Source entity
            target: Target entity
            relation_type: Type of relationship to find (optional)
            max_depth: Maximum traversal depth
            max_hypotheses: Maximum number of hypotheses

        Returns:
            List of hypothesis dictionaries
        """
        hypotheses = []

        # Find direct relationships
        direct_rels = self.cache.get_entity_relationships(source)
        for rel in direct_rels:
            if rel["target"] == target:
                if relation_type is None or relation_type in rel["type"].lower():
                    hypotheses.append({
                        "intent": "direct_relationship",
                        "confidence": 0.9,
                        "reasoning": f"Direct relationship found: {rel['type']}",
                        "result": f"{source} {rel['type']} {target}",
                        "evidence": {
                            "type": "direct",
                            "path_length": 1,
                            "relationship": rel
                        }
                    })

        # Find indirect paths using BFS
        if len(hypotheses) < max_hypotheses:
            paths = self._find_paths_bfs(source, target, max_depth)
            for i, path in enumerate(paths[:max_hypotheses - len(hypotheses)]):
                confidence = 0.8 / (len(path) ** 0.5)  # Decrease confidence with path length
                hypotheses.append({
                    "intent": "indirect_relationship",
                    "confidence": min(confidence, 0.85),
                    "reasoning": f"Indirect path found with {len(path)} steps",
                    "result": f"{source} connected to {target} through {' -> '.join(path)}",
                    "evidence": {
                        "type": "indirect",
                        "path_length": len(path),
                        "path": path
                    }
                })

        return hypotheses[:max_hypotheses]

    def _generate_attribute_hypotheses(
        self,
        entity: str,
        attribute: str | None,
        max_hypotheses: int
    ) -> list[dict[str, Any]]:
        """Generate hypotheses for attribute queries.

        Args:
            entity: Entity name
            attribute: Attribute name (optional)
            max_hypotheses: Maximum number of hypotheses

        Returns:
            List of hypothesis dictionaries
        """
        hypotheses = []
        entity_data = self.cache.get_entities()

        if entity not in entity_data:
            return hypotheses

        attrs = entity_data[entity].get("attributes", {})

        if attribute:
            # Query about specific attribute
            if attribute in attrs:
                hypotheses.append({
                    "intent": f"get_attribute_{attribute}",
                    "confidence": 0.95,
                    "reasoning": f"Direct attribute lookup for {entity}.{attribute}",
                    "result": f"{entity}'s {attribute} is {attrs[attribute]}",
                    "evidence": {
                        "type": "direct_attribute",
                        "entity": entity,
                        "attribute": attribute,
                        "value": attrs[attribute]
                    }
                })
        else:
            # Return all attributes
            for attr_name, attr_value in list(attrs.items())[:max_hypotheses]:
                hypotheses.append({
                    "intent": f"get_attribute_{attr_name}",
                    "confidence": 0.9,
                    "reasoning": f"Available attribute for {entity}",
                    "result": f"{entity}'s {attr_name} is {attr_value}",
                    "evidence": {
                        "type": "direct_attribute",
                        "entity": entity,
                        "attribute": attr_name,
                        "value": attr_value
                    }
                })

        return hypotheses[:max_hypotheses]

    def _generate_general_hypotheses(
        self,
        query: str,
        max_hypotheses: int
    ) -> list[dict[str, Any]]:
        """Generate hypotheses for general queries.

        Args:
            query: Query string
            max_hypotheses: Maximum number of hypotheses

        Returns:
            List of hypothesis dictionaries
        """
        hypotheses = []

        # Look for patterns that match the query
        patterns = self.cache.get_patterns()
        query_words = set(query.lower().split())

        for pattern in patterns[:max_hypotheses * 2]:
            pattern_data = pattern.get("data", {})
            pattern_text = str(pattern_data).lower()
            pattern_words = set(pattern_text.split())

            # Calculate word overlap
            overlap = len(query_words & pattern_words)
            if overlap > 0:
                confidence = min(0.7, overlap / len(query_words))
                hypotheses.append({
                    "intent": f"pattern_match_{pattern['type']}",
                    "confidence": confidence,
                    "reasoning": f"Matched pattern type {pattern['type']} with {overlap} word overlap",
                    "result": f"Related pattern found: {pattern['type']}",
                    "evidence": {
                        "type": "pattern_match",
                        "pattern_id": pattern.get("id"),
                        "pattern_type": pattern["type"],
                        "overlap_count": overlap
                    }
                })

        return sorted(hypotheses, key=lambda h: h["confidence"], reverse=True)[:max_hypotheses]

    def _find_paths_bfs(
        self,
        source: str,
        target: str,
        max_depth: int
    ) -> list[list[str]]:
        """Find paths between entities using BFS.

        Args:
            source: Source entity
            target: Target entity
            max_depth: Maximum path depth

        Returns:
            List of paths (each path is a list of entity names)
        """
        paths = []
        queue = deque([(source, [source])])
        visited = {source}

        while queue and len(paths) < 5:  # Limit to 5 paths
            current, path = queue.popleft()

            if len(path) > max_depth:
                continue

            if current == target and len(path) > 1:
                paths.append(path)
                continue

            # Get relationships for current entity
            relationships = self.cache.get_entity_relationships(current)
            for rel in relationships:
                next_entity = rel["target"] if rel["source"] == current else rel["source"]

                if next_entity not in visited or next_entity == target:
                    new_path = path + [next_entity]
                    queue.append((next_entity, new_path))

                    if next_entity != target:
                        visited.add(next_entity)

        return paths

    def _compute_entropy(self, hypotheses: list[dict[str, Any]]) -> float:
        """Compute entropy of hypotheses based on confidence distribution.

        High entropy (close to 1.0) indicates ambiguity/uncertainty.
        Low entropy (close to 0.0) indicates clear answer.

        Args:
            hypotheses: List of hypothesis dictionaries with confidence scores

        Returns:
            Entropy value between 0.0 and 1.0
        """
        if not hypotheses:
            return 1.0  # Maximum uncertainty with no hypotheses

        # Normalize confidences to create probability distribution
        confidences = [h["confidence"] for h in hypotheses]
        total = sum(confidences)

        if total == 0:
            return 1.0

        probs = [c / total for c in confidences]

        # Calculate Shannon entropy
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize to [0, 1] range
        # Max entropy for n hypotheses is log2(n)
        max_entropy = math.log2(len(hypotheses)) if len(hypotheses) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Adjust based on absolute confidence levels
        max_confidence = max(confidences)
        if max_confidence > 0.9:
            # Very confident in top hypothesis - reduce entropy
            normalized_entropy *= 0.5
        elif max_confidence < 0.5:
            # Low confidence across the board - increase entropy
            normalized_entropy = min(1.0, normalized_entropy * 1.5)

        return min(1.0, max(0.0, normalized_entropy))

    def _get_all_relationships(self) -> list[dict[str, Any]]:
        """Get all relationships from cache.

        Returns:
            List of relationship dictionaries
        """
        cache_data = self.cache._read_cache()
        return cache_data.get("relationships", [])

    def query(
        self,
        query_string: str,
        verbose: bool = False
    ) -> dict[str, Any]:
        """Execute a query against the relational cache.

        Args:
            query_string: Natural language query
            verbose: Whether to include detailed evidence

        Returns:
            Query result with hypotheses, confidence, and answer
        """
        result = self.generate_hypotheses(query_string)

        # Format answer from top hypothesis
        if result["hypotheses"]:
            top_hypothesis = result["hypotheses"][0]
            answer = top_hypothesis.get("result", "Unable to determine answer")
            confidence = top_hypothesis["confidence"]
        else:
            answer = "No information found in relational cache"
            confidence = 0.0

        response = {
            "query": query_string,
            "answer": answer,
            "confidence": confidence,
            "entropy": result["entropy"],
            "requires_llm": result["requires_llm"],
            "query_type": result["query_type"]
        }

        if verbose:
            response["hypotheses"] = result["hypotheses"]
            response["cache_stats"] = {
                "entities": result["entities_found"],
                "relationships": result["relationships_available"]
            }

        return response
