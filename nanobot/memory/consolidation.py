"""Consolidation pipeline for memory system.

This module implements the consolidation pipeline that transforms episodic session
data into relational patterns, then into fractal nodes, and finally rolls them up
into the main MEMORY.yaml file.

Pipeline stages:
1. Sessions → Relational Cache: Extract patterns and relationships from session events
2. Relational Cache → Fractal Nodes: Convert patterns into hierarchical memory nodes
3. Fractal Nodes → MEMORY.yaml: Rollup fractal nodes into long-term memory
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_types import ContentType
from nanobot.memory.relational_cache import RelationalCache
from nanobot.memory.session_store import SessionStore


class ConsolidationPipeline:
    """Manages the multi-stage memory consolidation pipeline.

    The pipeline processes episodic session logs through several stages to
    create increasingly abstract and structured memory representations.

    Attributes:
        workspace: Path to workspace directory
        session_store: SessionStore instance
        relational_cache: RelationalCache instance
        memory_store: MemoryStore instance
    """

    def __init__(self, workspace: Path, memory_config: dict[str, Any] | None = None):
        """Initialize consolidation pipeline.

        Args:
            workspace: Path to workspace directory
            memory_config: Optional memory configuration dictionary
        """
        self.workspace = Path(workspace)
        self.session_store = SessionStore(workspace)
        self.relational_cache = RelationalCache(workspace)
        self.memory_store = MemoryStore(workspace, memory_config or {})

    def consolidate_session(
        self,
        session_id: str,
        extract_entities: bool = True,
        archive_after: bool = False
    ) -> dict[str, int]:
        """Consolidate a session into the relational cache.

        Extracts patterns and relationships from session events and stores them
        in the relational cache for further processing.

        Args:
            session_id: Session identifier
            extract_entities: Whether to extract entity relationships
            archive_after: Whether to archive session after consolidation

        Returns:
            Dictionary with counts of extracted patterns and relationships
        """
        # Load session data
        session_data = self.session_store.load(session_id)
        events = session_data.get("events", [])

        patterns_added = 0
        relationships_added = 0

        # Process events
        for event in events:
            event_type = event.get("type", "")
            payload = event.get("payload", {})

            # Extract patterns based on event type
            if event_type == "interaction":
                # User-agent interaction pattern
                self.relational_cache.add_pattern(
                    pattern_type="interaction",
                    data={
                        "user_message": payload.get("user_message", ""),
                        "agent_response": payload.get("agent_response", ""),
                        "timestamp": event.get("timestamp")
                    }
                )
                patterns_added += 1

            elif event_type == "reasoning":
                # Reasoning pattern
                self.relational_cache.add_pattern(
                    pattern_type="reasoning",
                    data={
                        "hypotheses": payload.get("hypotheses", []),
                        "entropy": payload.get("entropy", 0.0),
                        "strategic_direction": payload.get("strategic_direction", ""),
                        "timestamp": event.get("timestamp")
                    }
                )
                patterns_added += 1

            elif event_type == "entity_relation" and extract_entities:
                # Entity relationship
                source = payload.get("source")
                target = payload.get("target")
                relation_type = payload.get("relation_type")

                if source and target and relation_type:
                    self.relational_cache.add_relationship(
                        source=source,
                        target=target,
                        relation_type=relation_type,
                        properties=payload.get("properties", {})
                    )
                    relationships_added += 1

            elif event_type == "entity_attribute" and extract_entities:
                # Entity attribute update
                entity = payload.get("entity")
                attribute = payload.get("attribute")
                value = payload.get("value")

                if entity and attribute and value is not None:
                    self.relational_cache.update_entity_attribute(
                        entity=entity,
                        attribute=attribute,
                        value=value
                    )

        # Archive session if requested
        if archive_after:
            self.session_store.archive(session_id)

        return {
            "patterns_added": patterns_added,
            "relationships_added": relationships_added
        }

    def consolidate_patterns_to_fractal(
        self,
        pattern_limit: int = 50,
        min_pattern_count: int = 3
    ) -> list[str]:
        """Consolidate relational cache patterns into fractal nodes.

        Groups related patterns and creates fractal memory nodes via MemoryStore.

        Args:
            pattern_limit: Maximum number of patterns to process
            min_pattern_count: Minimum patterns needed to create a fractal node

        Returns:
            List of created fractal node IDs
        """
        patterns = self.relational_cache.get_patterns(limit=pattern_limit)

        # Group patterns by type
        pattern_groups = {}
        for pattern in patterns:
            pattern_type = pattern.get("type", "unknown")
            if pattern_type not in pattern_groups:
                pattern_groups[pattern_type] = []
            pattern_groups[pattern_type].append(pattern)

        created_nodes = []

        # Create fractal nodes for pattern groups
        for pattern_type, group_patterns in pattern_groups.items():
            if len(group_patterns) < min_pattern_count:
                continue

            # Build lesson content from patterns
            lesson_content = self._build_lesson_from_patterns(pattern_type, group_patterns)

            # Save via MemoryStore (which creates and returns a FractalNode)
            node = self.memory_store.save_fractal_node(
                content=lesson_content,
                tags=[pattern_type, "consolidated", "relational"],
                summary=f"Consolidated {pattern_type} patterns ({len(group_patterns)} patterns)",
                content_type=ContentType.TEXT
            )

            created_nodes.append(node.id)

        return created_nodes

    def rollup_to_long_term(
        self,
        node_ids: list[str] | None = None,
        summary_only: bool = True
    ) -> None:
        """Rollup fractal nodes into MEMORY.yaml long-term storage.

        Takes fractal nodes and integrates their knowledge into the main
        long-term memory file.

        Args:
            node_ids: Optional specific node IDs to rollup (if None, uses recent nodes)
            summary_only: Whether to rollup just summaries (True) or full content (False)
        """
        if node_ids is None:
            # Get recent fractal nodes from index
            recent_nodes = self.memory_store.retrieve_relevant_nodes(
                query="consolidated relational",
                k=10
            )
            node_ids = [node.id for node in recent_nodes]

        # Build consolidated memory entry
        for node_id in node_ids:
            try:
                # Read fractal node
                nodes = self.memory_store.retrieve_relevant_nodes(
                    query=node_id,
                    k=1
                )

                if nodes:
                    node = nodes[0]
                    content = node.context_summary if summary_only else node.content

                    # Append to long-term memory
                    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    entry = f"\n[{timestamp}] Consolidated Memory:\n{content}\n"

                    self.memory_store.append_long_term(entry)

            except Exception:
                # Skip nodes that can't be processed
                continue

    def run_full_pipeline(
        self,
        session_ids: list[str] | None = None,
        archive_sessions: bool = True
    ) -> dict[str, Any]:
        """Run the complete consolidation pipeline.

        Args:
            session_ids: Optional specific sessions to process (if None, processes all)
            archive_sessions: Whether to archive sessions after consolidation

        Returns:
            Dictionary with pipeline statistics
        """
        stats = {
            "sessions_processed": 0,
            "patterns_extracted": 0,
            "relationships_extracted": 0,
            "fractal_nodes_created": 0,
            "long_term_entries": 0
        }

        # Determine which sessions to process
        if session_ids is None:
            all_sessions = self.session_store.list_sessions()
            session_ids = [s["session_id"] for s in all_sessions]

        # Stage 1: Consolidate sessions to relational cache
        for session_id in session_ids:
            try:
                result = self.consolidate_session(
                    session_id=session_id,
                    archive_after=archive_sessions
                )
                stats["sessions_processed"] += 1
                stats["patterns_extracted"] += result["patterns_added"]
                stats["relationships_extracted"] += result["relationships_added"]
            except Exception:
                # Skip problematic sessions
                continue

        # Stage 2: Consolidate patterns to fractal nodes
        try:
            created_nodes = self.consolidate_patterns_to_fractal()
            stats["fractal_nodes_created"] = len(created_nodes)

            # Stage 3: Rollup to long-term memory
            if created_nodes:
                self.rollup_to_long_term(node_ids=created_nodes)
                stats["long_term_entries"] = len(created_nodes)
        except Exception:
            # Continue even if fractal consolidation fails
            pass

        return stats

    def _build_lesson_from_patterns(
        self,
        pattern_type: str,
        patterns: list[dict[str, Any]]
    ) -> str:
        """Build lesson content from a group of patterns.

        Args:
            pattern_type: Type of patterns being consolidated
            patterns: List of pattern dictionaries

        Returns:
            Formatted lesson content string
        """
        lesson_parts = [f"# Consolidated {pattern_type.title()} Patterns\n"]

        lesson_parts.append(f"Based on {len(patterns)} observations:\n")

        for i, pattern in enumerate(patterns[:10], 1):  # Limit to 10 examples
            data = pattern.get("data", {})
            timestamp = pattern.get("timestamp", "")[:10]  # Date only

            if pattern_type == "interaction":
                lesson_parts.append(
                    f"{i}. [{timestamp}] User interaction with response"
                )
            elif pattern_type == "reasoning":
                entropy = data.get("entropy", 0.0)
                lesson_parts.append(
                    f"{i}. [{timestamp}] Reasoning with entropy {entropy:.2f}"
                )
            else:
                lesson_parts.append(f"{i}. [{timestamp}] {pattern_type} pattern")

        return "\n".join(lesson_parts)
