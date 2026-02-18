"""Memory system for persistent agent memory."""

import base64
import json
import logging
import mimetypes
import time
from collections import deque
from pathlib import Path
from typing import Any

from nanobot.agent.memory_types import ActiveLearningState, ContentType, FractalNode
from nanobot.runtime.state import state
from nanobot.utils.helpers import ensure_dir

logger = logging.getLogger(__name__)

# Constants
MAX_CONTENT_PREVIEW_LENGTH = 200  # Maximum length for content preview in retrieval
INITIAL_CANDIDATE_MULTIPLIER = 2
ENTANGLEMENT_STRENGTH_THRESHOLD = 0.5
MAX_ENTANGLEMENT_HOPS = 2


class MemoryStore:
    """
    Upgraded memory system with Fractal Memory and Active Learning State.

    Maintains backward compatibility with legacy MEMORY.md and HISTORY.md,
    while adding new capabilities:
    - Fractal nodes stored as lesson_X.json
    - Lightweight fractal_index.json for fast retrieval
    - Active Learning State in ALS.json
    - Token-efficient top-K retrieval
    - Multi-modal memory (text, code, images)
    - Hierarchical node relationships
    - mem0 integration for vector embeddings
    """

    def __init__(self, workspace: Path, config: dict[str, Any] | None = None):
        self.workspace = workspace
        self.memory_dir = ensure_dir(workspace / "memory")
        self.config = config or {}

        # Legacy files (preserved for backward compatibility)
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"

        # Triune Memory: YAML paths for token-efficient storage
        self.memory_yaml = self.memory_dir / "MEMORY.yaml"
        self.history_yaml = self.memory_dir / "HISTORY.yaml"

        # --- NEW: Fractal Architecture ---
        self.archives_dir = ensure_dir(self.memory_dir / "archives")
        self.index_file = self.memory_dir / "fractal_index.json"
        self.als_file = self.memory_dir / "ALS.json"

        # Pattern Cache for high-entropy latent states
        self.pattern_cache_file = self.memory_dir / "pattern_cache.jsonl"

        # mem0 provider (optional)
        self._mem0_provider = None
        self._init_provider()

        self._init_fractal_structure()

    def _init_provider(self) -> None:
        """Initialize memory provider based on configuration."""
        if not state.mem0_enabled:
            return
        provider_type = self.config.get("provider", "local")

        if provider_type == "mem0":
            try:
                from nanobot.agent.mem0_provider import Mem0Provider
                self._mem0_provider = Mem0Provider(self.config, self.workspace)
                if self._mem0_provider.is_available():
                    logger.info("mem0 provider initialized")
                else:
                    logger.warning("mem0 provider not available, using local storage")
                    self._mem0_provider = None
            except ImportError:
                logger.warning("mem0_provider module not found, using local storage")
                self._mem0_provider = None
            except Exception as e:
                logger.error(f"Failed to initialize mem0 provider: {e}")
                self._mem0_provider = None

    def _init_fractal_structure(self) -> None:
        """Initialize index and ALS if they don't exist."""
        if not self.index_file.exists():
            self.index_file.write_text(json.dumps([], indent=2))
        if not self.als_file.exists():
            default_als = ActiveLearningState().model_dump_json(indent=2)
            self.als_file.write_text(default_als)

    def read_long_term(self) -> str:
        """
        Read long-term memory content.

        Triune Memory: Prefers .yaml for token efficiency,
        falls back to .md for backward compatibility.
        """
        # Prefer YAML (token-efficient)
        if self.memory_yaml.exists():
            try:
                import yaml
                data = yaml.safe_load(self.memory_yaml.read_text(encoding="utf-8"))
                return self._yaml_data_to_str(data)
            except Exception as e:
                logger.warning(f"Failed to load MEMORY.yaml: {e}")

        # Fall back to MD (backward compatibility)
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def _yaml_data_to_str(self, data: dict[str, Any]) -> str:
        """
        Convert YAML memory data to string format.

        Uses shared formatting from translator module.
        """
        from nanobot.utils.translator import yaml_data_to_context

        if not isinstance(data, dict):
            return str(data)

        return yaml_data_to_context(data)

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_long_term(self, content: str) -> None:
        """Append to long-term memory (teaching flow entries)."""
        with open(self.memory_file, "a", encoding="utf-8") as f:
            f.write("\n" + content.rstrip() + "\n")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    # --- NEW METHODS for Fractal Memory ---

    def save_fractal_node(
        self,
        content: str,
        tags: list[str],
        summary: str,
        content_type: ContentType = ContentType.TEXT,
        language: str | None = None,
        binary_data: str | None = None,
        mime_type: str | None = None,
        parent_id: str | None = None,
    ) -> FractalNode:
        """
        Creates a lesson archive and updates the index.

        Supports multi-modal content (text, code, images) and hierarchical relationships.

        Args:
            content: The core lesson/fact content
            tags: List of tags for categorization and retrieval
            summary: Brief summary for the index
            content_type: Type of content (text, code, image, mixed)
            language: Programming language for code snippets
            binary_data: Base64-encoded binary data for images
            mime_type: MIME type for binary data
            parent_id: ID of parent node (for hierarchical relationships)

        Returns:
            The created FractalNode
        """
        # Use mem0 provider if available
        if self._mem0_provider:
            try:
                metadata = {}
                if language:
                    metadata["language"] = language
                if parent_id:
                    metadata["parent_id"] = parent_id
                if mime_type:
                    metadata["mime_type"] = mime_type

                node = self._mem0_provider.add_memory(
                    content=content,
                    tags=tags,
                    summary=summary,
                    content_type=content_type,
                    metadata=metadata,
                )

                # Update binary data if provided
                if binary_data:
                    node.binary_data = binary_data
                    node.mime_type = mime_type

                # Update language if provided
                if language:
                    node.language = language

                # Update parent relationship
                if parent_id:
                    node.parent_id = parent_id
                    parent = self.get_node_by_id(parent_id)
                    if parent:
                        parent.children_ids.append(node.id)
                        node.depth = parent.depth + 1
                        self._update_node(parent)

                # Save with updated fields
                archive_path = self.archives_dir / f"lesson_{node.id}.json"
                archive_path.write_text(node.model_dump_json(indent=2), encoding="utf-8")

                # Update index
                self._update_index(node)

                return node

            except Exception as e:
                logger.error(f"Failed to use mem0 provider: {e}")
                # Fall through to local storage

        # Local storage implementation
        node = FractalNode(
            content=content,
            tags=tags,
            context_summary=summary,
            content_type=content_type,
            language=language,
            binary_data=binary_data,
            mime_type=mime_type,
            parent_id=parent_id,
        )

        # Handle hierarchical relationship
        if parent_id:
            parent = self.get_node_by_id(parent_id)
            if parent:
                parent.children_ids.append(node.id)
                node.depth = parent.depth + 1
                self._update_node(parent)

        # 1. Save full archive (Lesson)
        archive_path = self.archives_dir / f"lesson_{node.id}.json"
        archive_path.write_text(node.model_dump_json(indent=2), encoding="utf-8")

        # 2. Update Index
        self._update_index(node)

        logger.info(f"Saved Fractal Node: {node.id} with tags: {tags}")
        return node

    def _update_index(self, node: FractalNode) -> None:
        """Update the index with a new or modified node."""
        try:
            index_data = json.loads(self.index_file.read_text(encoding="utf-8"))
        except Exception:
            index_data = []

        # Remove existing entry if updating
        index_data = [entry for entry in index_data if entry.get("id") != node.id]

        # Add new entry
        index_entry = {
            "id": node.id,
            "tags": node.tags,
            "summary": node.context_summary,
            "timestamp": node.timestamp.isoformat(),
            "content_type": node.content_type.value,
            "parent_id": node.parent_id,
            "depth": node.depth,
        }
        index_data.append(index_entry)
        self.index_file.write_text(json.dumps(index_data, indent=2), encoding="utf-8")

    def _update_node(self, node: FractalNode) -> None:
        """Update an existing node."""
        archive_path = self.archives_dir / f"lesson_{node.id}.json"
        archive_path.write_text(node.model_dump_json(indent=2), encoding="utf-8")
        self._update_index(node)

    def _vector_search(self, query: str, k: int) -> list[tuple[FractalNode, float]]:
        """Return node candidates with a normalized relevance score."""
        if self._mem0_provider:
            nodes = self._mem0_provider.search_memories(query, k=k)
            if not nodes:
                return []
            denom = max(len(nodes) - 1, 1)
            return [(node, (len(nodes) - 1 - idx) / denom) for idx, node in enumerate(nodes)]

        try:
            index_data = json.loads(self.index_file.read_text(encoding="utf-8"))
        except Exception:
            return []

        query_words = set(query.lower().split())
        scored_entries: list[tuple[float, dict[str, Any]]] = []
        for entry in index_data:
            score = 0.0
            for tag in entry.get("tags", []):
                if tag.lower() in query_words:
                    score += 5.0
            summary_words = set(entry.get("summary", "").lower().split())
            score += float(len(query_words.intersection(summary_words)))
            if score > 0:
                scored_entries.append((score, entry))

        if not scored_entries:
            return []

        scored_entries.sort(key=lambda x: x[0], reverse=True)
        max_score = scored_entries[0][0]
        top_results = scored_entries[:k]
        results: list[tuple[FractalNode, float]] = []

        for score, entry in top_results:
            archive_path = self.archives_dir / f"lesson_{entry['id']}.json"
            if not archive_path.exists():
                continue
            try:
                node = FractalNode.model_validate_json(archive_path.read_text(encoding="utf-8"))
                normalized = score / max_score if max_score > 0 else 0.0
                results.append((node, normalized))
            except Exception as e:
                logger.warning(f"Could not load node {entry['id']}: {e}")

        return results

    def get_entangled_context(self, query: str, top_k: int = 5) -> list[FractalNode]:
        """
        Retrieve context with hybrid scoring:
        score = (semantic * w_s) + (entanglement * w_e) + (importance * w_i)
        """
        initial_results = self._vector_search(query, k=max(top_k * INITIAL_CANDIDATE_MULTIPLIER, 1))
        if not initial_results:
            return []

        candidates: dict[str, dict[str, Any]] = {
            node.id: {"node": node, "vec_score": score, "raw_entanglement": 0.0}
            for node, score in initial_results
        }
        visited: set[str] = set(candidates.keys())
        max_hops = max(int(self.config.get("max_entanglement_hops", MAX_ENTANGLEMENT_HOPS)), 0)
        queue = deque((node, 0) for node, _ in initial_results)

        while queue:
            node, hops = queue.popleft()
            if hops >= max_hops:
                continue
            base_score = candidates.get(node.id, {}).get("vec_score", 0.0)
            for ent_id, strength in node.entangled_ids.items():
                if ent_id in visited:
                    continue
                # Keep only strong links during one-hop expansion.
                if strength <= ENTANGLEMENT_STRENGTH_THRESHOLD:
                    continue
                ent_node = self.get_node_by_id(ent_id)
                if ent_node:
                    visited.add(ent_id)
                    candidates[ent_id] = {
                        "node": ent_node,
                        "vec_score": base_score * strength,
                        "raw_entanglement": 0.0,
                    }
                    queue.append((ent_node, hops + 1))

        max_entanglement = 0.0
        for candidate_id, data in candidates.items():
            incoming_score = 0.0
            for other in candidates.values():
                incoming_score += other["node"].entangled_ids.get(candidate_id, 0.0)
            data["raw_entanglement"] = incoming_score
            max_entanglement = max(max_entanglement, incoming_score)

        cfg = self.config or {}
        semantic_weight = float(cfg.get("semantic_weight", 0.7))
        entanglement_weight = float(cfg.get("entanglement_weight", 0.3))
        importance_weight = float(cfg.get("importance_weight", 0.0))
        total_weight = semantic_weight + entanglement_weight + importance_weight
        if total_weight <= 0:
            semantic_weight, entanglement_weight, importance_weight = 1.0, 0.0, 0.0
        else:
            semantic_weight /= total_weight
            entanglement_weight /= total_weight
            importance_weight /= total_weight
        decay_rate = float(cfg.get("importance_decay_rate", 0.01))

        final_scores: list[tuple[FractalNode, float]] = []
        for data in candidates.values():
            normalized_strength = (
                data["raw_entanglement"] / max_entanglement if max_entanglement > 0 else 0.0
            )
            node = data["node"]
            node_importance = float(getattr(node, "importance", 0.0))
            node_ts = node.timestamp.timestamp()
            age_hours = max((time.time() - node_ts) / 3600, 0.0)
            decayed_importance = node_importance * ((1 - decay_rate) ** age_hours)
            final_score = (data["vec_score"] * semantic_weight) + (
                normalized_strength * entanglement_weight
            ) + (
                decayed_importance * importance_weight
            )
            final_scores.append((node, final_score))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in final_scores[:top_k]]

    def retrieve_relevant_nodes(self, query: str, k: int = 5) -> str:
        """
        Token-efficient retrieval using semantic search (mem0) or keyword matching (local).

        Args:
            query: Search query
            k: Number of top results to return

        Returns:
            Formatted string with relevant memory nodes
        """
        if not state.fractal_memory_enabled and not state.mem0_enabled:
            return ""

        # Use mem0 provider if available
        if self._mem0_provider:
            try:
                nodes = self._mem0_provider.search_memories(query, k=k)
                return self._format_nodes(nodes)
            except Exception as e:
                logger.error(f"mem0 search failed: {e}")
                # Fall through to local search

        # Local keyword-based search
        try:
            index_data = json.loads(self.index_file.read_text(encoding="utf-8"))

            # Simple scoring: count overlapping words between query and tags/summary
            query_words = set(query.lower().split())

            scored_nodes = []
            for entry in index_data:
                score = 0
                # Check tags (higher weight)
                for tag in entry.get("tags", []):
                    if tag.lower() in query_words:
                        score += 5
                # Check summary (lower weight)
                summary_words = set(entry.get("summary", "").lower().split())
                score += len(query_words.intersection(summary_words))

                scored_nodes.append((score, entry))

            # Sort by score descending and take top K
            scored_nodes.sort(key=lambda x: x[0], reverse=True)
            top_nodes = scored_nodes[:k]

            # Load full content for top K nodes
            nodes = []
            for score, entry in top_nodes:
                if score == 0:
                    continue  # Skip irrelevant nodes

                archive_path = self.archives_dir / f"lesson_{entry['id']}.json"
                if archive_path.exists():
                    try:
                        node = FractalNode.model_validate_json(
                            archive_path.read_text(encoding="utf-8")
                        )
                        nodes.append(node)
                    except Exception as e:
                        logger.warning(f"Could not load node {entry['id']}: {e}")

            return self._format_nodes(nodes)

        except Exception as e:
            logger.error(f"Error retrieving nodes: {e}")
            return ""

    def _format_nodes(self, nodes: list[FractalNode]) -> str:
        """Format nodes for context display."""
        if not nodes:
            return ""

        context_str = "## Relevant Resources (Fractal Memory)\n"
        for node in nodes:
            timestamp_str = node.timestamp.strftime('%Y-%m-%d')
            tags_str = ', '.join(node.tags)

            # Format based on content type
            type_prefix = ""
            if node.content_type == ContentType.CODE:
                type_prefix = f"[{node.language or 'code'}] " if node.language else "[code] "
            elif node.content_type == ContentType.IMAGE:
                type_prefix = "[image] "
            elif node.content_type == ContentType.MIXED:
                type_prefix = "[mixed] "

            # Show hierarchy depth
            indent = "  " * node.depth

            context_str += f"{indent}- **{type_prefix}[{timestamp_str}] {tags_str}**: {node.content[:MAX_CONTENT_PREVIEW_LENGTH]}"
            if len(node.content) > MAX_CONTENT_PREVIEW_LENGTH:
                context_str += "..."
            context_str += "\n"

        return context_str

    def format_nodes_for_context(self, nodes: list[FractalNode]) -> str:
        """Format retrieved nodes for prompt context.

        Args:
            nodes: Retrieved fractal or entangled memory nodes.

        Returns:
            A formatted markdown string suitable for context injection.
        """
        return self._format_nodes(nodes)

    def get_als_context(self) -> str:
        """
        Returns the Active Learning State summary for context.

        Returns:
            Formatted ALS summary
        """
        if self.als_file.exists():
            try:
                als = ActiveLearningState.model_validate_json(
                    self.als_file.read_text(encoding="utf-8")
                )
                return (
                    f"## Active Learning State\n"
                    f"Current Focus: {als.current_focus}\n"
                    f"Evolution Stage: {als.evolution_stage}\n"
                )
            except Exception as e:
                logger.warning(f"Could not load ALS: {e}")
        return ""

    def update_als(
        self,
        focus: str | None = None,
        reflection: str | None = None,
        evolution_stage: int | None = None
    ) -> None:
        """
        Update the Active Learning State.

        Args:
            focus: New focus area (optional)
            reflection: New reflection to add (optional)
            evolution_stage: New evolution stage (optional)
        """
        try:
            # Load existing ALS
            if self.als_file.exists():
                als = ActiveLearningState.model_validate_json(
                    self.als_file.read_text(encoding="utf-8")
                )
            else:
                als = ActiveLearningState()

            # Update fields
            if focus:
                als.current_focus = focus
            if reflection:
                als.recent_reflections.append(reflection)
                # Keep only last 10 reflections
                als.recent_reflections = als.recent_reflections[-10:]
            if evolution_stage is not None:
                als.evolution_stage = evolution_stage

            from datetime import datetime
            als.last_updated = datetime.now()

            # Save
            self.als_file.write_text(als.model_dump_json(indent=2), encoding="utf-8")
            logger.info("Updated Active Learning State")

        except Exception as e:
            logger.error(f"Error updating ALS: {e}")

    # --- NEW: Entropy-Based Routing Methods ---

    def route_latent_state(
        self,
        user_message: str,
        hypotheses: list[dict[str, Any]],
        entropy: float,
        strategic_direction: str,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        """
        Route latent reasoning state based on entropy threshold.

        - Always appends teaching flow to MEMORY.md
        - Always appends timestamped entry to HISTORY.md
        - Routes to Fractal Memory if entropy < threshold (low entropy = clear intent)
        - Routes to Pattern Cache otherwise (high entropy = ambiguous)
        - Updates ALS with routing decision

        Args:
            user_message: The user's input message
            hypotheses: List of hypothesis dicts (intent, confidence, reasoning)
            entropy: Calculated entropy value
            strategic_direction: Strategic direction from reasoning
            model: Model name (optional, for metadata)
            provider: Provider name (optional, for metadata)
        """
        from datetime import datetime

        threshold = float(self.config.get("clarify_entropy_threshold", 0.8))
        timestamp = datetime.now()

        # Format teaching flow entry for MEMORY.md
        teaching_entry = self._format_teaching_flow(
            timestamp, user_message, hypotheses, entropy, strategic_direction
        )

        # Append to MEMORY.md
        self.append_long_term(teaching_entry)

        # Append to HISTORY.md
        history_entry = f"[{timestamp.strftime('%Y-%m-%d %H:%M')}] User message: {user_message[:100]}"
        if len(user_message) > 100:
            history_entry += "..."
            history_entry += f"\nLatent reasoning: entropy={entropy:.3f}, hypotheses={len(hypotheses)}"
        self.append_history(history_entry)

        # Route based on entropy
        if entropy < threshold:
            # Low entropy -> Fractal Memory (clear, structured intent)
            self._route_to_fractal(user_message, hypotheses, entropy, strategic_direction)
            destination = "fractal_memory"
            logger.info(f"Routed to Fractal Memory: entropy={entropy:.3f} < threshold={threshold}")
        else:
            # High entropy -> Pattern Cache (ambiguous, uncertain)
            self._route_to_pattern_cache(
                user_message, hypotheses, entropy, strategic_direction, model, provider
            )
            destination = "pattern_cache"
            logger.info(f"Routed to Pattern Cache: entropy={entropy:.3f} >= threshold={threshold}")

        # Update ALS with routing decision
        reflection = (
            f"Latent routing: {destination} (entropy={entropy:.3f}, threshold={threshold}). "
            f"Strategic direction: {strategic_direction[:50]}..."
        )
        self.update_als(reflection=reflection)

    def _format_teaching_flow(
        self,
        timestamp: Any,
        user_message: str,
        hypotheses: list[dict[str, Any]],
        entropy: float,
        strategic_direction: str,
    ) -> str:
        """Format a teaching flow entry for MEMORY.md."""
        entry = f"\n## Teaching Flow - {timestamp.strftime('%Y-%m-%d %H:%M')}\n"
        entry += f"User: {user_message}\n\n"
        entry += f"**Latent Reasoning** (entropy={entropy:.3f}):\n"
        for i, hyp in enumerate(hypotheses, 1):
            entry += f"{i}. {hyp.get('intent', 'unknown')} (confidence={hyp.get('confidence', 0.0):.2f})\n"
            if reasoning := hyp.get('reasoning'):
                entry += f"   - {reasoning}\n"
        entry += f"\n**Strategic Direction**: {strategic_direction}\n"
        return entry

    def _route_to_fractal(
        self,
        user_message: str,
        hypotheses: list[dict[str, Any]],
        entropy: float,
        strategic_direction: str,
    ) -> None:
        """Route low-entropy state to Fractal Memory."""
        # Extract tags from top hypothesis
        tags = ["latent-reasoning", "low-entropy"]
        if hypotheses:
            top_intent = hypotheses[0].get("intent", "")
            # Simple tag extraction from intent
            tags.extend(top_intent.lower().split()[:3])

        # Create summary
        summary = f"Low-entropy latent state (ε={entropy:.3f}): {strategic_direction[:80]}"

        # Save as fractal node
        content = f"User message: {user_message}\n\n"
        content += f"Strategic direction: {strategic_direction}\n\n"
        content += "Hypotheses:\n"
        for hyp in hypotheses:
            content += f"- {hyp.get('intent', 'unknown')} ({hyp.get('confidence', 0.0):.2f})\n"

        self.save_fractal_node(
            content=content,
            tags=tags,
            summary=summary,
        )

    def _route_to_pattern_cache(
        self,
        user_message: str,
        hypotheses: list[dict[str, Any]],
        entropy: float,
        strategic_direction: str,
        model: str | None,
        provider: str | None,
    ) -> None:
        """Route high-entropy state to Pattern Cache."""
        from datetime import datetime

        record = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "hypotheses": hypotheses,
            "entropy": entropy,
            "strategic_direction": strategic_direction,
        }

        # Add model/provider metadata if available
        if model:
            record["model"] = model
        if provider:
            record["provider"] = provider

        # Append to pattern cache (JSONL format)
        with open(self.pattern_cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get_pattern_cache_entries(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Retrieve recent entries from pattern cache.

        Args:
            limit: Maximum number of entries to return (most recent first)

        Returns:
            List of pattern cache records
        """
        if not self.pattern_cache_file.exists():
            return []

        entries = []
        with open(self.pattern_cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON in pattern cache: {e}")

        # Return most recent first
        return entries[-limit:][::-1]

    # --- NEW: Hierarchical Navigation Methods ---

    def get_node_by_id(self, node_id: str) -> FractalNode | None:
        """
        Get a specific node by ID.

        Args:
            node_id: The node ID

        Returns:
            The FractalNode or None if not found
        """
        if self._mem0_provider:
            return self._mem0_provider.get_memory_by_id(node_id)

        archive_path = self.archives_dir / f"lesson_{node_id}.json"
        if archive_path.exists():
            try:
                return FractalNode.model_validate_json(
                    archive_path.read_text(encoding="utf-8")
                )
            except Exception as e:
                logger.error(f"Failed to load node {node_id}: {e}")
        return None

    def get_children(self, node_id: str) -> list[FractalNode]:
        """
        Get all child nodes of a given node.

        Args:
            node_id: Parent node ID

        Returns:
            List of child FractalNodes
        """
        parent = self.get_node_by_id(node_id)
        if not parent:
            return []

        children = []
        for child_id in parent.children_ids:
            child = self.get_node_by_id(child_id)
            if child:
                children.append(child)

        return children

    def get_parent(self, node_id: str) -> FractalNode | None:
        """
        Get the parent node of a given node.

        Args:
            node_id: Child node ID

        Returns:
            Parent FractalNode or None
        """
        node = self.get_node_by_id(node_id)
        if not node or not node.parent_id:
            return None

        return self.get_node_by_id(node.parent_id)

    def get_hierarchy_tree(self, root_id: str, max_depth: int = 3) -> dict:
        """
        Get a hierarchical tree structure starting from a root node.

        Args:
            root_id: Root node ID
            max_depth: Maximum depth to traverse

        Returns:
            Dictionary representing the tree structure
        """
        def build_tree(node_id: str, current_depth: int) -> dict | None:
            if current_depth > max_depth:
                return None

            node = self.get_node_by_id(node_id)
            if not node:
                return None

            tree = {
                "id": node.id,
                "summary": node.context_summary,
                "tags": node.tags,
                "content_type": node.content_type.value,
                "children": []
            }

            for child_id in node.children_ids:
                child_tree = build_tree(child_id, current_depth + 1)
                if child_tree:
                    tree["children"].append(child_tree)

            return tree

        return build_tree(root_id, 0) or {}

    # --- NEW: Multi-modal Support Methods ---

    def save_code_snippet(
        self,
        code: str,
        language: str,
        tags: list[str],
        summary: str,
        parent_id: str | None = None,
    ) -> FractalNode:
        """
        Save a code snippet as a fractal node.

        Args:
            code: The code content
            language: Programming language (python, javascript, etc.)
            tags: Tags for categorization
            summary: Brief summary
            parent_id: Optional parent node ID

        Returns:
            The created FractalNode
        """
        return self.save_fractal_node(
            content=code,
            tags=tags + [language, "code"],
            summary=summary,
            content_type=ContentType.CODE,
            language=language,
            parent_id=parent_id,
        )

    def save_image(
        self,
        image_path: str | Path,
        tags: list[str],
        summary: str,
        description: str = "",
        parent_id: str | None = None,
    ) -> FractalNode:
        """
        Save an image as a fractal node.

        Args:
            image_path: Path to the image file
            tags: Tags for categorization
            summary: Brief summary
            description: Optional text description of the image
            parent_id: Optional parent node ID

        Returns:
            The created FractalNode
        """
        image_path = Path(image_path)

        # Read and encode image
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()

            binary_data = base64.b64encode(image_data).decode("utf-8")

            # Detect MIME type
            mime_type, _ = mimetypes.guess_type(str(image_path))
            if not mime_type:
                mime_type = "image/png"  # Default

            return self.save_fractal_node(
                content=description or f"Image: {image_path.name}",
                tags=tags + ["image"],
                summary=summary,
                content_type=ContentType.IMAGE,
                binary_data=binary_data,
                mime_type=mime_type,
                parent_id=parent_id,
            )

        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            raise

    def get_image_data(self, node_id: str) -> tuple[bytes, str] | None:
        """
        Get decoded image data from a node.

        Args:
            node_id: The node ID

        Returns:
            Tuple of (image_bytes, mime_type) or None
        """
        node = self.get_node_by_id(node_id)
        if not node or node.content_type != ContentType.IMAGE:
            return None

        if not node.binary_data:
            return None

        try:
            image_bytes = base64.b64decode(node.binary_data)
            return (image_bytes, node.mime_type or "image/png")
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            return None

    def search_by_type(
        self,
        content_type: ContentType,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[FractalNode]:
        """
        Search nodes by content type.

        Args:
            content_type: The content type to filter by
            tags: Optional tags to filter by
            limit: Maximum number of results

        Returns:
            List of matching FractalNodes
        """
        if self._mem0_provider:
            filters = {"content_type": content_type}
            if tags:
                filters["tags"] = tags
            return self._mem0_provider.get_all_memories(filters=filters)[:limit]

        # Local search
        matches = []
        for archive_path in self.archives_dir.glob("lesson_*.json"):
            try:
                node = FractalNode.model_validate_json(
                    archive_path.read_text(encoding="utf-8")
                )

                # Check content type
                if node.content_type != content_type:
                    continue

                # Check tags if provided
                if tags and not any(tag in node.tags for tag in tags):
                    continue

                matches.append(node)

                if len(matches) >= limit:
                    break

            except Exception as e:
                logger.warning(f"Failed to load node from {archive_path}: {e}")

        return matches
