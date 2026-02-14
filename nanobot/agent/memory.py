"""Memory system for persistent agent memory."""

import base64
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any

from nanobot.agent.memory_types import ActiveLearningState, ContentType, FractalNode
from nanobot.utils.helpers import ensure_dir

logger = logging.getLogger(__name__)

# Constants
MAX_CONTENT_PREVIEW_LENGTH = 200  # Maximum length for content preview in retrieval


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
        
        # --- NEW: Fractal Architecture ---
        self.archives_dir = ensure_dir(self.memory_dir / "archives")
        self.index_file = self.memory_dir / "fractal_index.json"
        self.als_file = self.memory_dir / "ALS.json"
        
        # mem0 provider (optional)
        self._mem0_provider = None
        self._init_provider()
        
        self._init_fractal_structure()
    
    def _init_provider(self) -> None:
        """Initialize memory provider based on configuration."""
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
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

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
        entangled_with: list[str] | None = None,
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
            entangled_with: Optional node IDs to entangle with this node
        
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
                
                # Update entanglement links
                if entangled_with:
                    for ent_id in entangled_with:
                        node.entangled_ids[ent_id] = 1.0
                
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
        
        if entangled_with:
            for ent_id in entangled_with:
                node.entangled_ids[ent_id] = 1.0
                entangled_node = self.get_node_by_id(ent_id)
                if entangled_node:
                    entangled_node.entangled_ids[node.id] = 1.0
                    self._update_node(entangled_node)
        
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
    
    def retrieve_relevant_nodes(self, query: str, k: int = 5) -> str:
        """
        Token-efficient retrieval using semantic search (mem0) or keyword matching (local).
        
        Args:
            query: Search query
            k: Number of top results to return
        
        Returns:
            Formatted string with relevant memory nodes
        """
        # Use mem0 provider if available
        if self._mem0_provider:
            try:
                nodes = self._mem0_provider.search_memories(query, k=k)
                base_scores = {
                    node.id: (len(nodes) - idx) / max(len(nodes), 1)
                    for idx, node in enumerate(nodes)
                }
                expanded_nodes = self.get_entangled_context(nodes)
                ranked = self._rerank_nodes(expanded_nodes, nodes, base_scores)
                return self._format_nodes(ranked)
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
            base_scores = {}
            for score, entry in top_nodes:
                if score == 0:
                    continue  # Skip irrelevant nodes
                base_scores[entry["id"]] = score
                
                archive_path = self.archives_dir / f"lesson_{entry['id']}.json"
                if archive_path.exists():
                    try:
                        node = FractalNode.model_validate_json(
                            archive_path.read_text(encoding="utf-8")
                        )
                        nodes.append(node)
                    except Exception as e:
                        logger.warning(f"Could not load node {entry['id']}: {e}")
            
            expanded_nodes = self.get_entangled_context(nodes)
            ranked = self._rerank_nodes(expanded_nodes, nodes, base_scores)
            return self._format_nodes(ranked)
            
        except Exception as e:
            logger.error(f"Error retrieving nodes: {e}")
            return ""
    
    def get_entangled_context(self, retrieved_nodes: list[FractalNode]) -> list[FractalNode]:
        """Expand retrieval set with strongly entangled nodes."""
        found_ids = {n.id for n in retrieved_nodes}
        entangled_pool: list[FractalNode] = []
        
        for node in retrieved_nodes:
            for ent_id, strength in node.entangled_ids.items():
                if ent_id not in found_ids and strength > 0.5:
                    entangled_node = self.get_node_by_id(ent_id)
                    if entangled_node:
                        entangled_pool.append(entangled_node)
                        found_ids.add(ent_id)
        
        return retrieved_nodes + entangled_pool
    
    def _rerank_nodes(
        self,
        nodes: list[FractalNode],
        base_nodes: list[FractalNode],
        base_scores: dict[str, float],
    ) -> list[FractalNode]:
        """Re-rank nodes with vector/keyword and entanglement blend."""
        if not nodes:
            return nodes
        
        max_score = max(base_scores.values()) if base_scores else 1.0
        base_ids = {n.id for n in base_nodes}
        
        def score_node(node: FractalNode) -> float:
            base_similarity = base_scores.get(node.id, 1.0 if node.id in base_ids else 0.0)
            vector_similarity = base_similarity / max_score
            
            entanglement_strength = 0.0
            for source in base_nodes:
                entanglement_strength = max(
                    entanglement_strength,
                    source.entangled_ids.get(node.id, 0.0),
                )
            
            return (vector_similarity * 0.7) + (entanglement_strength * 0.3)
        
        return sorted(nodes, key=score_node, reverse=True)
    
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
