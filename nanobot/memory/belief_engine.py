"""NanobotBeliefEngine — deterministic, memory-only reasoning engine.

Stores facts as a directed graph with confidence scores and timestamps,
handles inverse relationships automatically, detects and resolves cycles
by evicting weakest facts, and computes superlatives deterministically.
"""

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional

from nanobot.memory.belief_ontology import (
    DEFAULT_ONTOLOGY,
    PredicateSpec,
    get_inverse,
    is_transitive,
)
from nanobot.memory.config_v2 import (
    LOGGER_NAME,
    SUPERLATIVE_POLICY,
    UNKNOWN_ENTITY_MESSAGE,
    UNKNOWN_PAIRWISE_MESSAGE,
    UNKNOWN_SELF_COMPARISON_MESSAGE,
    UNKNOWN_SUPERLATIVE_MESSAGE,
)
from nanobot.memory.types_v2 import RelationType, TruthValue

logger = logging.getLogger(LOGGER_NAME)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    """A single fact stored in the belief engine."""
    subject: str
    predicate: RelationType
    object: str
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)
    source: str = "api"
    is_inverse: bool = False

    @property
    def key(self) -> tuple[str, RelationType, str]:
        """Return the canonical key for this fact."""
        return (self.subject, self.predicate, self.object)


@dataclass
class BeliefQueryResult:
    """Result of a belief engine query."""
    value: TruthValue
    message: str
    source: str
    confidence: float
    path: Optional[list[str]] = None
    depth: int = 0
    cycle_detected: bool = False
    details: Optional[dict[str, Any]] = None


@dataclass
class CycleResolutionResult:
    """Result of cycle detection and resolution."""
    resolved: bool
    evicted_fact: Optional["Fact"] = None
    message: str = ""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class NanobotBeliefEngine:
    """Fully deterministic, memory-only reasoning engine.

    Stores facts as a directed graph with confidence scores and timestamps.
    Automatically handles inverse relationships, detects and resolves cycles
    by evicting the weakest fact, and computes superlatives deterministically.

    Attributes:
        ontology: Predicate specifications (default: DEFAULT_ONTOLOGY)
        superlative_policy: Policy for ambiguous superlatives
        facts: Dict keyed by (subject, predicate, object) -> Fact
        entities: Set of all known entities
        adjacency: Adjacency list: (entity, predicate) -> set of targets
    """

    def __init__(
        self,
        ontology: Optional[dict[RelationType, PredicateSpec]] = None,
        superlative_policy: str = "unknown_if_not_unique",
    ) -> None:
        self.ontology: dict[RelationType, PredicateSpec] = (
            ontology if ontology is not None else DEFAULT_ONTOLOGY
        )
        self.superlative_policy = superlative_policy
        self.facts: dict[tuple[str, RelationType, str], Fact] = {}
        self.entities: set[str] = set()
        self.adjacency: defaultdict[tuple[str, RelationType], set[str]] = defaultdict(set)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_fact(
        self,
        subject: str,
        predicate: RelationType,
        obj: str,
        confidence: float = 0.5,
        source: str = "api",
    ) -> Optional[CycleResolutionResult]:
        """Add a fact and its inverse, then check for cycles.

        Args:
            subject: Subject entity
            predicate: Relation type
            obj: Object entity
            confidence: Confidence score, clamped to [0, 1]
            source: Source identifier

        Returns:
            CycleResolutionResult if a cycle was detected/resolved, else None
        """
        confidence = max(0.0, min(1.0, confidence))
        ts = time.time()
        fact = Fact(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            timestamp=ts,
            source=source,
            is_inverse=False,
        )
        self._store(fact)

        # Store inverse
        inv_pred = get_inverse(predicate, self.ontology)
        if inv_pred is not None and inv_pred != predicate:
            inv_fact = Fact(
                subject=obj,
                predicate=inv_pred,
                object=subject,
                confidence=confidence,
                timestamp=ts,
                source=source,
                is_inverse=True,
            )
            self._store(inv_fact)
        elif inv_pred == predicate:
            # Symmetric relation (e.g. RELATED_TO) — already stored as direct
            pass

        return self._detect_and_resolve_cycle(predicate, fact)

    def delete_fact(
        self, subject: str, predicate: RelationType, obj: str
    ) -> bool:
        """Delete a fact and its inverse.

        Args:
            subject: Subject entity
            predicate: Relation type
            obj: Object entity

        Returns:
            True if the fact existed and was removed, False otherwise
        """
        key = (subject, predicate, obj)
        if key not in self.facts:
            return False

        # Remove direct
        del self.facts[key]
        self.adjacency[(subject, predicate)].discard(obj)

        # Remove inverse
        inv_pred = get_inverse(predicate, self.ontology)
        if inv_pred is not None:
            inv_key = (obj, inv_pred, subject)
            self.facts.pop(inv_key, None)
            self.adjacency[(obj, inv_pred)].discard(subject)

        logger.debug("Deleted fact: %s %s %s (and inverse)", subject, predicate, obj)
        return True

    def query(
        self,
        subject: str,
        predicate: RelationType,
        obj: Optional[str] = None,
    ) -> "BeliefQueryResult | list[str]":
        """Query the belief engine.

        Args:
            subject: Subject entity
            predicate: Relation type
            obj: Object entity (if None, returns all targets)

        Returns:
            BeliefQueryResult for pairwise queries, list[str] for target queries
        """
        if obj is not None:
            return self._query_pairwise(subject, predicate, obj)
        return self._query_targets(subject, predicate)

    def query_superlative(
        self,
        attribute: str,
        qualifier: str = "most",
    ) -> BeliefQueryResult:
        """Query for the superlative entity by an attribute.

        Args:
            attribute: Attribute name (e.g. "tall", "fast")
            qualifier: "most" or "least"

        Returns:
            BeliefQueryResult with the winning entity or UNKNOWN
        """
        rel = self._resolve_superlative_relation(attribute, qualifier)
        if rel is None:
            return BeliefQueryResult(
                value=TruthValue.UNKNOWN,
                message=UNKNOWN_SUPERLATIVE_MESSAGE,
                source="superlative",
                confidence=0.0,
            )

        if self._has_cycle(rel):
            return BeliefQueryResult(
                value=TruthValue.UNKNOWN,
                message="UNKNOWN: Cycle detected in relation graph",
                source="superlative",
                confidence=0.0,
                cycle_detected=True,
            )

        tiers = self._compute_tiers(rel)
        if not tiers:
            return BeliefQueryResult(
                value=TruthValue.UNKNOWN,
                message=UNKNOWN_SUPERLATIVE_MESSAGE,
                source="superlative",
                confidence=0.0,
            )

        # Tier-0 entities are "beaten by nobody" — they are the top entities
        top_tier = tiers[0]

        if len(top_tier) == 1:
            (winner,) = top_tier
            conf = self._chain_confidence(winner, rel)
            return BeliefQueryResult(
                value=TruthValue.TRUE,
                message=f"{winner} is the {qualifier} {attribute}",
                source="superlative",
                confidence=conf,
                details={"winner": winner, "tiers": tiers},
            )

        # Multiple candidates
        if self.superlative_policy == "unknown_if_not_unique":
            return BeliefQueryResult(
                value=TruthValue.UNKNOWN,
                message=UNKNOWN_SUPERLATIVE_MESSAGE,
                source="superlative",
                confidence=0.0,
                details={"candidates": top_tier, "tiers": tiers},
            )

        # Fallback: pick by highest chain confidence
        best = max(top_tier, key=lambda e: self._chain_confidence(e, rel))
        return BeliefQueryResult(
            value=TruthValue.TRUE,
            message=f"{best} is the {qualifier} {attribute} (tiebreak by confidence)",
            source="superlative",
            confidence=self._chain_confidence(best, rel),
            details={"winner": best, "tiers": tiers},
        )

    def query_attribute(
        self, entity: str, attribute: str
    ) -> "BeliefQueryResult | list[str]":
        """Query all entities related to the given entity via an attribute mapping.

        Args:
            entity: Entity to query about
            attribute: Attribute name (e.g. "height", "speed")

        Returns:
            BeliefQueryResult or list of related entities
        """
        rel = self._attribute_to_relation(attribute)
        if rel is None:
            return BeliefQueryResult(
                value=TruthValue.UNKNOWN,
                message=UNKNOWN_PAIRWISE_MESSAGE,
                source="attribute",
                confidence=0.0,
            )
        return self._query_targets(entity, rel)

    def generate_hypotheses(
        self,
        query_text: str,
        max_hypotheses: int = 3,
    ) -> dict[str, Any]:
        """Generate hypotheses from stored facts without LLM (keyword-based dispatch).

        Returns dict compatible with HypothesisEngine format.
        """
        query_lower = query_text.lower()
        hypotheses: list[dict[str, Any]] = []
        entities_found: list[str] = []
        query_type = "unknown"

        # Detect query type
        superlative_keywords = {
            "tallest": ("tall", "most"),
            "shortest": ("tall", "least"),
            "fastest": ("fast", "most"),
            "slowest": ("fast", "least"),
            "greatest": ("great", "most"),
            "least": ("great", "least"),
        }
        for kw, (attr, qual) in superlative_keywords.items():
            if kw in query_lower:
                query_type = "superlative"
                result = self.query_superlative(attr, qual)
                if result.value == TruthValue.TRUE and result.details:
                    winner = result.details.get("winner", "")
                    entities_found = [winner] if winner else []
                    hypotheses.append({
                        "intent": f"{qual} {attr}",
                        "confidence": result.confidence,
                        "reasoning": result.message,
                        "result": winner,
                        "evidence": result.details,
                    })
                break

        # Fallback: entity-based hypotheses from adjacency
        if not hypotheses:
            for entity in list(self.entities)[:max_hypotheses]:
                facts_for = [
                    f for f in self.facts.values()
                    if f.subject == entity and not f.is_inverse
                ]
                if facts_for:
                    best_fact = max(facts_for, key=lambda x: x.confidence)
                    hypotheses.append({
                        "intent": f"{entity} {best_fact.predicate.value} {best_fact.object}",
                        "confidence": best_fact.confidence,
                        "reasoning": f"Stored fact: {entity} {best_fact.predicate.value} {best_fact.object}",
                        "result": best_fact.object,
                        "evidence": {"subject": entity, "predicate": best_fact.predicate.value, "object": best_fact.object},
                    })
                    if entity not in entities_found:
                        entities_found.append(entity)

        hypotheses = hypotheses[:max_hypotheses]
        entropy = self._compute_entropy(hypotheses)

        return {
            "hypotheses": hypotheses,
            "entropy": entropy,
            "requires_llm": entropy > 0.8,
            "query_type": query_type,
            "entities_found": entities_found,
            "facts_stored": self.get_fact_count(),
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_all_facts(self) -> list[Fact]:
        """Return all stored facts."""
        return list(self.facts.values())

    def get_entity_count(self) -> int:
        """Return number of known entities."""
        return len(self.entities)

    def get_fact_count(self) -> int:
        """Return number of stored facts (including inverses)."""
        return len(self.facts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _store(self, fact: Fact) -> None:
        """Store a fact; keep higher confidence on conflict, tiebreak by recency."""
        key = fact.key
        existing = self.facts.get(key)
        if existing is not None:
            if fact.confidence > existing.confidence:
                self.facts[key] = fact
            elif fact.confidence == existing.confidence and fact.timestamp > existing.timestamp:
                self.facts[key] = fact
            # else keep existing
        else:
            self.facts[key] = fact

        self.entities.add(fact.subject)
        self.entities.add(fact.object)
        self.adjacency[(fact.subject, fact.predicate)].add(fact.object)

    def _detect_and_resolve_cycle(
        self, predicate: RelationType, new_fact: Fact
    ) -> Optional[CycleResolutionResult]:
        """Detect and resolve cycles after adding a new fact.

        Only checks transitive predicates. On cycle: evict weakest fact.
        """
        if not is_transitive(predicate, self.ontology):
            return None

        # BFS from new_fact.object back to new_fact.subject
        path = self._find_path_bfs(new_fact.object, new_fact.subject, predicate)
        if path is None:
            return None  # No cycle

        # Cycle found: find weakest fact on the cycle path
        cycle_path = [new_fact.subject, new_fact.object] + path  # full cycle
        facts_on_cycle = self._get_facts_on_path(cycle_path, predicate)
        if not facts_on_cycle:
            return CycleResolutionResult(resolved=False, message="Cycle detected but no facts found on path")

        weakest = min(facts_on_cycle, key=lambda f: (f.confidence, f.timestamp))

        if weakest.key == new_fact.key:
            # New fact is the weakest — reject it
            self.delete_fact(new_fact.subject, new_fact.predicate, new_fact.object)
            return CycleResolutionResult(
                resolved=True,
                evicted_fact=weakest,
                message=f"Cycle resolved: new fact rejected (confidence {weakest.confidence:.3f})",
            )
        else:
            # Evict the weakest existing fact
            self.delete_fact(weakest.subject, weakest.predicate, weakest.object)
            return CycleResolutionResult(
                resolved=True,
                evicted_fact=weakest,
                message=f"Cycle resolved: evicted {weakest.subject}->{weakest.object} (confidence {weakest.confidence:.3f})",
            )

    def _has_cycle(self, predicate: RelationType) -> bool:
        """DFS-based cycle detection for a given predicate."""
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in self.adjacency.get((node, predicate), set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.discard(node)
            return False

        for entity in list(self.entities):
            if entity not in visited:
                if dfs(entity):
                    return True
        return False

    def _get_facts_on_path(
        self, path: list[str], predicate: RelationType
    ) -> list[Fact]:
        """Return facts along a path for a given predicate."""
        result = []
        for i in range(len(path) - 1):
            key = (path[i], predicate, path[i + 1])
            f = self.facts.get(key)
            if f is not None:
                result.append(f)
        return result

    def _find_path_bfs(
        self,
        start: str,
        target: str,
        predicate: RelationType,
        max_depth: int = 100,
    ) -> Optional[list[str]]:
        """BFS returning path from start to target (not including start), or None."""
        if start == target:
            return []
        queue: deque[tuple[str, list[str]]] = deque([(start, [])])
        visited: set[str] = {start}
        while queue:
            node, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for neighbor in self.adjacency.get((node, predicate), set()):
                if neighbor == target:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def _find_path_metadata(
        self,
        start: str,
        target: str,
        predicate: RelationType,
    ) -> tuple[Optional[list[str]], float]:
        """BFS returning (path, chain_confidence) where chain_confidence = product of edge confidences."""
        path = self._find_path_bfs(start, target, predicate)
        if path is None:
            return None, 0.0
        full_path = [start] + path
        conf = 1.0
        for i in range(len(full_path) - 1):
            key = (full_path[i], predicate, full_path[i + 1])
            f = self.facts.get(key)
            if f is not None:
                conf *= f.confidence
        return path, conf

    def _query_pairwise(
        self, subject: str, predicate: RelationType, obj: str
    ) -> BeliefQueryResult:
        """Query whether subject predicate obj holds."""
        # Self-comparison
        if subject == obj:
            return BeliefQueryResult(
                value=TruthValue.UNKNOWN,
                message=UNKNOWN_SELF_COMPARISON_MESSAGE,
                source="self_comparison",
                confidence=0.0,
            )

        # Unknown entity
        if subject not in self.entities or obj not in self.entities:
            return BeliefQueryResult(
                value=TruthValue.UNKNOWN,
                message=UNKNOWN_ENTITY_MESSAGE,
                source="unknown_entity",
                confidence=0.0,
            )

        # Direct fact
        key = (subject, predicate, obj)
        if key in self.facts:
            f = self.facts[key]
            return BeliefQueryResult(
                value=TruthValue.TRUE,
                message=f"Direct fact: {subject} {predicate.value} {obj}",
                source="direct",
                confidence=f.confidence,
                path=[subject, obj],
                depth=1,
            )

        # Transitive inference
        if is_transitive(predicate, self.ontology):
            path, chain_conf = self._find_path_metadata(subject, obj, predicate)
            if path is not None:
                full_path = [subject] + path
                return BeliefQueryResult(
                    value=TruthValue.TRUE,
                    message=f"Transitive inference: {subject} {predicate.value} {obj}",
                    source="transitive",
                    confidence=chain_conf,
                    path=full_path,
                    depth=len(path),
                )

        # Opposite provable → FALSE
        inv_pred = get_inverse(predicate, self.ontology)
        if inv_pred is not None and inv_pred != predicate:
            inv_key = (subject, inv_pred, obj)
            if inv_key in self.facts:
                return BeliefQueryResult(
                    value=TruthValue.FALSE,
                    message=f"Opposite relation holds: {subject} {inv_pred.value} {obj}",
                    source="opposite",
                    confidence=self.facts[inv_key].confidence,
                )
            # Transitive opposite
            if is_transitive(inv_pred, self.ontology):
                opp_path, opp_conf = self._find_path_metadata(subject, obj, inv_pred)
                if opp_path is not None:
                    return BeliefQueryResult(
                        value=TruthValue.FALSE,
                        message=f"Opposite relation transitively holds",
                        source="opposite_transitive",
                        confidence=opp_conf,
                    )

        return BeliefQueryResult(
            value=TruthValue.UNKNOWN,
            message=UNKNOWN_PAIRWISE_MESSAGE,
            source="unknown",
            confidence=0.0,
        )

    def _query_targets(
        self, subject: str, predicate: RelationType
    ) -> list[str]:
        """Return all entities reachable from subject via predicate."""
        if is_transitive(predicate, self.ontology):
            # BFS for all reachable
            visited: set[str] = set()
            queue: deque[str] = deque([subject])
            while queue:
                node = queue.popleft()
                for neighbor in self.adjacency.get((node, predicate), set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            return list(visited)
        else:
            return list(self.adjacency.get((subject, predicate), set()))

    def _compute_tiers(self, predicate: RelationType) -> list[list[str]]:
        """Kahn's algorithm to compute topological tiers.

        Returns list of tiers from "top" (beaten by nobody) to "bottom".
        For TALLER_THAN: Alice>Bob>Carol, tier-0=[Alice], tier-1=[Bob], tier-2=[Carol]
        """
        # Build in-degree count: how many entities beat this entity
        in_degree: dict[str, int] = {}
        for entity in self.entities:
            in_degree[entity] = 0

        for (subj, pred), targets in self.adjacency.items():
            if pred == predicate:
                for target in targets:
                    in_degree[target] = in_degree.get(target, 0) + 1

        # Only consider entities that appear in this predicate's graph
        relevant: set[str] = set()
        for (subj, pred) in self.adjacency:
            if pred == predicate:
                relevant.add(subj)
                relevant.update(self.adjacency[(subj, pred)])

        if not relevant:
            return []

        tiers: list[list[str]] = []
        remaining = {e: in_degree.get(e, 0) for e in relevant}
        while remaining:
            tier = sorted(e for e, d in remaining.items() if d == 0)
            if not tier:
                break  # Cycle present, stop
            tiers.append(tier)
            for e in tier:
                del remaining[e]
            for (subj, pred), targets in self.adjacency.items():
                if pred == predicate and subj in tier:
                    for t in targets:
                        if t in remaining:
                            remaining[t] -= 1

        return tiers

    def _chain_confidence(self, entity: str, predicate: RelationType) -> float:
        """Return the minimum confidence of outgoing edges for the entity."""
        targets = self.adjacency.get((entity, predicate), set())
        if not targets:
            return 1.0
        confs = []
        for t in targets:
            key = (entity, predicate, t)
            f = self.facts.get(key)
            if f:
                confs.append(f.confidence)
        return min(confs) if confs else 1.0

    def _resolve_superlative_relation(
        self, attribute: str, qualifier: str
    ) -> Optional[RelationType]:
        """Map (attribute, qualifier) -> RelationType."""
        attr = attribute.lower().strip()
        qual = qualifier.lower().strip()

        mapping: dict[tuple[str, str], RelationType] = {
            ("tall", "most"): RelationType.TALLER_THAN,
            ("tall", "least"): RelationType.SHORTER_THAN,
            ("short", "most"): RelationType.SHORTER_THAN,
            ("short", "least"): RelationType.TALLER_THAN,
            ("fast", "most"): RelationType.FASTER_THAN,
            ("fast", "least"): RelationType.SLOWER_THAN,
            ("slow", "most"): RelationType.SLOWER_THAN,
            ("slow", "least"): RelationType.FASTER_THAN,
            ("great", "most"): RelationType.GREATER_THAN,
            ("great", "least"): RelationType.LESS_THAN,
            ("big", "most"): RelationType.GREATER_THAN,
            ("big", "least"): RelationType.LESS_THAN,
            ("small", "most"): RelationType.LESS_THAN,
            ("small", "least"): RelationType.GREATER_THAN,
        }
        return mapping.get((attr, qual))

    def _attribute_to_relation(self, attribute: str) -> Optional[RelationType]:
        """Map attribute name to RelationType."""
        attr = attribute.lower().strip()
        mapping: dict[str, RelationType] = {
            "height": RelationType.TALLER_THAN,
            "tall": RelationType.TALLER_THAN,
            "speed": RelationType.FASTER_THAN,
            "fast": RelationType.FASTER_THAN,
            "size": RelationType.GREATER_THAN,
            "great": RelationType.GREATER_THAN,
            "dependency": RelationType.DEPENDS_ON,
        }
        return mapping.get(attr)

    def _compute_entropy(self, hypotheses: list[dict[str, Any]]) -> float:
        """Normalized Shannon entropy matching HypothesisEngine implementation."""
        if not hypotheses:
            return 1.0
        confs = [h.get("confidence", 0.0) for h in hypotheses]
        total = sum(confs)
        if total == 0.0:
            return 1.0
        probs = [c / total for c in confs]
        raw = -sum(p * math.log2(p) for p in probs if p > 0)
        max_entropy = math.log2(len(hypotheses)) if len(hypotheses) > 1 else 1.0
        return raw / max_entropy if max_entropy > 0 else 0.0
