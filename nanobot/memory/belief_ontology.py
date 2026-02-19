"""Ontology configuration for NanobotBeliefEngine predicate definitions."""

from dataclasses import dataclass
from typing import Optional

from nanobot.memory.types_v2 import RelationType


@dataclass(frozen=True)
class PredicateSpec:
    """Specification for a predicate in the belief ontology."""
    relation_type: RelationType
    inverse: Optional[RelationType]
    transitive: bool
    label: str


DEFAULT_ONTOLOGY: dict[RelationType, PredicateSpec] = {
    RelationType.TALLER_THAN: PredicateSpec(
        relation_type=RelationType.TALLER_THAN,
        inverse=RelationType.SHORTER_THAN,
        transitive=True,
        label="taller than",
    ),
    RelationType.SHORTER_THAN: PredicateSpec(
        relation_type=RelationType.SHORTER_THAN,
        inverse=RelationType.TALLER_THAN,
        transitive=True,
        label="shorter than",
    ),
    RelationType.FASTER_THAN: PredicateSpec(
        relation_type=RelationType.FASTER_THAN,
        inverse=RelationType.SLOWER_THAN,
        transitive=True,
        label="faster than",
    ),
    RelationType.SLOWER_THAN: PredicateSpec(
        relation_type=RelationType.SLOWER_THAN,
        inverse=RelationType.FASTER_THAN,
        transitive=True,
        label="slower than",
    ),
    RelationType.GREATER_THAN: PredicateSpec(
        relation_type=RelationType.GREATER_THAN,
        inverse=RelationType.LESS_THAN,
        transitive=True,
        label="greater than",
    ),
    RelationType.LESS_THAN: PredicateSpec(
        relation_type=RelationType.LESS_THAN,
        inverse=RelationType.GREATER_THAN,
        transitive=True,
        label="less than",
    ),
    RelationType.DEPENDS_ON: PredicateSpec(
        relation_type=RelationType.DEPENDS_ON,
        inverse=RelationType.DEPENDENCY_OF,
        transitive=True,
        label="depends on",
    ),
    RelationType.DEPENDENCY_OF: PredicateSpec(
        relation_type=RelationType.DEPENDENCY_OF,
        inverse=RelationType.DEPENDS_ON,
        transitive=True,
        label="dependency of",
    ),
    RelationType.IMPACTS: PredicateSpec(
        relation_type=RelationType.IMPACTS,
        inverse=RelationType.IMPACTED_BY,
        transitive=True,
        label="impacts",
    ),
    RelationType.IMPACTED_BY: PredicateSpec(
        relation_type=RelationType.IMPACTED_BY,
        inverse=RelationType.IMPACTS,
        transitive=True,
        label="impacted by",
    ),
    RelationType.CONTAINS: PredicateSpec(
        relation_type=RelationType.CONTAINS,
        inverse=RelationType.CONTAINED_IN,
        transitive=True,
        label="contains",
    ),
    RelationType.CONTAINED_IN: PredicateSpec(
        relation_type=RelationType.CONTAINED_IN,
        inverse=RelationType.CONTAINS,
        transitive=True,
        label="contained in",
    ),
    RelationType.SUPPLIES: PredicateSpec(
        relation_type=RelationType.SUPPLIES,
        inverse=RelationType.SUPPLIED_BY,
        transitive=False,
        label="supplies",
    ),
    RelationType.SUPPLIED_BY: PredicateSpec(
        relation_type=RelationType.SUPPLIED_BY,
        inverse=RelationType.SUPPLIES,
        transitive=False,
        label="supplied by",
    ),
    RelationType.RELATED_TO: PredicateSpec(
        relation_type=RelationType.RELATED_TO,
        inverse=RelationType.RELATED_TO,
        transitive=False,
        label="related to",
    ),
}


def get_inverse(
    relation_type: RelationType,
    ontology: Optional[dict[RelationType, PredicateSpec]] = None,
) -> Optional[RelationType]:
    """Return the inverse of a relation type according to the ontology."""
    o = ontology if ontology is not None else DEFAULT_ONTOLOGY
    spec = o.get(relation_type)
    return spec.inverse if spec else None


def is_transitive(
    relation_type: RelationType,
    ontology: Optional[dict[RelationType, PredicateSpec]] = None,
) -> bool:
    """Return whether a relation type is transitive according to the ontology."""
    o = ontology if ontology is not None else DEFAULT_ONTOLOGY
    spec = o.get(relation_type)
    return spec.transitive if spec else False
