# NanobotBeliefEngine

A fully deterministic, memory-only reasoning engine for Nanobot. It stores facts as a directed graph with confidence scores and timestamps, handles inverse relationships automatically, detects and resolves cycles by evicting the weakest fact, and computes superlatives deterministically without any LLM calls.

## Architecture

```
NanobotBeliefEngine
├── facts: dict[(subject, predicate, object) → Fact]
├── entities: set[str]
├── adjacency: dict[(entity, predicate) → set[entity]]
└── ontology: dict[RelationType → PredicateSpec]
```

### Key data classes

| Class | Purpose |
|---|---|
| `Fact` | A single stored fact with subject, predicate, object, confidence, timestamp, source |
| `BeliefQueryResult` | Result of a pairwise or superlative query (value, message, source, confidence, path) |
| `CycleResolutionResult` | Outcome of cycle detection after `add_fact` |
| `PredicateSpec` | Ontology entry: inverse relation, transitivity flag, human label |

## Supported relations

All relations defined in `RelationType` (from `nanobot.memory.types_v2`) are supported. Each has an entry in `DEFAULT_ONTOLOGY` specifying:

- **inverse** — automatically stored when the direct fact is added
- **transitive** — whether BFS-based transitive inference applies
- **label** — human-readable label

| Relation | Inverse | Transitive |
|---|---|---|
| `TALLER_THAN` | `SHORTER_THAN` | ✓ |
| `FASTER_THAN` | `SLOWER_THAN` | ✓ |
| `GREATER_THAN` | `LESS_THAN` | ✓ |
| `DEPENDS_ON` | `DEPENDENCY_OF` | ✓ |
| `IMPACTS` | `IMPACTED_BY` | ✓ |
| `CONTAINS` | `CONTAINED_IN` | ✓ |
| `SUPPLIES` | `SUPPLIED_BY` | ✗ |
| `RELATED_TO` | `RELATED_TO` (symmetric) | ✗ |

## Usage

### Adding facts

```python
from nanobot.memory.belief_engine import NanobotBeliefEngine
from nanobot.memory.types_v2 import RelationType

engine = NanobotBeliefEngine()
engine.add_fact("Alice", RelationType.TALLER_THAN, "Bob", confidence=0.9)
engine.add_fact("Bob", RelationType.TALLER_THAN, "Carol", confidence=0.8)
```

`add_fact` automatically stores the inverse (`Bob SHORTER_THAN Alice`) and checks for cycles. Confidence is clamped to `[0, 1]`.

### Pairwise queries

```python
result = engine.query("Alice", RelationType.TALLER_THAN, "Carol")
# TruthValue.TRUE (transitive inference: Alice > Bob > Carol)
# result.confidence ≈ 0.72  (product of edge confidences: 0.9 × 0.8)
# result.source == "transitive"
# result.path == ["Alice", "Bob", "Carol"]
```

Possible `TruthValue` outcomes:

| Value | Condition |
|---|---|
| `TRUE` | Direct fact or transitive inference found |
| `FALSE` | Opposite relation holds (direct or transitive) |
| `UNKNOWN` | No relation known, self-comparison, or unknown entity |

### Target queries

```python
targets = engine.query("Alice", RelationType.TALLER_THAN)
# ["Bob", "Carol"]  — all transitively reachable targets
```

### Superlative queries

```python
result = engine.query_superlative("tall", "most")
# TruthValue.TRUE, winner="Alice"

result = engine.query_superlative("tall", "least")
# TruthValue.TRUE, winner="Carol"
```

Supported `(attribute, qualifier)` mappings:

| attribute | most | least |
|---|---|---|
| `tall` / `short` | `TALLER_THAN` / `SHORTER_THAN` | `SHORTER_THAN` / `TALLER_THAN` |
| `fast` / `slow` | `FASTER_THAN` / `SLOWER_THAN` | `SLOWER_THAN` / `FASTER_THAN` |
| `great` / `big` / `small` | `GREATER_THAN` / `GREATER_THAN` / `LESS_THAN` | `LESS_THAN` / `LESS_THAN` / `GREATER_THAN` |

Superlatives use **Kahn's topological tiers algorithm**. If the top tier contains multiple entities and `superlative_policy="unknown_if_not_unique"` (default), the result is `UNKNOWN`. Use `superlative_policy="tiebreak_by_confidence"` to pick the highest-confidence entity.

### Hypothesis generation

```python
output = engine.generate_hypotheses("Who is the tallest?")
# {
#   "hypotheses": [{"intent": "most tall", "confidence": 0.9, "result": "Alice", ...}],
#   "entropy": 0.0,
#   "requires_llm": False,
#   "query_type": "superlative",
#   "entities_found": ["Alice"],
#   "facts_stored": 4,
# }
```

This is a pure keyword-dispatch function — no LLM required. Output is compatible with `HypothesisEngine` format.

## Cycle detection and resolution

After every `add_fact` for a **transitive** predicate, the engine performs a BFS to check if the new edge creates a cycle. If a cycle is detected:

1. All facts on the cycle path are collected.
2. The **weakest fact** (lowest confidence; timestamp used as tiebreaker) is evicted.
3. A `CycleResolutionResult` is returned describing the outcome.

```python
result = engine.add_fact("Carol", RelationType.TALLER_THAN, "Alice", confidence=0.3)
if result is not None:
    print(result.message)
    # "Cycle resolved: new fact rejected (confidence 0.300)"
```

## Toggle integration

The belief engine is controlled via the `belief_engine_enabled` toggle on `RuntimeState`:

```bash
nanobot reasoning belief-engine on
nanobot reasoning belief-engine off
nanobot reasoning status
```

The toggle participates in baseline mode: it is saved/restored when entering/exiting baseline mode, and is reset to `False` during baseline.

## Custom ontology

Pass a custom `ontology` dict to override the default predicate specifications:

```python
from nanobot.memory.belief_ontology import PredicateSpec
from nanobot.memory.types_v2 import RelationType

my_ontology = {
    RelationType.TALLER_THAN: PredicateSpec(
        relation_type=RelationType.TALLER_THAN,
        inverse=RelationType.SHORTER_THAN,
        transitive=False,  # disable transitivity
        label="taller than",
    ),
}
engine = NanobotBeliefEngine(ontology=my_ontology)
```

## Thread safety

`NanobotBeliefEngine` itself is **not** thread-safe (no internal locks). If concurrent access is needed, synchronize externally. `RuntimeState.belief_engine_enabled` uses the shared `RuntimeState` lock and is thread-safe.
