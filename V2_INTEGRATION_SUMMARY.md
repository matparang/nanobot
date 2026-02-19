# Nanobot V2 Integration Summary

This document summarizes the integration of Nanobot v2 deterministic, memory-first reasoning alongside the existing v1 modules.

## Overview

The v2 memory system provides deterministic, graph-based relational reasoning without LLM fallback. It operates side-by-side with the existing v1 hypothesis engine, selectable via configuration flag.

## Configuration

Add to your `config.json` in the `memory` section:

```json
{
  "memory": {
    "use_memory_v2": false,
    "confidence_threshold": 0.9,
    "enable_llm_fallback": true
  }
}
```

- `use_memory_v2`: Enable v2 deterministic reasoning (default: `false`)
- `confidence_threshold`: Minimum confidence for relation extraction (default: `0.9`)
- `enable_llm_fallback`: Allow LLM fallback when memory cannot answer (default: `true`)

### LLM-Disable Mode

Control LLM usage globally with CLI flags or configuration:

**CLI Flags** (override config):
```bash
# Disable all LLM calls (memory-only mode)
nanobot agent --disable-llm

# Force enable LLM calls
nanobot agent --enable-llm

# Gateway also supports these flags
nanobot gateway --disable-llm
```

**Config Setting**:
```json
{
  "memory": {
    "enable_llm_fallback": false  // Disable LLM fallback globally
  }
}
```

**When LLM is disabled:**
- No LLM provider initialization or API calls
- Memory-first reasoning only
- Returns deterministic UNKNOWN when memory cannot answer
- Useful for offline operation, cost control, or privacy

## Architecture

### V1 Mode (Default)
```
User Query → MemoryAwareReasoner → HypothesisEngine → RelationalCache (v1)
                                 ↓
                            Entropy Check → LLM Fallback (if LLM enabled)
                                         ↓
                                    UNKNOWN (if LLM disabled)
```

### V2 Mode (Enabled)
```
User Query → MemoryAwareReasoner → MemoryFirstReasonerV2 → RelationalCacheV2
                                 ↓
                            Deterministic Answer or UNKNOWN
```

## Key Components

### 1. MemoryAwareReasoner
**File:** `nanobot/memory/memory_aware_reasoner.py`

Extended to support both v1 and v2 modes:
- Initializes v2 components (`MemoryFirstReasonerV2`, `RelationalCacheV2`) when flag enabled
- Routes queries to v2 reasoner for supported types (superlatives, pairwise comparisons)
- Logs cache hits/misses and UNKNOWN outcomes
- Falls back to v1 behavior when v2 disabled

### 2. ConsolidationPipeline
**File:** `nanobot/memory/consolidation.py`

Extended to support v2 extraction:
- Uses `RelationExtractionEngineV2` with confidence gating when v2 enabled
- Extracts relations from interaction events with confidence thresholds
- Logs accepted/rejected extractions
- Maintains v1 behavior when v2 disabled

### 3. V2 Core Modules (Pre-existing)
- `types_v2.py`: Enums and dataclasses for v2 reasoning
- `config_v2.py`: Configuration constants and defaults
- `relational_cache_v2.py`: Graph-based relational cache with BFS reasoning
- `relation_extractor_v2.py`: Template-based extraction with confidence scoring
- `memory_first_reasoner_v2.py`: Query interface with strict no-LLM policy

## V2 Features

### Deterministic Behavior
- Answers based solely on memory graph
- No LLM invocation for comparative queries
- Repeatable results for same queries

### UNKNOWN Handling
Returns UNKNOWN for:
- Ambiguous superlatives (multiple candidates at same tier)
- Self-comparisons (A vs A)
- Unknown entities (not in knowledge base)
- Incomplete information (no known relation)

### Confidence-Gated Extraction
- Only accepts relations with confidence ≥ threshold
- Logs rejected low-confidence extractions
- Prevents noisy data from entering the graph

### Graph Operations
- **Explicit inverse storage**: Automatically stores both A>B and B<A
- **Transitive reasoning**: Uses BFS to find indirect paths
- **Cycle detection**: Prevents inconsistent graph states
- **Tiered ranking**: Supports partial orderings

## Supported Query Types

### V2 Supported Queries
1. **Superlatives**: "Who is tallest?", "Who is shortest?"
2. **Pairwise**: "Is Alice taller than Bob?"
3. **Ranking**: Full entity ranking with tiers

### V1 Fallback (when v2 disabled or unsupported query)
- All other query types
- Probabilistic reasoning with entropy-based LLM invocation

## Testing

### Test Coverage
- **Total**: 213 tests passing
- **V2 Unit Tests**: 76 tests (cache, extractor, reasoner)
- **Integration Tests**: 20 tests (flag toggling, coexistence)
- **V1 Tests**: All existing tests unchanged and passing

### Key Test Categories
1. **Initialization**: v1/v2 mode switching
2. **Pairwise Queries**: Direct, transitive, opposite, unknown
3. **Superlatives**: Unique, ambiguous, invalid
4. **Ranking**: Total order, partial order, cycles
5. **Extraction**: Confidence gating, self-relations, templates
6. **Determinism**: Repeated queries, no numeric hallucination
7. **Logging**: Cache hits/misses, UNKNOWN outcomes

## Migration Guide

### Enabling V2 for Testing
1. Update `config.json`:
   ```json
   {"memory": {"use_memory_v2": true}}
   ```
2. Test comparative queries
3. Monitor logs for cache hits and UNKNOWN results
4. Verify deterministic behavior

### Adjusting Confidence Threshold
- **Higher (0.95-1.0)**: Stricter, fewer relations accepted
- **Lower (0.8-0.9)**: More permissive, more relations accepted
- **Default (0.9)**: Balanced for most use cases

### Rolling Back to V1
Simply set `use_memory_v2: false` in config. No code changes needed.

## Examples

### Example 1: Superlative Query
```python
# V2 enabled in config
user: "Who is tallest?"

# V2 cache contains: Alice > Bob > Carol
response: "Alice is tallest"
# Source: deterministic v2 answer from graph

# V2 cache contains: Alice > Carol, Bob > Carol (ambiguous)
response: UNKNOWN (falls through to LLM)
# Log: "V2 superlative query returned UNKNOWN: Insufficient information"
```

### Example 2: Relation Extraction
```python
# V2 enabled in config
user: "Alice is taller than Bob"

# Extraction result:
# Status: ACCEPTED
# Confidence: 1.0 (explicit "is" pattern)
# Relation: (Alice, TALLER_THAN, Bob)
# Log: "V2 extraction: ACCEPTED Alice TALLER_THAN Bob (confidence=1.00)"

# Graph updates:
# - Alice TALLER_THAN Bob (direct)
# - Bob SHORTER_THAN Alice (inverse)
```

## Logging

### V2 Logs to Monitor
```
INFO: MemoryFirstReasonerV2 initialized for memory-aware reasoning (v2)
INFO: V2 cache hit for superlative query: Alice is tallest
INFO: V2 superlative query returned UNKNOWN: Insufficient information
INFO: V2 extraction: ACCEPTED Alice TALLER_THAN Bob (confidence=1.00)
INFO: V2 extraction: REJECTED - Confidence 0.88 below threshold 0.90
DEBUG: V2 cache miss: query not supported or no information
```

## Performance Characteristics

### V2 Advantages
- **Deterministic**: Same query always returns same result
- **Fast**: O(V+E) graph operations, no LLM calls
- **Transparent**: Clear source attribution (direct, transitive, etc.)
- **Conservative**: Returns UNKNOWN rather than hallucinating

### V2 Limitations
- **Limited query types**: Only comparative reasoning
- **No probabilistic reasoning**: Strict TRUE/FALSE/UNKNOWN
- **Requires structured input**: Template-based extraction
- **No LLM fallback**: UNKNOWN when information insufficient

## Future Enhancements

Potential improvements for v2 system:
1. Support for more relation types (faster_than, heavier_than, etc.)
2. Fuzzy confidence scoring for partial matches
3. Temporal reasoning (relations that change over time)
4. Conflict resolution for contradictory relations
5. Graph visualization and explanation generation

## Security

- ✅ CodeQL scan: 0 alerts
- ✅ No external dependencies added
- ✅ No sensitive data exposure
- ✅ Input validation via confidence thresholds

## References

- V2 Design Doc: `IMPLEMENTATION_SUMMARY.md`
- Configuration Guide: `CONFIG.md`
- Test Examples: `tests/test_v2_integration.py`
- V2 Modules: `nanobot/memory/*_v2.py`
