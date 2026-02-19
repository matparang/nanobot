"""Comprehensive unit tests for NanobotBeliefEngine."""

import pytest

from nanobot.memory.belief_engine import (
    BeliefQueryResult,
    CycleResolutionResult,
    Fact,
    NanobotBeliefEngine,
)
from nanobot.memory.belief_ontology import (
    DEFAULT_ONTOLOGY,
    get_inverse,
    is_transitive,
)
from nanobot.memory.types_v2 import RelationType, TruthValue


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> NanobotBeliefEngine:
    return NanobotBeliefEngine()


@pytest.fixture
def height_engine() -> NanobotBeliefEngine:
    """Engine pre-loaded with: Alice > Bob > Carol (TALLER_THAN)."""
    e = NanobotBeliefEngine()
    e.add_fact("Alice", RelationType.TALLER_THAN, "Bob", confidence=0.9)
    e.add_fact("Bob", RelationType.TALLER_THAN, "Carol", confidence=0.8)
    return e


# ---------------------------------------------------------------------------
# Ontology tests
# ---------------------------------------------------------------------------

class TestOntology:
    def test_inverse_taller_than(self):
        assert get_inverse(RelationType.TALLER_THAN) == RelationType.SHORTER_THAN

    def test_inverse_shorter_than(self):
        assert get_inverse(RelationType.SHORTER_THAN) == RelationType.TALLER_THAN

    def test_inverse_related_to_is_self(self):
        assert get_inverse(RelationType.RELATED_TO) == RelationType.RELATED_TO

    def test_transitive_taller_than(self):
        assert is_transitive(RelationType.TALLER_THAN) is True

    def test_not_transitive_supplies(self):
        assert is_transitive(RelationType.SUPPLIES) is False

    def test_unknown_relation_inverse_is_none(self):
        # Supplying a custom ontology with no entry
        assert get_inverse(RelationType.TALLER_THAN, {}) is None

    def test_unknown_relation_not_transitive(self):
        assert is_transitive(RelationType.TALLER_THAN, {}) is False


# ---------------------------------------------------------------------------
# add_fact / delete_fact / introspection
# ---------------------------------------------------------------------------

class TestAddDeleteFact:
    def test_add_fact_stores_direct_and_inverse(self, engine):
        engine.add_fact("Alice", RelationType.TALLER_THAN, "Bob", confidence=0.9)
        assert engine.get_fact_count() == 2  # direct + inverse
        assert engine.get_entity_count() == 2

    def test_add_fact_clamps_confidence(self, engine):
        engine.add_fact("A", RelationType.TALLER_THAN, "B", confidence=1.5)
        key = ("A", RelationType.TALLER_THAN, "B")
        assert engine.facts[key].confidence == 1.0

    def test_add_fact_clamps_negative_confidence(self, engine):
        engine.add_fact("A", RelationType.TALLER_THAN, "B", confidence=-0.5)
        key = ("A", RelationType.TALLER_THAN, "B")
        assert engine.facts[key].confidence == 0.0

    def test_conflict_keeps_higher_confidence(self, engine):
        engine.add_fact("A", RelationType.TALLER_THAN, "B", confidence=0.6)
        engine.add_fact("A", RelationType.TALLER_THAN, "B", confidence=0.9)
        assert engine.facts[("A", RelationType.TALLER_THAN, "B")].confidence == 0.9

    def test_conflict_keeps_existing_on_lower_confidence(self, engine):
        engine.add_fact("A", RelationType.TALLER_THAN, "B", confidence=0.9)
        engine.add_fact("A", RelationType.TALLER_THAN, "B", confidence=0.6)
        assert engine.facts[("A", RelationType.TALLER_THAN, "B")].confidence == 0.9

    def test_delete_fact_removes_direct_and_inverse(self, engine):
        engine.add_fact("Alice", RelationType.TALLER_THAN, "Bob", confidence=0.9)
        result = engine.delete_fact("Alice", RelationType.TALLER_THAN, "Bob")
        assert result is True
        assert engine.get_fact_count() == 0

    def test_delete_nonexistent_fact_returns_false(self, engine):
        assert engine.delete_fact("X", RelationType.TALLER_THAN, "Y") is False

    def test_symmetric_relation_not_duplicated(self, engine):
        engine.add_fact("A", RelationType.RELATED_TO, "B", confidence=0.7)
        # RELATED_TO has inverse=RELATED_TO; should not double-store
        assert engine.get_fact_count() == 1

    def test_get_all_facts(self, engine):
        engine.add_fact("Alice", RelationType.TALLER_THAN, "Bob", confidence=0.9)
        facts = engine.get_all_facts()
        assert len(facts) == 2
        assert all(isinstance(f, Fact) for f in facts)


# ---------------------------------------------------------------------------
# Pairwise query
# ---------------------------------------------------------------------------

class TestQueryPairwise:
    def test_direct_true(self, height_engine):
        result = height_engine.query("Alice", RelationType.TALLER_THAN, "Bob")
        assert result.value == TruthValue.TRUE
        assert result.source == "direct"
        assert result.confidence == 0.9

    def test_transitive_true(self, height_engine):
        result = height_engine.query("Alice", RelationType.TALLER_THAN, "Carol")
        assert result.value == TruthValue.TRUE
        assert result.source == "transitive"
        assert result.confidence == pytest.approx(0.9 * 0.8)

    def test_inverse_automatically_added(self, height_engine):
        result = height_engine.query("Bob", RelationType.SHORTER_THAN, "Alice")
        assert result.value == TruthValue.TRUE

    def test_self_comparison_unknown(self, height_engine):
        result = height_engine.query("Alice", RelationType.TALLER_THAN, "Alice")
        assert result.value == TruthValue.UNKNOWN
        assert result.source == "self_comparison"

    def test_unknown_entity_returns_unknown(self, height_engine):
        result = height_engine.query("Alice", RelationType.TALLER_THAN, "Zara")
        assert result.value == TruthValue.UNKNOWN
        assert result.source == "unknown_entity"

    def test_no_relation_returns_unknown(self, engine):
        engine.add_fact("A", RelationType.TALLER_THAN, "B", confidence=0.9)
        result = engine.query("B", RelationType.TALLER_THAN, "A")
        # B is SHORTER_THAN A, not TALLER_THAN A
        assert result.value in (TruthValue.FALSE, TruthValue.UNKNOWN)

    def test_opposite_relation_returns_false(self, engine):
        engine.add_fact("Alice", RelationType.TALLER_THAN, "Bob", confidence=0.9)
        # Alice is SHORTER_THAN Bob should be FALSE
        result = engine.query("Alice", RelationType.SHORTER_THAN, "Bob")
        assert result.value == TruthValue.FALSE

    def test_query_returns_list_when_no_obj(self, height_engine):
        targets = height_engine.query("Alice", RelationType.TALLER_THAN)
        assert isinstance(targets, list)
        assert "Bob" in targets
        assert "Carol" in targets  # transitive

    def test_non_transitive_query_targets(self, engine):
        engine.add_fact("A", RelationType.SUPPLIES, "B")
        engine.add_fact("B", RelationType.SUPPLIES, "C")
        targets = engine.query("A", RelationType.SUPPLIES)
        assert "B" in targets
        assert "C" not in targets  # SUPPLIES is not transitive


# ---------------------------------------------------------------------------
# Superlative query
# ---------------------------------------------------------------------------

class TestQuerySuperlative:
    def test_unique_tallest(self, height_engine):
        result = height_engine.query_superlative("tall", "most")
        assert result.value == TruthValue.TRUE
        assert result.details["winner"] == "Alice"

    def test_unique_shortest(self, height_engine):
        result = height_engine.query_superlative("tall", "least")
        assert result.value == TruthValue.TRUE
        assert result.details["winner"] == "Carol"

    def test_unknown_attribute_returns_unknown(self, height_engine):
        result = height_engine.query_superlative("nonexistent", "most")
        assert result.value == TruthValue.UNKNOWN

    def test_tie_returns_unknown_with_default_policy(self, engine):
        # Two entities, neither beats the other
        engine.add_fact("A", RelationType.TALLER_THAN, "C", confidence=0.9)
        engine.add_fact("B", RelationType.TALLER_THAN, "C", confidence=0.9)
        result = engine.query_superlative("tall", "most")
        assert result.value == TruthValue.UNKNOWN

    def test_tiebreak_confidence_policy(self):
        e = NanobotBeliefEngine(superlative_policy="tiebreak_by_confidence")
        e.add_fact("A", RelationType.TALLER_THAN, "C", confidence=0.9)
        e.add_fact("B", RelationType.TALLER_THAN, "C", confidence=0.7)
        result = e.query_superlative("tall", "most")
        assert result.value == TruthValue.TRUE
        assert result.details["winner"] == "A"

    def test_empty_engine_superlative_unknown(self, engine):
        result = engine.query_superlative("tall", "most")
        assert result.value == TruthValue.UNKNOWN

    def test_fastest(self, engine):
        engine.add_fact("Cheetah", RelationType.FASTER_THAN, "Horse", confidence=0.95)
        engine.add_fact("Horse", RelationType.FASTER_THAN, "Human", confidence=0.9)
        result = engine.query_superlative("fast", "most")
        assert result.value == TruthValue.TRUE
        assert result.details["winner"] == "Cheetah"


# ---------------------------------------------------------------------------
# Cycle detection / resolution
# ---------------------------------------------------------------------------

class TestCycleDetection:
    def test_direct_cycle_resolved(self, engine):
        engine.add_fact("A", RelationType.TALLER_THAN, "B", confidence=0.9)
        engine.add_fact("B", RelationType.TALLER_THAN, "C", confidence=0.8)
        # This creates a cycle A>B>C>A; weakest edge should be evicted
        result = engine.add_fact("C", RelationType.TALLER_THAN, "A", confidence=0.5)
        assert result is not None
        assert result.resolved is True
        assert result.evicted_fact is not None

    def test_weakest_fact_evicted(self, engine):
        engine.add_fact("A", RelationType.TALLER_THAN, "B", confidence=0.9)
        engine.add_fact("B", RelationType.TALLER_THAN, "C", confidence=0.8)
        result = engine.add_fact("C", RelationType.TALLER_THAN, "A", confidence=0.5)
        # Weakest is C->A with 0.5
        assert result.evicted_fact.confidence == pytest.approx(0.5)

    def test_no_cycle_returns_none(self, engine):
        engine.add_fact("A", RelationType.TALLER_THAN, "B", confidence=0.9)
        result = engine.add_fact("B", RelationType.TALLER_THAN, "C", confidence=0.8)
        # No cycle A->B->C, result should be None
        assert result is None

    def test_non_transitive_no_cycle_check(self, engine):
        engine.add_fact("A", RelationType.SUPPLIES, "B", confidence=0.9)
        engine.add_fact("B", RelationType.SUPPLIES, "C", confidence=0.8)
        result = engine.add_fact("C", RelationType.SUPPLIES, "A", confidence=0.5)
        # SUPPLIES is not transitive — no cycle detection
        assert result is None

    def test_has_cycle_false_for_dag(self, height_engine):
        assert height_engine._has_cycle(RelationType.TALLER_THAN) is False

    def test_superlative_cycle_detected_returns_unknown(self, engine):
        # Manually force a cycle without resolution by directly inserting
        engine._store(
            Fact("A", RelationType.TALLER_THAN, "B", confidence=0.9)
        )
        engine._store(
            Fact("B", RelationType.TALLER_THAN, "C", confidence=0.8)
        )
        engine._store(
            Fact("C", RelationType.TALLER_THAN, "A", confidence=0.7)
        )
        engine.adjacency[("A", RelationType.TALLER_THAN)].add("B")
        engine.adjacency[("B", RelationType.TALLER_THAN)].add("C")
        engine.adjacency[("C", RelationType.TALLER_THAN)].add("A")
        result = engine.query_superlative("tall", "most")
        assert result.value == TruthValue.UNKNOWN
        assert result.cycle_detected is True


# ---------------------------------------------------------------------------
# Generate hypotheses
# ---------------------------------------------------------------------------

class TestGenerateHypotheses:
    def test_superlative_hypothesis(self, height_engine):
        output = height_engine.generate_hypotheses("Who is the tallest?")
        assert output["query_type"] == "superlative"
        assert len(output["hypotheses"]) >= 1
        assert output["hypotheses"][0]["result"] == "Alice"

    def test_fallback_entity_hypotheses(self, height_engine):
        output = height_engine.generate_hypotheses("Tell me about the entities")
        assert len(output["hypotheses"]) >= 1
        assert isinstance(output["facts_stored"], int)

    def test_empty_engine_hypotheses(self, engine):
        output = engine.generate_hypotheses("Who is tallest?")
        assert output["facts_stored"] == 0

    def test_max_hypotheses_respected(self, height_engine):
        output = height_engine.generate_hypotheses("entities", max_hypotheses=1)
        assert len(output["hypotheses"]) <= 1

    def test_entropy_in_output(self, height_engine):
        output = height_engine.generate_hypotheses("Who is tallest?")
        assert "entropy" in output
        assert 0.0 <= output["entropy"] <= 1.0

    def test_requires_llm_field(self, height_engine):
        output = height_engine.generate_hypotheses("Who is tallest?")
        assert "requires_llm" in output


# ---------------------------------------------------------------------------
# query_attribute
# ---------------------------------------------------------------------------

class TestQueryAttribute:
    def test_known_attribute_returns_list(self, height_engine):
        targets = height_engine.query_attribute("Alice", "height")
        assert isinstance(targets, list)
        assert "Bob" in targets

    def test_unknown_attribute_returns_unknown(self, height_engine):
        result = height_engine.query_attribute("Alice", "nonexistent")
        assert isinstance(result, BeliefQueryResult)
        assert result.value == TruthValue.UNKNOWN


# ---------------------------------------------------------------------------
# RuntimeState integration
# ---------------------------------------------------------------------------

class TestRuntimeStateIntegration:
    def test_belief_engine_toggle_default_false(self):
        from nanobot.runtime.state import RuntimeState
        s = RuntimeState()
        assert s.belief_engine_enabled is False

    def test_belief_engine_toggle_set(self):
        from nanobot.runtime.state import RuntimeState
        s = RuntimeState()
        s.belief_engine_enabled = True
        assert s.belief_engine_enabled is True
        s.belief_engine_enabled = False  # reset

    def test_belief_engine_in_get_all_toggles(self):
        from nanobot.runtime.state import RuntimeState
        s = RuntimeState()
        toggles = s.get_all_toggles()
        assert "belief_engine" in toggles

    def test_belief_engine_in_set_baseline_mode(self):
        from nanobot.runtime.state import RuntimeState
        s = RuntimeState()
        s.belief_engine_enabled = True
        previous = s.set_baseline_mode()
        assert "belief_engine" in previous
        assert previous["belief_engine"] is True
        assert s.belief_engine_enabled is False
        # Restore
        s.restore_toggles(previous)

    def test_restore_toggles_belief_engine(self):
        from nanobot.runtime.state import RuntimeState
        s = RuntimeState()
        s.restore_toggles({"belief_engine": True})
        assert s.belief_engine_enabled is True
        s.restore_toggles({"belief_engine": False})

    def test_enter_exit_baseline_mode_restores_belief_engine(self):
        from nanobot.runtime.state import RuntimeState
        s = RuntimeState()
        s._baseline_active = False  # ensure clean state
        s.belief_engine_enabled = True
        s.enter_baseline_mode()
        assert s.belief_engine_enabled is False
        s.exit_baseline_mode(restore=True)
        assert s.belief_engine_enabled is True
        s.belief_engine_enabled = False  # cleanup


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_add_same_fact_twice_no_duplicate(self, engine):
        engine.add_fact("A", RelationType.TALLER_THAN, "B", confidence=0.8)
        engine.add_fact("A", RelationType.TALLER_THAN, "B", confidence=0.8)
        assert engine.get_fact_count() == 2  # direct + inverse only

    def test_chain_of_four(self, engine):
        engine.add_fact("A", RelationType.TALLER_THAN, "B", confidence=1.0)
        engine.add_fact("B", RelationType.TALLER_THAN, "C", confidence=1.0)
        engine.add_fact("C", RelationType.TALLER_THAN, "D", confidence=1.0)
        result = engine.query("A", RelationType.TALLER_THAN, "D")
        assert result.value == TruthValue.TRUE
        assert result.depth == 3

    def test_fact_key(self):
        f = Fact("Alice", RelationType.TALLER_THAN, "Bob", confidence=0.9)
        assert f.key == ("Alice", RelationType.TALLER_THAN, "Bob")

    def test_depends_on_transitive(self, engine):
        engine.add_fact("A", RelationType.DEPENDS_ON, "B", confidence=0.9)
        engine.add_fact("B", RelationType.DEPENDS_ON, "C", confidence=0.9)
        result = engine.query("A", RelationType.DEPENDS_ON, "C")
        assert result.value == TruthValue.TRUE

    def test_contains_transitive(self, engine):
        engine.add_fact("Universe", RelationType.CONTAINS, "Galaxy", confidence=1.0)
        engine.add_fact("Galaxy", RelationType.CONTAINS, "SolarSystem", confidence=1.0)
        result = engine.query("Universe", RelationType.CONTAINS, "SolarSystem")
        assert result.value == TruthValue.TRUE

    def test_compute_tiers_empty(self, engine):
        tiers = engine._compute_tiers(RelationType.TALLER_THAN)
        assert tiers == []
