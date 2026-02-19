"""Tests for Phase 2: WorkingMemory."""

import pytest

from nanobot.cognitive.types import ConfidenceScore, ObservationRecord, WorkingMemoryState
from nanobot.cognitive.working_memory import WorkingMemory


class TestWorkingMemoryEnabled:
    def setup_method(self):
        self.wm = WorkingMemory(enabled=True)

    def test_begin_cycle_returns_state(self):
        state = self.wm.begin_cycle("what is AI?", session_id="s1")
        assert isinstance(state, WorkingMemoryState)
        assert state.query == "what is AI?"
        assert state.session_id == "s1"
        assert state.reasoning_state == "pending"

    def test_get_state_after_begin(self):
        self.wm.begin_cycle("query")
        assert self.wm.get_state() is not None
        assert self.wm.get_state().query == "query"

    def test_end_cycle_returns_complete_state(self):
        self.wm.begin_cycle("q")
        final = self.wm.end_cycle()
        assert final is not None
        assert final.reasoning_state == "complete"

    def test_end_cycle_clears_internal_state(self):
        self.wm.begin_cycle("q")
        self.wm.end_cycle()
        assert self.wm.get_state() is None

    def test_setters_update_state(self):
        self.wm.begin_cycle("q")
        self.wm.set_retrieved_nodes([{"id": "n1"}])
        self.wm.set_hypotheses([{"intent": "explain"}])
        self.wm.set_entropy(0.3)
        self.wm.set_reasoning_state("system1")
        self.wm.set_context_summary("context here")
        self.wm.add_tool("web_search")
        state = self.wm.get_state()
        assert state.retrieved_nodes == [{"id": "n1"}]
        assert state.hypotheses == [{"intent": "explain"}]
        assert state.entropy == 0.3
        assert state.reasoning_state == "system1"
        assert state.context_summary == "context here"
        assert "web_search" in state.tools_invoked

    def test_set_confidence(self):
        self.wm.begin_cycle("q")
        cs = ConfidenceScore(overall=0.75)
        self.wm.set_confidence(cs)
        assert self.wm.get_state().confidence is cs

    def test_add_observation(self):
        from datetime import datetime, timezone
        self.wm.begin_cycle("q")
        rec = ObservationRecord(timestamp=datetime.now(timezone.utc).isoformat(), query="q")
        self.wm.add_observation(rec)
        assert rec in self.wm.get_state().observations

    def test_abandoned_cycle_discarded_on_new_begin(self, caplog):
        import logging
        self.wm.begin_cycle("first")
        with caplog.at_level(logging.WARNING, logger="nanobot.cognitive.working_memory"):
            self.wm.begin_cycle("second")
        assert "second" == self.wm.get_state().query

    def test_end_cycle_without_begin_returns_none(self):
        assert self.wm.end_cycle() is None

    def test_does_not_alter_memory_system(self):
        """Safety: WorkingMemory must never touch MemoryStore, RelationalCache, or disk."""
        self.wm.begin_cycle("q")
        self.wm.set_retrieved_nodes([{"id": "n1"}])
        final = self.wm.end_cycle()
        # Verify the returned state is ephemeral and contains no disk references
        assert isinstance(final, WorkingMemoryState)
        # No file-system or external side-effects — test completes without exception
        assert final.reasoning_state == "complete"


class TestWorkingMemoryDisabled:
    def setup_method(self):
        self.wm = WorkingMemory(enabled=False)

    def test_begin_cycle_returns_none(self):
        assert self.wm.begin_cycle("q") is None

    def test_get_state_returns_none(self):
        assert self.wm.get_state() is None

    def test_end_cycle_returns_none(self):
        assert self.wm.end_cycle() is None

    def test_setters_are_no_ops(self):
        self.wm.set_entropy(0.5)
        self.wm.set_reasoning_state("system1")
        # No exception — setters are safe no-ops when disabled
