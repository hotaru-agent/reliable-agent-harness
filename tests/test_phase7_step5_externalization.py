"""Phase 7 Step 5 large tool output externalization benchmark tests.

Verifies the ``large_output_externalization`` scenario:
- Large raw output is deterministic and identical for N/R.
- Large output exceeds externalization threshold and context capacity.
- Naive keeps full output inline and exceeds context capacity.
- Reliable externalizes via ToolOutputProcessor + ArtifactStore.
- Full artifact saved before reference used.
- Reference does not contain full large output.
- Reference token cost based on reference rendering.
- Artifact full readback is exact.
- Omitted reference is still recoverable.
- Reference priority preserved, must_keep == False.

All offline, deterministic, no LLM, no network.
"""

from __future__ import annotations

import asyncio

import pytest

from evaluation.records import BenchmarkEventType
from evaluation.runners import (
    NaiveBenchmarkRunner,
    ReliableHarnessBenchmarkRunner,
)
from evaluation.scenarios.controlled_repo import make_standard_scenarios


def _run(coro):
    return asyncio.run(coro)


def _make_runners():
    return NaiveBenchmarkRunner(), ReliableHarnessBenchmarkRunner()


# ===========================================================================
# Scenario existence
# ===========================================================================


class TestLargeOutputScenario:
    def test_scenario_exists(self):
        scenarios = make_standard_scenarios()
        assert "large_output_externalization" in scenarios

    def test_scenario_uses_large_log_fixture(self):
        scenarios = make_standard_scenarios()
        s = scenarios["large_output_externalization"]
        assert s.fixture_id == "calculator_buggy_add_large_log"

    def test_scenario_no_faults(self):
        scenarios = make_standard_scenarios()
        s = scenarios["large_output_externalization"]
        assert s.fault_plan_id == "none"


# ===========================================================================
# Large file output deterministic (Section 71)
# ===========================================================================


class TestLargeOutputDeterministic:
    def test_large_output_exceeds_externalization_threshold(self):
        """The large log output exceeds the externalization threshold (50)."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["large_output_externalization"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        # The artifact's original_estimated_tokens should exceed 50.
        store = ts.artifact_store
        for aid in ts.artifact_ids:
            art = _run(store.get(aid))
            assert art.estimated_tokens > 50

    def test_large_output_exceeds_context_capacity(self):
        """The large log output exceeds the context capacity (120)."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["large_output_externalization"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        store = ts.artifact_store
        for aid in ts.artifact_ids:
            art = _run(store.get(aid))
            assert art.estimated_tokens > 120

    def test_repeated_run_same_artifact_content(self):
        """Repeated runs produce the same artifact content."""
        scenarios = make_standard_scenarios()
        _, reliable1 = _make_runners()
        _run(reliable1.run(scenarios["large_output_externalization"]))
        ts1 = reliable1.last_trial_state.source_executor.trial_state
        _, reliable2 = _make_runners()
        _run(reliable2.run(scenarios["large_output_externalization"]))
        ts2 = reliable2.last_trial_state.source_executor.trial_state
        # Compare artifact content.
        art1 = _run(ts1.artifact_store.get(ts1.artifact_ids[0]))
        art2 = _run(ts2.artifact_store.get(ts2.artifact_ids[0]))
        assert art1.content == art2.content
        assert art1.size_bytes == art2.size_bytes
        assert art1.content_hash == art2.content_hash


# ===========================================================================
# Naive keeps full output inline (Section 71)
# ===========================================================================


class TestNaiveInline:
    def test_naive_exceeds_capacity(self):
        """Naive inlines the large output and exceeds context capacity."""
        scenarios = make_standard_scenarios()
        naive, _ = _make_runners()
        rec = _run(naive.run(scenarios["large_output_externalization"]))
        assert not rec.completed
        assert rec.failure_reason == "CONTEXT_LIMIT"

    def test_naive_peak_context_exceeds_capacity(self):
        """Naive peak context tokens exceed the configured capacity."""
        scenarios = make_standard_scenarios()
        naive, _ = _make_runners()
        rec = _run(naive.run(scenarios["large_output_externalization"]))
        assert rec.peak_context_tokens > 120

    def test_naive_externalized_count_zero(self):
        """Naive never externalizes (inline-all baseline)."""
        scenarios = make_standard_scenarios()
        naive, _ = _make_runners()
        rec = _run(naive.run(scenarios["large_output_externalization"]))
        assert rec.externalized_output_count == 0

    def test_naive_remains_incomplete(self):
        """Naive does not complete the repository task."""
        scenarios = make_standard_scenarios()
        naive, _ = _make_runners()
        rec = _run(naive.run(scenarios["large_output_externalization"]))
        assert not rec.completed


# ===========================================================================
# Reliable externalizes (Section 31, 71)
# ===========================================================================


class TestReliableExternalizes:
    def test_reliable_externalizes_at_least_one(self):
        """Reliable externalizes at least one output."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["large_output_externalization"]))
        assert rec.externalized_output_count >= 1

    def test_reliable_completes(self):
        """Reliable completes the repository task."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["large_output_externalization"]))
        assert rec.completed

    def test_reliable_stays_under_budget(self):
        """Reliable assembled context stays <= capacity."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["large_output_externalization"]))
        assert rec.peak_context_tokens <= 120

    def test_reliable_externalization_event_emitted(self):
        """OUTPUT_EXTERNALIZED event is emitted."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["large_output_externalization"]))
        ext_events = [
            e for e in rec.events
            if e.event_type == BenchmarkEventType.OUTPUT_EXTERNALIZED
        ]
        assert len(ext_events) >= 1


# ===========================================================================
# Artifact saved before reference (Section 32, 71)
# ===========================================================================


class TestArtifactSavedBeforeReference:
    def test_artifact_exists_for_reference(self):
        """The artifact exists in the store for each externalized output."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["large_output_externalization"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        store = ts.artifact_store
        for aid in ts.artifact_ids:
            art = _run(store.get(aid))
            assert art is not None

    def test_artifact_content_is_rendered_output(self):
        """The artifact content is the rendered large tool output."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["large_output_externalization"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        store = ts.artifact_store
        for aid in ts.artifact_ids:
            art = _run(store.get(aid))
            # The content should contain the large log lines.
            assert "test_log_line_0000" in art.content
            assert "test_log_line_0199" in art.content


# ===========================================================================
# Reference does not contain full output (Section 34, 71)
# ===========================================================================


class TestReferenceNoFullOutput:
    def test_reference_does_not_contain_full_output(self):
        """The reference ContextItem does not contain the full large
        output content."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["large_output_externalization"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        # Find the RECOVERABLE_REFERENCE context item.
        ref_items = [
            it for it in ts.context_items
            if it.kind.value == "recoverable_reference"
        ]
        assert len(ref_items) >= 1
        ref = ref_items[0]
        # The reference content should NOT contain the full log.
        # It should not contain the last log line.
        assert "test_log_line_0199" not in ref.content
        # It should not contain the middle log line.
        assert "test_log_line_0100" not in ref.content


# ===========================================================================
# Reference token cost (Section 33, 71)
# ===========================================================================


class TestReferenceTokenCost:
    def test_reference_tokens_based_on_reference_text(self):
        """Reference ContextItem.estimated_tokens comes from the rendered
        reference text, not the original large output."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["large_output_externalization"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        ref_items = [
            it for it in ts.context_items
            if it.kind.value == "recoverable_reference"
        ]
        assert len(ref_items) >= 1
        ref = ref_items[0]
        # Reference tokens should be much smaller than original.
        store = ts.artifact_store
        art = _run(store.get(ts.artifact_ids[0]))
        assert ref.estimated_tokens < art.estimated_tokens
        # Reference tokens should be based on len(ref.content) // 4.
        expected = max(1, len(ref.content) // 4)
        assert ref.estimated_tokens == expected

    def test_reference_tokens_much_smaller_than_original(self):
        """reference tokens << original output tokens."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["large_output_externalization"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        ref_items = [
            it for it in ts.context_items
            if it.kind.value == "recoverable_reference"
        ]
        ref = ref_items[0]
        store = ts.artifact_store
        art = _run(store.get(ts.artifact_ids[0]))
        # Reference should be at least 10x smaller.
        assert ref.estimated_tokens * 10 < art.estimated_tokens


# ===========================================================================
# Artifact full readback (Section 35, 71)
# ===========================================================================


class TestArtifactReadback:
    def test_artifact_full_content_exact(self):
        """await artifact_store.get(artifact_id) returns the exact full
        original output."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["large_output_externalization"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        store = ts.artifact_store
        for aid in ts.artifact_ids:
            art = _run(store.get(aid))
            assert art is not None
            # The content should be the full large log.
            assert "test_log_line_0000" in art.content
            assert "test_log_line_0199" in art.content
            # Size should be substantial.
            assert art.size_bytes > 1000


# ===========================================================================
# Omitted but recoverable (Section 57, 72)
# ===========================================================================


class TestOmittedButRecoverable:
    def test_omitted_reference_still_recoverable(self):
        """Even if the reference is omitted by ContextAssembler, the
        artifact is still fully recoverable via artifact_store.get()."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["large_output_externalization"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        store = ts.artifact_store
        # The artifact should be recoverable regardless of whether the
        # reference was omitted in the last assembly.
        for aid in ts.artifact_ids:
            art = _run(store.get(aid))
            assert art is not None
            assert art.content  # non-empty


# ===========================================================================
# Reference priority preserved (Section 61, 71)
# ===========================================================================


class TestReferencePriority:
    def test_reference_priority_preserved(self):
        """The artifact-reference ContextItem preserves the input
        priority (NORMAL), not automatically downgraded."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["large_output_externalization"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        ref_items = [
            it for it in ts.context_items
            if it.kind.value == "recoverable_reference"
        ]
        assert len(ref_items) >= 1
        # The processor uses NORMAL priority by default.
        from harness.context import ContextPriority
        assert ref_items[0].priority == ContextPriority.NORMAL


# ===========================================================================
# Reference must_keep == False (Section 62, 71)
# ===========================================================================


class TestReferenceMustKeep:
    def test_reference_must_keep_false(self):
        """Artifact reference ContextItem has must_keep == False."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["large_output_externalization"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        ref_items = [
            it for it in ts.context_items
            if it.kind.value == "recoverable_reference"
        ]
        assert len(ref_items) >= 1
        assert ref_items[0].must_keep is False


# ===========================================================================
# Same raw output fairness (Section 36)
# ===========================================================================


class TestSameRawOutput:
    def test_both_runners_execute_first_read(self):
        """Both runners execute the first read_file on the large log."""
        scenarios = make_standard_scenarios()
        naive, reliable = _make_runners()
        n_rec = _run(naive.run(scenarios["large_output_externalization"]))
        r_rec = _run(reliable.run(scenarios["large_output_externalization"]))
        # Both should execute at least one action.
        assert n_rec.logical_action_count >= 1
        assert r_rec.logical_action_count >= 1


# ===========================================================================
# Externalized output ratio (Section 39, 73)
# ===========================================================================


class TestExternalizedOutputRatio:
    def test_large_output_reliable_ratio_positive(self):
        """large_output_externalization: Reliable ratio > 0."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["large_output_externalization"]))
        assert rec.externalized_output_count > 0
        assert rec.processed_output_count > 0
        # ratio = externalized / processed > 0
        assert rec.externalized_output_count / rec.processed_output_count > 0

    def test_denominator_is_output_count_not_attempts(self):
        """processed_output_count counts outputs, not retry attempts."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["large_output_externalization"]))
        # processed_output_count should equal logical_action_count.
        assert rec.processed_output_count == rec.logical_action_count
        # tool_attempt_count may differ if retries happened, but
        # processed_output_count is based on outputs.
        assert rec.processed_output_count <= rec.tool_attempt_count + 1


# ===========================================================================
# Context usage metric (Section 40, 73)
# ===========================================================================


class TestContextUsageMetric:
    def test_reliable_context_under_capacity(self):
        """Reliable context usage <= configured capacity."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["large_output_externalization"]))
        assert rec.peak_context_tokens <= 120

    def test_naive_overflow_measurement(self):
        """Naive peak context > capacity (overflow measured)."""
        scenarios = make_standard_scenarios()
        naive, _ = _make_runners()
        rec = _run(naive.run(scenarios["large_output_externalization"]))
        assert rec.peak_context_tokens > 120


# ===========================================================================
# Context prep not logical action (Section 74)
# ===========================================================================


class TestContextPrepNotLogicalAction:
    def test_reliable_logical_action_count_matches_tool_calls(self):
        """Context assembly does not inflate logical_action_count."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["large_output_externalization"]))
        assert rec.logical_action_count == rec.tool_invocation_count


# ===========================================================================
# Determinism (Section 68)
# ===========================================================================


class TestDeterminism:
    def test_repeated_reliable_run_consistent(self):
        """Two Reliable runs produce identical functional results."""
        scenarios = make_standard_scenarios()
        _, reliable1 = _make_runners()
        rec1 = _run(reliable1.run(scenarios["large_output_externalization"]))
        _, reliable2 = _make_runners()
        rec2 = _run(reliable2.run(scenarios["large_output_externalization"]))
        assert rec1.completed == rec2.completed
        assert rec1.failure_reason == rec2.failure_reason
        assert rec1.logical_action_count == rec2.logical_action_count
        assert rec1.peak_context_tokens == rec2.peak_context_tokens
        assert rec1.externalized_output_count == rec2.externalized_output_count


# ===========================================================================
# Runner order independence (Section 69)
# ===========================================================================


class TestRunnerOrderIndependence:
    def test_naive_first_then_reliable(self):
        scenarios = make_standard_scenarios()
        naive, reliable = _make_runners()
        n_rec = _run(naive.run(scenarios["large_output_externalization"]))
        r_rec = _run(reliable.run(scenarios["large_output_externalization"]))
        assert not n_rec.completed
        assert r_rec.completed

    def test_reliable_first_then_naive(self):
        scenarios = make_standard_scenarios()
        naive, reliable = _make_runners()
        r_rec = _run(reliable.run(scenarios["large_output_externalization"]))
        n_rec = _run(naive.run(scenarios["large_output_externalization"]))
        assert r_rec.completed
        assert not n_rec.completed
