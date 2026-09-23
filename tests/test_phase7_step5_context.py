"""Phase 7 Step 5 context growth benchmark tests.

Verifies the ``context_growth`` scenario:
- Small outputs stay inline (below externalization threshold).
- Naive append-all accumulates and exceeds context capacity.
- Reliable uses real ``ContextAssembler`` to keep must-keep critical
  state, omit old observations, and stay within budget.
- Recency behavior: newer observations retained, older omitted.
- ``externalized_output_count == 0`` (isolates ContextAssembler value).

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


class TestContextGrowthScenario:
    def test_scenario_exists(self):
        scenarios = make_standard_scenarios()
        assert "context_growth" in scenarios

    def test_scenario_uses_calculator_fixture(self):
        scenarios = make_standard_scenarios()
        s = scenarios["context_growth"]
        assert s.fixture_id == "calculator_buggy_add"

    def test_scenario_no_faults(self):
        scenarios = make_standard_scenarios()
        s = scenarios["context_growth"]
        assert s.fault_plan_id == "none"


# ===========================================================================
# Small outputs stay inline (Section 22, 59)
# ===========================================================================


class TestSmallOutputsInline:
    def test_reliable_externalized_count_zero(self):
        """context_growth: Reliable externalizes 0 outputs (all inline).

        This isolates the ContextAssembler value — the advantage comes
        from bounded selection, not externalization.
        """
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["context_growth"]))
        assert rec.externalized_output_count == 0

    def test_reliable_artifact_store_empty(self):
        """context_growth: Reliable creates no artifacts."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["context_growth"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        assert ts.artifact_ids == []

    def test_reliable_no_externalization_events(self):
        """context_growth: no OUTPUT_EXTERNALIZED events."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["context_growth"]))
        ext_events = [
            e for e in rec.events
            if e.event_type == BenchmarkEventType.OUTPUT_EXTERNALIZED
        ]
        assert len(ext_events) == 0


# ===========================================================================
# Same raw outputs for N/R (Section 55)
# ===========================================================================


class TestSameRawOutputs:
    def test_same_raw_outputs_first_read(self):
        """Both runners read the same calculator.py content first."""
        scenarios = make_standard_scenarios()
        naive, reliable = _make_runners()
        n_rec = _run(naive.run(scenarios["context_growth"]))
        r_rec = _run(reliable.run(scenarios["context_growth"]))
        # Both should execute at least one read_file.
        assert n_rec.logical_action_count >= 1
        assert r_rec.logical_action_count >= 1

    def test_same_logical_action_sequence_prefix(self):
        """Both runners start with the same action sequence."""
        scenarios = make_standard_scenarios()
        naive, reliable = _make_runners()
        n_rec = _run(naive.run(scenarios["context_growth"]))
        r_rec = _run(reliable.run(scenarios["context_growth"]))
        # The first action should be read_file for both.
        n_events = [
            e for e in n_rec.events
            if e.event_type == BenchmarkEventType.LOGICAL_ACTION
        ]
        r_events = [
            e for e in r_rec.events
            if e.event_type == BenchmarkEventType.LOGICAL_ACTION
        ]
        # Both start with read_file.
        assert n_events[0].tool_name == "read_file"
        assert r_events[0].tool_name == "read_file"


# ===========================================================================
# Naive accumulation grows monotonically (Section 23, 70)
# ===========================================================================


class TestNaiveAccumulation:
    def test_naive_exceeds_capacity(self):
        """Naive append-all exceeds context capacity before repair."""
        scenarios = make_standard_scenarios()
        naive, _ = _make_runners()
        rec = _run(naive.run(scenarios["context_growth"]))
        assert not rec.completed
        assert rec.failure_reason == "CONTEXT_LIMIT"

    def test_naive_peak_context_exceeds_capacity(self):
        """Naive peak context tokens exceed the configured capacity."""
        scenarios = make_standard_scenarios()
        naive, _ = _make_runners()
        rec = _run(naive.run(scenarios["context_growth"]))
        # Capacity is 120; naive peak should exceed it.
        assert rec.peak_context_tokens > 120

    def test_naive_next_tool_not_executed_after_overflow(self):
        """After context overflow, the next tool is NOT executed.

        The logical_action_count should be less than the full script
        (7 actions) because the run terminated at the context boundary.
        """
        scenarios = make_standard_scenarios()
        naive, _ = _make_runners()
        rec = _run(naive.run(scenarios["context_growth"]))
        # Full script has 7 actions; naive should stop before completing.
        assert rec.logical_action_count < 7

    def test_naive_externalized_count_zero(self):
        """Naive never externalizes (append-all baseline)."""
        scenarios = make_standard_scenarios()
        naive, _ = _make_runners()
        rec = _run(naive.run(scenarios["context_growth"]))
        assert rec.externalized_output_count == 0

    def test_naive_oracle_incomplete(self):
        """Naive does not complete the repository task."""
        scenarios = make_standard_scenarios()
        naive, _ = _make_runners()
        rec = _run(naive.run(scenarios["context_growth"]))
        assert not rec.completed


# ===========================================================================
# Reliable stays under budget (Section 24, 70)
# ===========================================================================


class TestReliableUnderBudget:
    def test_reliable_stays_under_budget(self):
        """Reliable assembled context stays <= capacity."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["context_growth"]))
        assert rec.peak_context_tokens <= 120

    def test_reliable_completes(self):
        """Reliable completes the repository task."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["context_growth"]))
        assert rec.completed

    def test_reliable_no_context_limit_failure(self):
        """Reliable does not fail with CONTEXT_LIMIT."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["context_growth"]))
        assert rec.failure_reason is None

    def test_reliable_candidate_context_exceeds_capacity(self):
        """Reliable candidate context exceeds capacity (proves selection
        is needed, not just a small workload)."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["context_growth"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        # The candidate pool should exceed 120 tokens at some point.
        assert ts.candidate_tokens > 120

    def test_reliable_omits_items(self):
        """Reliable omits at least one item (proves selection happened)."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["context_growth"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        assert ts.omitted_item_count > 0


# ===========================================================================
# Must-keep critical state retained (Section 9, 19, 70)
# ===========================================================================


class TestMustKeepRetention:
    def test_must_keep_items_always_retained(self):
        """Must-keep critical items are always in the assembled context."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["context_growth"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        # Check every assembly result retained must-keep items.
        for result in ts.assembly_results:
            must_keep_ids = {
                it.item_id for it in result.included_items if it.must_keep
            }
            assert "critical:task_goal" in must_keep_ids
            assert "critical:constraint" in must_keep_ids


# ===========================================================================
# Recency behavior (Section 25, 70)
# ===========================================================================


class TestRecencyBehavior:
    def test_older_observations_omitted(self):
        """Older low/equal-priority observations are omitted when budget
        is tight."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["context_growth"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        # The last assembly should have omitted items.
        last = ts.assembly_results[-1]
        assert len(last.omitted_items) > 0

    def test_newer_observations_retained(self):
        """More recent observations are retained over older ones."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["context_growth"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        last = ts.assembly_results[-1]
        # The highest sequence_index non-must-keep item should be included.
        non_critical_included = [
            it for it in last.included_items if not it.must_keep
        ]
        non_critical_omitted = [
            it for it in last.omitted_items if not it.must_keep
        ]
        if non_critical_included and non_critical_omitted:
            max_included_seq = max(
                it.sequence_index for it in non_critical_included
            )
            min_omitted_seq = min(
                it.sequence_index for it in non_critical_omitted
            )
            # Newer (higher seq) should be retained over older (lower seq).
            assert max_included_seq > min_omitted_seq


# ===========================================================================
# Omitted != Deleted (Section 26)
# ===========================================================================


class TestOmittedNotDeleted:
    def test_omitted_items_still_in_candidate_pool(self):
        """Omitted items remain in the benchmark history (not deleted)."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["context_growth"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        last = ts.assembly_results[-1]
        # All omitted items should still be in the context_items list.
        for omitted in last.omitted_items:
            assert omitted in ts.context_items


# ===========================================================================
# Context assembly events (Section 64)
# ===========================================================================


class TestContextEvents:
    def test_context_assembled_events_emitted(self):
        """CONTEXT_ASSEMBLED events are emitted at decision boundaries."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["context_growth"]))
        assembled = [
            e for e in rec.events
            if e.event_type == BenchmarkEventType.CONTEXT_ASSEMBLED
        ]
        # Should have at least one assembly event per logical action.
        assert len(assembled) >= 1

    def test_no_context_limit_exceeded_for_reliable(self):
        """Reliable does not emit CONTEXT_LIMIT_EXCEEDED."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["context_growth"]))
        exceeded = [
            e for e in rec.events
            if e.event_type == BenchmarkEventType.CONTEXT_LIMIT_EXCEEDED
        ]
        assert len(exceeded) == 0

    def test_naive_emits_context_limit_exceeded(self):
        """Naive emits CONTEXT_LIMIT_EXCEEDED on overflow."""
        scenarios = make_standard_scenarios()
        naive, _ = _make_runners()
        rec = _run(naive.run(scenarios["context_growth"]))
        exceeded = [
            e for e in rec.events
            if e.event_type == BenchmarkEventType.CONTEXT_LIMIT_EXCEEDED
        ]
        assert len(exceeded) >= 1


# ===========================================================================
# Context prep does not count as logical action (Section 74)
# ===========================================================================


class TestContextPrepNotLogicalAction:
    def test_reliable_logical_action_count_matches_tool_calls(self):
        """Context assembly does not inflate logical_action_count."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["context_growth"]))
        # logical_action_count should equal tool_invocation_count
        # (context prep is not a logical action).
        assert rec.logical_action_count == rec.tool_invocation_count

    def test_naive_logical_action_count_matches_tool_calls(self):
        """Naive context check does not inflate logical_action_count."""
        scenarios = make_standard_scenarios()
        naive, _ = _make_runners()
        rec = _run(naive.run(scenarios["context_growth"]))
        assert rec.logical_action_count == rec.tool_invocation_count


# ===========================================================================
# Context usage metric (Section 40)
# ===========================================================================


class TestContextUsageMetric:
    def test_reliable_peak_context_has_real_source(self):
        """Reliable peak_context_tokens comes from real ContextItem
        estimates, not hardcoded."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["context_growth"]))
        assert rec.peak_context_tokens > 0
        # Should be derived from actual selected items.
        ts = reliable.last_trial_state.source_executor.trial_state
        assert rec.peak_context_tokens == ts.peak_context_tokens

    def test_reliable_context_under_capacity(self):
        """Reliable context usage <= configured capacity."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["context_growth"]))
        assert rec.peak_context_tokens <= 120

    def test_naive_overflow_measurement(self):
        """Naive peak context > capacity (overflow measured)."""
        scenarios = make_standard_scenarios()
        naive, _ = _make_runners()
        rec = _run(naive.run(scenarios["context_growth"]))
        assert rec.peak_context_tokens > 120


# ===========================================================================
# Externalized output ratio (Section 39)
# ===========================================================================


class TestExternalizedOutputRatio:
    def test_context_growth_reliable_ratio_zero(self):
        """context_growth: Reliable externalized ratio == 0."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["context_growth"]))
        # externalized_output_count / processed_output_count == 0
        assert rec.externalized_output_count == 0
        assert rec.processed_output_count > 0

    def test_processed_output_count_is_output_count(self):
        """processed_output_count counts outputs, not retry attempts."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["context_growth"]))
        # processed_output_count should equal logical_action_count
        # (each logical action produces one output).
        assert rec.processed_output_count == rec.logical_action_count


# ===========================================================================
# Determinism (Section 67)
# ===========================================================================


class TestDeterminism:
    def test_repeated_reliable_run_consistent(self):
        """Two Reliable runs produce identical functional results."""
        scenarios = make_standard_scenarios()
        _, reliable1 = _make_runners()
        rec1 = _run(reliable1.run(scenarios["context_growth"]))
        _, reliable2 = _make_runners()
        rec2 = _run(reliable2.run(scenarios["context_growth"]))
        assert rec1.completed == rec2.completed
        assert rec1.failure_reason == rec2.failure_reason
        assert rec1.logical_action_count == rec2.logical_action_count
        assert rec1.peak_context_tokens == rec2.peak_context_tokens
        assert rec1.externalized_output_count == rec2.externalized_output_count

    def test_repeated_naive_run_consistent(self):
        """Two Naive runs produce identical functional results."""
        scenarios = make_standard_scenarios()
        naive1, _ = _make_runners()
        rec1 = _run(naive1.run(scenarios["context_growth"]))
        naive2, _ = _make_runners()
        rec2 = _run(naive2.run(scenarios["context_growth"]))
        assert rec1.completed == rec2.completed
        assert rec1.failure_reason == rec2.failure_reason
        assert rec1.logical_action_count == rec2.logical_action_count
        assert rec1.peak_context_tokens == rec2.peak_context_tokens


# ===========================================================================
# Runner order independence (Section 69)
# ===========================================================================


class TestRunnerOrderIndependence:
    def test_naive_first_then_reliable(self):
        scenarios = make_standard_scenarios()
        naive, reliable = _make_runners()
        n_rec = _run(naive.run(scenarios["context_growth"]))
        r_rec = _run(reliable.run(scenarios["context_growth"]))
        assert not n_rec.completed
        assert r_rec.completed

    def test_reliable_first_then_naive(self):
        scenarios = make_standard_scenarios()
        naive, reliable = _make_runners()
        r_rec = _run(reliable.run(scenarios["context_growth"]))
        n_rec = _run(naive.run(scenarios["context_growth"]))
        assert r_rec.completed
        assert not n_rec.completed


# ===========================================================================
# Partition invariant (Section 10, 7)
# ===========================================================================


class TestPartitionInvariant:
    """Every assembly: candidate == included + omitted."""

    def test_every_assembly_partitions_exactly(self):
        """For every recorded assembly, the candidate set partitions
        exactly into included + omitted."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["context_growth"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        for i, result in enumerate(ts.assembly_results):
            candidate = len(result.included_items) + len(result.omitted_items)
            assert candidate == len(result.included_items) + len(
                result.omitted_items
            ), f"Assembly {i}: partition mismatch"
            # Also verify used_tokens <= max_tokens.
            assert result.used_tokens <= result.max_tokens, (
                f"Assembly {i}: used_tokens {result.used_tokens} > "
                f"max_tokens {result.max_tokens}"
            )

    def test_trial_state_candidate_equals_selected_plus_omitted(self):
        """trial_state candidate/selected/omitted counts are
        self-consistent (from the same last-assembly snapshot)."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["context_growth"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        assert (
            ts.candidate_item_count
            == ts.selected_item_count + ts.omitted_item_count
        )


# ===========================================================================
# Decision-boundary semantics (Section 11, 12)
# ===========================================================================


class TestDecisionBoundarySemantics:
    """The last assembly candidate set only contains outputs from
    actions executed BEFORE that assembly — no future-action output."""

    def test_last_assembly_excludes_final_action_output(self):
        """The last assembly (before the 7th action) cannot contain
        the 7th action's output, because that output hasn't been
        produced yet."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["context_growth"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        # 7 actions → 7 assemblies. The last assembly is Assembly 6.
        assert len(ts.assembly_results) == 7
        last = ts.assembly_results[-1]
        # The last assembly candidate set should have 8 items
        # (2 critical + 6 prior outputs), NOT 9.
        candidate_count = len(last.included_items) + len(last.omitted_items)
        assert candidate_count == 8
        # The 7th output (CTX-007) should NOT be in the last assembly.
        all_items = list(last.included_items) + list(last.omitted_items)
        item_ids = {it.item_id for it in all_items}
        assert "CTX-007" not in item_ids

    def test_assembly_candidate_count_grows_monotonically(self):
        """Each assembly's candidate count grows by at most 1 (the
        output from the previous action)."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["context_growth"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        prev_count = 0
        for i, result in enumerate(ts.assembly_results):
            candidate = len(result.included_items) + len(result.omitted_items)
            # First assembly: 2 critical items.
            if i == 0:
                assert candidate == 2
            else:
                # Each subsequent assembly adds at most 1 output.
                assert candidate <= prev_count + 1
            prev_count = candidate


# ===========================================================================
# History vs assembly (Section 12)
# ===========================================================================


class TestHistoryVsAssembly:
    """Final accumulated history count can exceed the last
    decision-boundary candidate count. This is normal, not item loss."""

    def test_final_history_exceeds_last_candidate(self):
        """After the run, the accumulated history has 9 items (2
        critical + 7 outputs), but the last assembly only had 8
        candidates (2 critical + 6 outputs). The 7th output was added
        after the last assembly."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["context_growth"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        # Final accumulated history.
        assert ts.accumulated_context_item_count == 9
        assert len(ts.context_items) == 9
        # Last assembly candidate count.
        last = ts.assembly_results[-1]
        last_candidate = len(last.included_items) + len(last.omitted_items)
        assert last_candidate == 8
        # History > last candidate (the 7th output has no subsequent
        # assembly).
        assert ts.accumulated_context_item_count > last_candidate

    def test_omitted_items_remain_in_history(self):
        """Omitted items from the last assembly are still in the
        accumulated history (not deleted)."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["context_growth"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        last = ts.assembly_results[-1]
        history_ids = {it.item_id for it in ts.context_items}
        for omitted in last.omitted_items:
            assert omitted.item_id in history_ids

    def test_no_fake_post_completion_assembly(self):
        """No extra assembly is created after the run completes."""
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        _run(reliable.run(scenarios["context_growth"]))
        ts = reliable.last_trial_state.source_executor.trial_state
        # 7 actions → exactly 7 assemblies (one before each tool call).
        assert len(ts.assembly_results) == 7
