"""Phase 7 Step 4 smoke tests — scenario matrix and basic integration.

Verifies the full 6-scenario matrix and basic loop/replan smoke
behavior. All offline, deterministic, no LLM, no network.
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


# ===========================================================================
# Scenario existence
# ===========================================================================


class TestScenarioExistence:
    def test_loop_replan_scenario_exists(self):
        scenarios = make_standard_scenarios()
        assert "loop_replan" in scenarios

    def test_loop_replan_no_faults(self):
        scenarios = make_standard_scenarios()
        s = scenarios["loop_replan"]
        assert s.fault_plan_id == "none"

    def test_loop_replan_max_actions(self):
        scenarios = make_standard_scenarios()
        s = scenarios["loop_replan"]
        assert s.task.max_logical_actions == 6

    def test_loop_replan_fixture(self):
        scenarios = make_standard_scenarios()
        s = scenarios["loop_replan"]
        assert s.fixture_id == "calculator_buggy_add"


# ===========================================================================
# Full matrix smoke
# ===========================================================================


class TestFullMatrix:
    """Full 6-scenario matrix: N/R results for each scenario."""

    @pytest.mark.parametrize("scenario_id,naive_expected,reliable_expected", [
        ("clean_success", True, True),
        ("transient_failure", False, True),
        ("timeout", False, True),
        ("permanent_failure", False, False),
        ("checkpoint_recovery", False, True),
        ("loop_replan", False, True),
    ])
    def test_matrix(self, scenario_id, naive_expected, reliable_expected):
        scenarios = make_standard_scenarios()
        s = scenarios[scenario_id]

        naive = NaiveBenchmarkRunner()
        reliable = ReliableHarnessBenchmarkRunner()
        n = _run(naive.run(s))
        r = _run(reliable.run(s))

        assert n.completed is naive_expected, (
            f"{scenario_id}: naive expected {naive_expected}, "
            f"got {n.completed} (reason={n.failure_reason})"
        )
        assert r.completed is reliable_expected, (
            f"{scenario_id}: reliable expected {reliable_expected}, "
            f"got {r.completed} (reason={r.failure_reason})"
        )


# ===========================================================================
# Loop/replan smoke
# ===========================================================================


class TestLoopReplanSmoke:
    def test_naive_loops_to_max_actions(self):
        scenarios = make_standard_scenarios()
        s = scenarios["loop_replan"]
        naive = NaiveBenchmarkRunner()
        rec = _run(naive.run(s))

        assert rec.completed is False
        assert rec.failure_reason == "MAX_ACTIONS"
        assert rec.logical_action_count == 6
        assert rec.loop_detection_count == 0
        assert rec.replan_count == 0

    def test_reliable_detects_and_escapes(self):
        scenarios = make_standard_scenarios()
        s = scenarios["loop_replan"]
        reliable = ReliableHarnessBenchmarkRunner()
        rec = _run(reliable.run(s))

        assert rec.completed is True
        assert rec.logical_action_count == 5
        assert rec.loop_detection_count == 1
        assert rec.replan_count == 1
        assert rec.failure_reason is None

    def test_reliable_events_contain_loop_replan_sequence(self):
        scenarios = make_standard_scenarios()
        s = scenarios["loop_replan"]
        reliable = ReliableHarnessBenchmarkRunner()
        rec = _run(reliable.run(s))

        event_types = [e.event_type for e in rec.events]
        assert BenchmarkEventType.LOOP_DETECTED in event_types
        assert BenchmarkEventType.REPLAN_BLOCKED in event_types
        assert BenchmarkEventType.REPLAN_ACKNOWLEDGED in event_types

    def test_reliable_checkpoint_count(self):
        """Reliable produces real checkpoints from Harness execution."""
        scenarios = make_standard_scenarios()
        s = scenarios["loop_replan"]
        reliable = ReliableHarnessBenchmarkRunner()
        rec = _run(reliable.run(s))

        # 5 actions + 1 COMPLETE step = 6 checkpoints
        assert rec.checkpoint_count == 6

    def test_reliable_no_retry(self):
        """Loop scenario has no faults, so no retries."""
        scenarios = make_standard_scenarios()
        s = scenarios["loop_replan"]
        reliable = ReliableHarnessBenchmarkRunner()
        rec = _run(reliable.run(s))

        assert rec.retry_count == 0
        assert rec.tool_attempt_count == rec.tool_invocation_count


# ===========================================================================
# Aggregate accounting regression (Phase 7 Step 4 Fix)
# ===========================================================================


# The controlled scenarios for the full Phase 7 matrix.
# Phase 7 Step 4 had 6 scenarios; Phase 7 Step 5 adds 2 context scenarios
# for a total of 8.
_ALL_SCENARIOS = [
    "clean_success",
    "transient_failure",
    "timeout",
    "permanent_failure",
    "checkpoint_recovery",
    "loop_replan",
    "context_growth",
    "large_output_externalization",
]


def _make_comparator():
    """Build a BenchmarkComparator with fresh runners."""
    from evaluation.comparison import BenchmarkComparator
    return BenchmarkComparator(
        NaiveBenchmarkRunner(),
        ReliableHarnessBenchmarkRunner(),
    )


class TestAggregateAccounting:
    """Aggregate completion counts derived from actual comparison results.

    These tests prevent the aggregate numbers in the summary from
    silently drifting when new scenarios are added. The counts are
    derived from actual ``BenchmarkComparison`` results, NOT hardcoded.
    """

    def test_total_scenarios(self):
        """The full controlled matrix has 8 scenarios."""
        scenarios = make_standard_scenarios()
        assert len(_ALL_SCENARIOS) == 8
        for sid in _ALL_SCENARIOS:
            assert sid in scenarios

    def test_naive_completion_count(self):
        """Naive runner completes 1 of 8 scenarios (clean_success only).

        Derived from actual comparison results, not hardcoded.
        """
        comparator = _make_comparator()
        scenarios = make_standard_scenarios()
        completed = 0
        for sid in _ALL_SCENARIOS:
            comparison = _run(comparator.compare(scenarios[sid]))
            if comparison.naive_completed:
                completed += 1
        assert completed == 1

    def test_reliable_completion_count(self):
        """Reliable runner completes 7 of 8 scenarios.

        Completes: clean_success, transient_failure, timeout,
        checkpoint_recovery, loop_replan, context_growth,
        large_output_externalization.
        Does NOT complete: permanent_failure (correctly no-retry).

        Derived from actual comparison results, not hardcoded.
        """
        comparator = _make_comparator()
        scenarios = make_standard_scenarios()
        completed = 0
        for sid in _ALL_SCENARIOS:
            comparison = _run(comparator.compare(scenarios[sid]))
            if comparison.reliable_completed:
                completed += 1
        assert completed == 7

    def test_reliable_advantage_count(self):
        """Reliable absolute completion advantage is 6 scenarios.

        Advantage scenarios (Reliable PASS, Naive FAIL):
        transient_failure, timeout, checkpoint_recovery, loop_replan,
        context_growth, large_output_externalization.

        Derived from actual comparison results, not hardcoded.
        """
        comparator = _make_comparator()
        scenarios = make_standard_scenarios()
        advantages = 0
        for sid in _ALL_SCENARIOS:
            comparison = _run(comparator.compare(scenarios[sid]))
            if comparison.reliable_advantage:
                advantages += 1
        assert advantages == 6

    def test_matrix_aggregate_cross_check(self):
        """Cross-check: sum of matrix rows == aggregate count.

        This invariant ensures the per-scenario matrix and the
        aggregate count can never silently diverge.
        """
        comparator = _make_comparator()
        scenarios = make_standard_scenarios()

        matrix_naive_completed = 0
        matrix_reliable_completed = 0
        matrix_advantage = 0
        for sid in _ALL_SCENARIOS:
            comparison = _run(comparator.compare(scenarios[sid]))
            if comparison.naive_completed:
                matrix_naive_completed += 1
            if comparison.reliable_completed:
                matrix_reliable_completed += 1
            if comparison.reliable_advantage:
                matrix_advantage += 1

        # The aggregate counts must match the matrix-derived counts.
        assert matrix_naive_completed == 1
        assert matrix_reliable_completed == 7
        assert matrix_advantage == 6

        # Advantage == reliable_completed - naive_completed (when
        # every naive completion is also a reliable completion, which
        # holds for this controlled matrix).
        assert matrix_advantage == matrix_reliable_completed - matrix_naive_completed

    def test_permanent_failure_remains_reliable_failure(self):
        """permanent_failure: Reliable must NOT complete (safety control).

        This ensures the aggregate count is not inflated by incorrectly
        retrying a permanent failure.
        """
        scenarios = make_standard_scenarios()
        s = scenarios["permanent_failure"]
        comparator = _make_comparator()
        comparison = _run(comparator.compare(s))
        assert comparison.reliable_completed is False
        assert comparison.naive_completed is False
