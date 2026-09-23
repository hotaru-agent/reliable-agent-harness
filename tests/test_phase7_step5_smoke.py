"""Phase 7 Step 5 smoke tests — scenario matrix and aggregate accounting.

Verifies the full 8-scenario matrix including the two new context
scenarios, and aggregate completion counts derived from actual
comparison results.

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


class TestScenarioExistence:
    def test_context_growth_exists(self):
        scenarios = make_standard_scenarios()
        assert "context_growth" in scenarios

    def test_large_output_externalization_exists(self):
        scenarios = make_standard_scenarios()
        assert "large_output_externalization" in scenarios

    def test_all_eight_scenarios_present(self):
        scenarios = make_standard_scenarios()
        expected = {
            "clean_success",
            "transient_failure",
            "permanent_failure",
            "timeout",
            "large_output",
            "interruption",
            "checkpoint_recovery",
            "loop_replan",
            "context_growth",
            "large_output_externalization",
        }
        assert set(scenarios.keys()) == expected


# ===========================================================================
# Context growth scenario matrix
# ===========================================================================


class TestContextGrowthMatrix:
    def test_naive_fail(self):
        scenarios = make_standard_scenarios()
        naive, _ = _make_runners()
        rec = _run(naive.run(scenarios["context_growth"]))
        assert not rec.completed
        assert rec.failure_reason == "CONTEXT_LIMIT"

    def test_reliable_pass(self):
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["context_growth"]))
        assert rec.completed
        assert rec.failure_reason is None

    def test_reliable_advantage(self):
        """context_growth: Reliable PASS, Naive FAIL → advantage."""
        scenarios = make_standard_scenarios()
        naive, reliable = _make_runners()
        n_rec = _run(naive.run(scenarios["context_growth"]))
        r_rec = _run(reliable.run(scenarios["context_growth"]))
        assert r_rec.completed and not n_rec.completed


# ===========================================================================
# Large output scenario matrix
# ===========================================================================


class TestLargeOutputMatrix:
    def test_naive_fail(self):
        scenarios = make_standard_scenarios()
        naive, _ = _make_runners()
        rec = _run(naive.run(scenarios["large_output_externalization"]))
        assert not rec.completed
        assert rec.failure_reason == "CONTEXT_LIMIT"

    def test_reliable_pass(self):
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios["large_output_externalization"]))
        assert rec.completed
        assert rec.failure_reason is None

    def test_reliable_advantage(self):
        """large_output_externalization: Reliable PASS, Naive FAIL."""
        scenarios = make_standard_scenarios()
        naive, reliable = _make_runners()
        n_rec = _run(naive.run(scenarios["large_output_externalization"]))
        r_rec = _run(reliable.run(scenarios["large_output_externalization"]))
        assert r_rec.completed and not n_rec.completed


# ===========================================================================
# Old six-scenario matrix unchanged (Section 50)
# ===========================================================================


class TestOldMatrixUnchanged:
    """The original six scenarios must produce the same results."""

    OLD_SCENARIOS = [
        "clean_success",
        "transient_failure",
        "timeout",
        "permanent_failure",
        "checkpoint_recovery",
        "loop_replan",
    ]

    @pytest.mark.parametrize("sid", OLD_SCENARIOS)
    def test_naive_result(self, sid):
        scenarios = make_standard_scenarios()
        naive, _ = _make_runners()
        rec = _run(naive.run(scenarios[sid]))
        expected = sid == "clean_success"
        assert rec.completed == expected

    @pytest.mark.parametrize("sid", OLD_SCENARIOS)
    def test_reliable_result(self, sid):
        scenarios = make_standard_scenarios()
        _, reliable = _make_runners()
        rec = _run(reliable.run(scenarios[sid]))
        # permanent_failure is the only Reliable failure.
        expected = sid != "permanent_failure"
        assert rec.completed == expected


# ===========================================================================
# Aggregate accounting (8 scenarios)
# ===========================================================================


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
    from evaluation.comparison import BenchmarkComparator
    return BenchmarkComparator(
        NaiveBenchmarkRunner(),
        ReliableHarnessBenchmarkRunner(),
    )


class TestAggregateAccounting:
    """Aggregate completion counts for the 8-scenario matrix."""

    def test_total_scenarios(self):
        scenarios = make_standard_scenarios()
        assert len(_ALL_SCENARIOS) == 8
        for sid in _ALL_SCENARIOS:
            assert sid in scenarios

    def test_naive_completion_count(self):
        """Naive completes 1 of 8 (clean_success only)."""
        comparator = _make_comparator()
        scenarios = make_standard_scenarios()
        completed = 0
        for sid in _ALL_SCENARIOS:
            comparison = _run(comparator.compare(scenarios[sid]))
            if comparison.naive_completed:
                completed += 1
        assert completed == 1

    def test_reliable_completion_count(self):
        """Reliable completes 7 of 8 (all except permanent_failure)."""
        comparator = _make_comparator()
        scenarios = make_standard_scenarios()
        completed = 0
        for sid in _ALL_SCENARIOS:
            comparison = _run(comparator.compare(scenarios[sid]))
            if comparison.reliable_completed:
                completed += 1
        assert completed == 7

    def test_reliable_advantage_count(self):
        """Reliable absolute advantage is 6 scenarios."""
        comparator = _make_comparator()
        scenarios = make_standard_scenarios()
        advantages = 0
        for sid in _ALL_SCENARIOS:
            comparison = _run(comparator.compare(scenarios[sid]))
            if comparison.reliable_advantage:
                advantages += 1
        assert advantages == 6

    def test_matrix_aggregate_cross_check(self):
        """Cross-check: matrix rows == aggregate counts."""
        comparator = _make_comparator()
        scenarios = make_standard_scenarios()
        matrix_naive = 0
        matrix_reliable = 0
        matrix_advantage = 0
        for sid in _ALL_SCENARIOS:
            comparison = _run(comparator.compare(scenarios[sid]))
            if comparison.naive_completed:
                matrix_naive += 1
            if comparison.reliable_completed:
                matrix_reliable += 1
            if comparison.reliable_advantage:
                matrix_advantage += 1
        assert matrix_naive == 1
        assert matrix_reliable == 7
        assert matrix_advantage == 6
        assert matrix_advantage == matrix_reliable - matrix_naive

    def test_permanent_failure_remains_reliable_failure(self):
        """permanent_failure: Reliable must NOT complete."""
        scenarios = make_standard_scenarios()
        comparator = _make_comparator()
        comparison = _run(comparator.compare(scenarios["permanent_failure"]))
        assert comparison.reliable_completed is False
        assert comparison.naive_completed is False


# ===========================================================================
# Fairness — same capacity for N/R (Section 6, 55, 56)
# ===========================================================================


class TestFairness:
    def test_context_growth_same_capacity(self):
        """Both runners use the same max_context_tokens."""
        # The ContextWorkload is shared via WorkloadRegistry, so both
        # runners get the same max_context_tokens (120).
        scenarios = make_standard_scenarios()
        naive, reliable = _make_runners()
        n_rec = _run(naive.run(scenarios["context_growth"]))
        r_rec = _run(reliable.run(scenarios["context_growth"]))
        # Naive exceeds 120; Reliable stays <= 120.
        assert n_rec.peak_context_tokens > 120
        assert r_rec.peak_context_tokens <= 120

    def test_large_output_same_capacity(self):
        scenarios = make_standard_scenarios()
        naive, reliable = _make_runners()
        n_rec = _run(naive.run(scenarios["large_output_externalization"]))
        r_rec = _run(reliable.run(scenarios["large_output_externalization"]))
        assert n_rec.peak_context_tokens > 120
        assert r_rec.peak_context_tokens <= 120
