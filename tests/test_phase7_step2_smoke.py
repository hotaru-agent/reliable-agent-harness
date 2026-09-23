"""Phase 7 Step 2 — Scenario matrix comparison smoke test.

Runs the four formal comparison scenarios (clean, transient, timeout,
permanent) through both NaiveBenchmarkRunner and
ReliableHarnessBenchmarkRunner and verifies the expected controlled
matrix:

    Scenario            Naive       Reliable
    clean_success       PASS        PASS
    transient_failure   FAIL        PASS
    timeout             FAIL        PASS
    permanent_failure   FAIL        FAIL

This is the controlled mechanism validation result. It is NOT a
real-world benchmark claim.
"""

from __future__ import annotations

import asyncio

import pytest

from evaluation.comparison import BenchmarkComparator
from evaluation.runners import NaiveBenchmarkRunner, ReliableHarnessBenchmarkRunner
from evaluation.scenarios.controlled_repo import make_standard_scenarios


def _run(coro):
    return asyncio.run(coro)


_TIMEOUT_SECONDS = 0.01


class _FakeSleeper:
    def __init__(self):
        self.sleeps: list[float] = []

    async def __call__(self, seconds: float):
        self.sleeps.append(seconds)


def _make_comparator():
    return BenchmarkComparator(
        NaiveBenchmarkRunner(
            run_tests_timeout_seconds=_TIMEOUT_SECONDS,
            sleep=_FakeSleeper(),
        ),
        ReliableHarnessBenchmarkRunner(
            run_tests_timeout_seconds=_TIMEOUT_SECONDS,
            sleep=_FakeSleeper(),
        ),
    )


# The four formal comparison scenarios for Phase 7 Step 2.
_FORMAL_SCENARIOS = [
    "clean_success",
    "transient_failure",
    "timeout",
    "permanent_failure",
]


class TestScenarioMatrix:
    """The controlled scenario matrix — actual execution results, not
    hardcoded."""

    @pytest.mark.parametrize("scenario_id", _FORMAL_SCENARIOS)
    def test_comparison_produces_records(self, scenario_id):
        """Each scenario produces a valid comparison with both records."""
        comparator = _make_comparator()
        scenario = make_standard_scenarios()[scenario_id]
        comparison = _run(comparator.compare(scenario))
        assert comparison.scenario_id == scenario_id
        assert comparison.naive_record is not None
        assert comparison.reliable_record is not None

    def test_clean_success_both_complete(self):
        comparator = _make_comparator()
        comparison = _run(
            comparator.compare(make_standard_scenarios()["clean_success"])
        )
        assert comparison.naive_completed is True
        assert comparison.reliable_completed is True

    def test_transient_failure_naive_fails_reliable_completes(self):
        comparator = _make_comparator()
        comparison = _run(
            comparator.compare(make_standard_scenarios()["transient_failure"])
        )
        assert comparison.naive_completed is False
        assert comparison.reliable_completed is True
        assert comparison.reliable_advantage is True

    def test_timeout_naive_fails_reliable_completes(self):
        comparator = _make_comparator()
        comparison = _run(
            comparator.compare(make_standard_scenarios()["timeout"])
        )
        assert comparison.naive_completed is False
        assert comparison.reliable_completed is True
        assert comparison.reliable_advantage is True

    def test_permanent_failure_both_fail(self):
        comparator = _make_comparator()
        comparison = _run(
            comparator.compare(make_standard_scenarios()["permanent_failure"])
        )
        assert comparison.naive_completed is False
        assert comparison.reliable_completed is False
        assert comparison.reliable_advantage is False
        # No unsafe retry on permanent failure.
        assert comparison.reliable_record.retry_count == 0


class TestControlledMatrixSummary:
    """The first quantitative comparison — a deterministic, controlled
    result. NOT a real-world completion rate claim."""

    def test_naive_completion_count(self):
        """Naive runner completes 1 of 4 scenarios (clean only)."""
        comparator = _make_comparator()
        scenarios = make_standard_scenarios()
        completed = 0
        for sid in _FORMAL_SCENARIOS:
            comparison = _run(comparator.compare(scenarios[sid]))
            if comparison.naive_completed:
                completed += 1
        assert completed == 1

    def test_reliable_completion_count(self):
        """Reliable runner completes 3 of 4 scenarios (clean, transient,
        timeout — not permanent)."""
        comparator = _make_comparator()
        scenarios = make_standard_scenarios()
        completed = 0
        for sid in _FORMAL_SCENARIOS:
            comparison = _run(comparator.compare(scenarios[sid]))
            if comparison.reliable_completed:
                completed += 1
        assert completed == 3

    def test_reliable_advantage_count(self):
        """Reliable runner gains an advantage on 2 scenarios (transient,
        timeout). Not on clean (both pass) or permanent (both fail)."""
        comparator = _make_comparator()
        scenarios = make_standard_scenarios()
        advantages = 0
        for sid in _FORMAL_SCENARIOS:
            comparison = _run(comparator.compare(scenarios[sid]))
            if comparison.reliable_advantage:
                advantages += 1
        assert advantages == 2

    def test_matrix_table(self):
        """The full controlled matrix — actual execution results."""
        comparator = _make_comparator()
        scenarios = make_standard_scenarios()
        results = {}
        for sid in _FORMAL_SCENARIOS:
            comparison = _run(comparator.compare(scenarios[sid]))
            results[sid] = (
                comparison.naive_completed,
                comparison.reliable_completed,
            )

        # The controlled matrix (derived from actual execution, not hardcoded
        # in the runner):
        assert results["clean_success"] == (True, True)
        assert results["transient_failure"] == (False, True)
        assert results["timeout"] == (False, True)
        assert results["permanent_failure"] == (False, False)


class TestOrderIndependenceSmoke:
    """Running naive→reliable vs reliable→naive produces the same matrix."""

    def test_reversed_order_same_matrix(self):
        comparator = _make_comparator()
        scenarios = make_standard_scenarios()
        for sid in _FORMAL_SCENARIOS:
            c1 = _run(comparator.compare(scenarios[sid]))
            c2 = _run(comparator.compare_reversed(scenarios[sid]))
            assert c1.naive_completed == c2.naive_completed
            assert c1.reliable_completed == c2.reliable_completed
            assert c1.naive_record.failure_reason == c2.naive_record.failure_reason
            assert c1.reliable_record.failure_reason == c2.reliable_record.failure_reason
