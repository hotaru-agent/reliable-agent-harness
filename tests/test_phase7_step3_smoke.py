"""Phase 7 Step 3 — Recovery scenario smoke test.

Runs the checkpoint_recovery scenario through both runners and
verifies the controlled matrix now includes the recovery row:

    Scenario              Naive       Reliable
    clean_success         PASS        PASS
    transient_failure     FAIL        PASS
    timeout               FAIL        PASS
    permanent_failure     FAIL        FAIL
    checkpoint_recovery   FAIL        PASS

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


# The five formal comparison scenarios (Step 2 four + Step 3 recovery).
_FORMAL_SCENARIOS = [
    "clean_success",
    "transient_failure",
    "timeout",
    "permanent_failure",
    "checkpoint_recovery",
]


class TestRecoveryScenarioMatrix:
    """The recovery scenario in the controlled matrix."""

    def test_checkpoint_recovery_naive_fails(self):
        comparator = _make_comparator()
        comparison = _run(
            comparator.compare(make_standard_scenarios()["checkpoint_recovery"])
        )
        assert comparison.naive_completed is False
        assert comparison.naive_record.failure_reason == "INTERRUPTED"
        assert comparison.naive_record.resume_count == 0

    def test_checkpoint_recovery_reliable_completes(self):
        comparator = _make_comparator()
        comparison = _run(
            comparator.compare(make_standard_scenarios()["checkpoint_recovery"])
        )
        assert comparison.reliable_completed is True
        assert comparison.reliable_record.resume_count == 1
        assert comparison.reliable_record.failure_reason is None

    def test_checkpoint_recovery_reliable_advantage(self):
        comparator = _make_comparator()
        comparison = _run(
            comparator.compare(make_standard_scenarios()["checkpoint_recovery"])
        )
        assert comparison.reliable_advantage is True


class TestFullMatrixUnchanged:
    """The full five-scenario matrix — actual execution results."""

    def test_full_matrix_table(self):
        comparator = _make_comparator()
        scenarios = make_standard_scenarios()
        results = {}
        for sid in _FORMAL_SCENARIOS:
            comparison = _run(comparator.compare(scenarios[sid]))
            results[sid] = (
                comparison.naive_completed,
                comparison.reliable_completed,
            )

        assert results["clean_success"] == (True, True)
        assert results["transient_failure"] == (False, True)
        assert results["timeout"] == (False, True)
        assert results["permanent_failure"] == (False, False)
        assert results["checkpoint_recovery"] == (False, True)

    def test_naive_completion_count(self):
        """Naive completes 1 of 5 scenarios (clean only)."""
        comparator = _make_comparator()
        scenarios = make_standard_scenarios()
        completed = 0
        for sid in _FORMAL_SCENARIOS:
            comparison = _run(comparator.compare(scenarios[sid]))
            if comparison.naive_completed:
                completed += 1
        assert completed == 1

    def test_reliable_completion_count(self):
        """Reliable completes 4 of 5 scenarios (clean, transient, timeout,
        recovery — not permanent)."""
        comparator = _make_comparator()
        scenarios = make_standard_scenarios()
        completed = 0
        for sid in _FORMAL_SCENARIOS:
            comparison = _run(comparator.compare(scenarios[sid]))
            if comparison.reliable_completed:
                completed += 1
        assert completed == 4

    def test_reliable_advantage_count(self):
        """Reliable gains advantage on 3 scenarios (transient, timeout,
        recovery). Not on clean (both pass) or permanent (both fail)."""
        comparator = _make_comparator()
        scenarios = make_standard_scenarios()
        advantages = 0
        for sid in _FORMAL_SCENARIOS:
            comparison = _run(comparator.compare(scenarios[sid]))
            if comparison.reliable_advantage:
                advantages += 1
        assert advantages == 3


class TestRecoveryOrderIndependence:
    """Recovery results are order-independent."""

    def test_reversed_order_same_recovery(self):
        comparator = _make_comparator()
        scenario = make_standard_scenarios()["checkpoint_recovery"]
        c1 = _run(comparator.compare(scenario))
        c2 = _run(comparator.compare_reversed(scenario))
        assert c1.naive_completed == c2.naive_completed
        assert c1.reliable_completed == c2.reliable_completed
        assert c1.reliable_record.resume_count == c2.reliable_record.resume_count
