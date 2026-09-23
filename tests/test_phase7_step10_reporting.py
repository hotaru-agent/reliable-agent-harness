"""Phase 7 Step 10 reporting tests.

Verifies that the final benchmark report data (suite registry,
scenario matrix, metric dictionary, suite aggregates) is
machine-readable, internally consistent, and derived from actual
benchmark execution — NOT hardcoded.

These tests prevent stale accounting drift between the documentation
(`FINAL_BENCHMARK_REPORT.md`) and the actual benchmark results.
"""

from __future__ import annotations

import asyncio

import pytest

from evaluation.report_data import (
    ALL_SUITES,
    METRIC_DICTIONARY,
    SuiteAggregate,
    ScenarioMatrixEntry,
    build_scenario_matrix,
    build_suite_aggregates,
    suites_are_disjoint,
    total_scenario_count,
)
from evaluation.suites import (
    CONTROLLED_V1,
    FILESYSTEM_PYTEST_V1,
    INTEGRATED_FILESYSTEM_V1,
)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helpers to run all scenarios and collect records
# ---------------------------------------------------------------------------

def _make_controlled_runners():
    from evaluation.runners import (
        NaiveBenchmarkRunner,
        ReliableHarnessBenchmarkRunner,
    )
    return NaiveBenchmarkRunner(), ReliableHarnessBenchmarkRunner()


def _make_fs_runners():
    from evaluation.filesystem_runners import (
        FilesystemNaiveBenchmarkRunner,
        FilesystemReliableHarnessBenchmarkRunner,
    )
    from evaluation.scenarios.filesystem_scenarios import (
        make_filesystem_recovery_actions,
    )
    return (
        FilesystemNaiveBenchmarkRunner(
            scenario_timeout_overrides={"filesystem_pytest_timeout": 1.0},
            scenario_action_factories={
                "filesystem_pytest_recovery": make_filesystem_recovery_actions,
            },
        ),
        FilesystemReliableHarnessBenchmarkRunner(
            scenario_timeout_overrides={"filesystem_pytest_timeout": 1.0},
            scenario_action_factories={
                "filesystem_pytest_recovery": make_filesystem_recovery_actions,
            },
        ),
    )


def _make_integrated_runners():
    from evaluation.filesystem_runners import (
        FilesystemNaiveBenchmarkRunner,
        FilesystemReliableHarnessBenchmarkRunner,
    )
    from evaluation.scenarios.filesystem_scenarios import (
        make_filesystem_integrated_multifault_actions,
    )
    return (
        FilesystemNaiveBenchmarkRunner(
            scenario_timeout_overrides={
                "filesystem_integrated_multifault": 2.0,
            },
            scenario_action_factories={
                "filesystem_integrated_multifault": make_filesystem_integrated_multifault_actions,
            },
        ),
        FilesystemReliableHarnessBenchmarkRunner(
            scenario_timeout_overrides={
                "filesystem_integrated_multifault": 2.0,
            },
            scenario_action_factories={
                "filesystem_integrated_multifault": make_filesystem_integrated_multifault_actions,
            },
        ),
    )


def _collect_all_records():
    """Run all scenarios and return (naive_records, reliable_records)."""
    from evaluation.scenarios.controlled_repo import make_standard_scenarios
    from evaluation.scenarios.filesystem_scenarios import (
        make_filesystem_standard_scenarios,
        make_filesystem_integrated_multifault_scenario,
    )
    naive_recs = {}
    reliable_recs = {}

    # Controlled
    controlled_scenarios = make_standard_scenarios()
    n_runner, r_runner = _make_controlled_runners()
    for sid in CONTROLLED_V1.scenario_ids:
        n_rec = _run(n_runner.run(controlled_scenarios[sid]))
        r_rec = _run(r_runner.run(controlled_scenarios[sid]))
        naive_recs[sid] = n_rec
        reliable_recs[sid] = r_rec

    # Filesystem
    fs_scenarios = make_filesystem_standard_scenarios()
    n_runner, r_runner = _make_fs_runners()
    for sid in FILESYSTEM_PYTEST_V1.scenario_ids:
        n_rec = _run(n_runner.run(fs_scenarios[sid]))
        r_rec = _run(r_runner.run(fs_scenarios[sid]))
        naive_recs[sid] = n_rec
        reliable_recs[sid] = r_rec

    # Integrated
    integrated = make_filesystem_integrated_multifault_scenario()
    n_runner, r_runner = _make_integrated_runners()
    n_rec = _run(n_runner.run(integrated))
    r_rec = _run(r_runner.run(integrated))
    naive_recs[integrated.scenario_id] = n_rec
    reliable_recs[integrated.scenario_id] = r_rec

    return naive_recs, reliable_recs


# ---------------------------------------------------------------------------
# Suite registry tests
# ---------------------------------------------------------------------------

class TestSuiteRegistry:
    def test_three_suites(self):
        assert len(ALL_SUITES) == 3

    def test_suite_ids(self):
        ids = {s.suite_id for s in ALL_SUITES}
        assert ids == {"controlled_v1", "filesystem_pytest_v1",
                        "integrated_filesystem_v1"}

    def test_suites_disjoint(self):
        assert suites_are_disjoint()

    def test_controlled_size(self):
        assert CONTROLLED_V1.size == 8

    def test_filesystem_size(self):
        assert FILESYSTEM_PYTEST_V1.size == 4

    def test_integrated_size(self):
        assert INTEGRATED_FILESYSTEM_V1.size == 1

    def test_total_scenario_count(self):
        assert total_scenario_count() == 13


# ---------------------------------------------------------------------------
# Metric dictionary tests
# ---------------------------------------------------------------------------

class TestMetricDictionary:
    def test_all_metrics_defined(self):
        expected = {
            "task_completed", "logical_action_count", "tool_invocation_count",
            "tool_attempt_count", "retry_count", "checkpoint_count",
            "resume_count", "loop_detection_count", "replan_count",
            "blocked_call_count", "processed_output_count",
            "externalized_output_count", "externalized_output_ratio",
            "peak_context_tokens", "wall_clock_seconds",
        }
        assert expected.issubset(set(METRIC_DICTIONARY.keys()))

    def test_each_metric_has_definition(self):
        for name, metric in METRIC_DICTIONARY.items():
            assert metric.name == name
            assert len(metric.definition) > 10
            assert len(metric.not_counted) > 5

    def test_wall_clock_not_deterministic(self):
        assert METRIC_DICTIONARY["wall_clock_seconds"].deterministic is False

    def test_logical_action_count_deterministic(self):
        assert METRIC_DICTIONARY["logical_action_count"].deterministic is True


# ---------------------------------------------------------------------------
# Scenario matrix tests (derived from actual execution)
# ---------------------------------------------------------------------------

class TestScenarioMatrix:
    def test_matrix_has_all_scenarios(self):
        naive_recs, reliable_recs = _collect_all_records()
        matrix = build_scenario_matrix(naive_recs, reliable_recs)
        assert len(matrix) == 13

    def test_matrix_suite_order(self):
        naive_recs, reliable_recs = _collect_all_records()
        matrix = build_scenario_matrix(naive_recs, reliable_recs)
        suite_ids = [e.suite_id for e in matrix]
        assert suite_ids[:8] == ["controlled_v1"] * 8
        assert suite_ids[8:12] == ["filesystem_pytest_v1"] * 4
        assert suite_ids[12:] == ["integrated_filesystem_v1"] * 1

    def test_matrix_entries_have_metadata(self):
        naive_recs, reliable_recs = _collect_all_records()
        matrix = build_scenario_matrix(naive_recs, reliable_recs)
        for entry in matrix:
            assert entry.mechanism != ""
            assert isinstance(entry.real_filesystem, bool)
            assert isinstance(entry.real_subprocess, bool)
            assert isinstance(entry.resume, bool)


# ---------------------------------------------------------------------------
# Suite aggregate tests (derived from actual execution)
# ---------------------------------------------------------------------------

class TestSuiteAggregates:
    def test_controlled_aggregate(self):
        naive_recs, reliable_recs = _collect_all_records()
        aggs = build_suite_aggregates(naive_recs, reliable_recs)
        controlled = [a for a in aggs if a.suite_id == "controlled_v1"][0]
        assert controlled.total_scenarios == 8
        assert controlled.naive_completed == 1
        assert controlled.reliable_completed == 7
        assert controlled.advantage == 6

    def test_filesystem_aggregate(self):
        naive_recs, reliable_recs = _collect_all_records()
        aggs = build_suite_aggregates(naive_recs, reliable_recs)
        fs = [a for a in aggs if a.suite_id == "filesystem_pytest_v1"][0]
        assert fs.total_scenarios == 4
        assert fs.naive_completed == 1
        assert fs.reliable_completed == 4
        assert fs.advantage == 3

    def test_integrated_aggregate(self):
        naive_recs, reliable_recs = _collect_all_records()
        aggs = build_suite_aggregates(naive_recs, reliable_recs)
        integ = [a for a in aggs if a.suite_id == "integrated_filesystem_v1"][0]
        assert integ.total_scenarios == 1
        assert integ.naive_completed == 0
        assert integ.reliable_completed == 1
        assert integ.advantage == 1

    def test_aggregate_matches_matrix(self):
        """Per-suite aggregate must equal sum of matrix outcomes."""
        naive_recs, reliable_recs = _collect_all_records()
        matrix = build_scenario_matrix(naive_recs, reliable_recs)
        aggs = build_suite_aggregates(naive_recs, reliable_recs)
        for agg in aggs:
            entries = [e for e in matrix if e.suite_id == agg.suite_id]
            n_pass = sum(1 for e in entries if e.naive_outcome == "PASS")
            r_pass = sum(1 for e in entries if e.reliable_outcome == "PASS")
            adv = sum(1 for e in entries
                      if e.reliable_outcome == "PASS" and e.naive_outcome == "FAIL")
            assert agg.naive_completed == n_pass
            assert agg.reliable_completed == r_pass
            assert agg.advantage == adv


# ---------------------------------------------------------------------------
# No misleading global rate
# ---------------------------------------------------------------------------

class TestNoGlobalRate:
    def test_suites_not_combined(self):
        """The three suites must NOT be combined into a single rate."""
        naive_recs, reliable_recs = _collect_all_records()
        aggs = build_suite_aggregates(naive_recs, reliable_recs)
        # There must be 3 separate aggregates, not one combined.
        assert len(aggs) == 3
        # Each has its own suite_id.
        assert len({a.suite_id for a in aggs}) == 3
