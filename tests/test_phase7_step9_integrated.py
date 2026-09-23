"""Phase 7 Step 9 tests — integrated multi-fault filesystem benchmark.

Verifies that a single Reliable trial can sequentially encounter:
- TRANSIENT_FAILURE (run_tests logical #1) → RetryPolicy retry
- TIMEOUT (run_tests logical #2) → real subprocess timeout + cleanup + retry
- PROCESS_INTERRUPTION (read_file logical #3) → checkpoint/resume

and complete the repository repair with an independent pytest oracle,
while Naive fails on the first transient fault.

All offline, deterministic, no LLM, no network. Real pytest subprocess.
"""

from __future__ import annotations

import asyncio

import pytest

from evaluation.records import BenchmarkEventType
from evaluation.scenarios.filesystem_scenarios import (
    make_filesystem_integrated_multifault_actions,
    make_filesystem_integrated_multifault_scenario,
)
from evaluation.suites import (
    CONTROLLED_V1,
    FILESYSTEM_PYTEST_V1,
    INTEGRATED_FILESYSTEM_V1,
)


# Per-scenario timeout override: the timeout fault needs a short
# ToolSpec timeout so the ToolRuntime fires before the artificial
# subprocess delay completes. Using 2.0s (delay is 3.0s) for a
# safer margin than the isolated timeout scenario's 1.0s, since
# the integrated trial runs more pytest invocations under load.
_INTEGRATED_TIMEOUT_OVERRIDES = {"filesystem_integrated_multifault": 2.0}

# Per-scenario action factory override: the integrated scenario uses
# a 10-action two-bug script.
_INTEGRATED_ACTION_FACTORIES = {
    "filesystem_integrated_multifault": make_filesystem_integrated_multifault_actions,
}


def _run(coro):
    return asyncio.run(coro)


def _make_integrated_runners():
    from evaluation.filesystem_runners import (
        FilesystemNaiveBenchmarkRunner,
        FilesystemReliableHarnessBenchmarkRunner,
    )
    return (
        FilesystemNaiveBenchmarkRunner(
            scenario_timeout_overrides=_INTEGRATED_TIMEOUT_OVERRIDES,
            scenario_action_factories=_INTEGRATED_ACTION_FACTORIES,
        ),
        FilesystemReliableHarnessBenchmarkRunner(
            scenario_timeout_overrides=_INTEGRATED_TIMEOUT_OVERRIDES,
            scenario_action_factories=_INTEGRATED_ACTION_FACTORIES,
        ),
    )


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


def _make_controlled_runners():
    from evaluation.runners import (
        NaiveBenchmarkRunner,
        ReliableHarnessBenchmarkRunner,
    )
    return NaiveBenchmarkRunner(), ReliableHarnessBenchmarkRunner()


# ===========================================================================
# Suite separation
# ===========================================================================


class TestIntegratedSuiteSeparation:
    def test_integrated_suite_exists(self):
        assert INTEGRATED_FILESYSTEM_V1.suite_id == "integrated_filesystem_v1"

    def test_integrated_suite_size(self):
        assert INTEGRATED_FILESYSTEM_V1.size == 1

    def test_integrated_scenario_in_suite(self):
        assert INTEGRATED_FILESYSTEM_V1.contains("filesystem_integrated_multifault")

    def test_integrated_not_in_controlled(self):
        for sid in INTEGRATED_FILESYSTEM_V1.scenario_ids:
            assert not CONTROLLED_V1.contains(sid)

    def test_integrated_not_in_filesystem_pytest(self):
        for sid in INTEGRATED_FILESYSTEM_V1.scenario_ids:
            assert not FILESYSTEM_PYTEST_V1.contains(sid)

    def test_all_suites_disjoint(self):
        c = set(CONTROLLED_V1.scenario_ids)
        f = set(FILESYSTEM_PYTEST_V1.scenario_ids)
        i = set(INTEGRATED_FILESYSTEM_V1.scenario_ids)
        assert c.isdisjoint(f)
        assert c.isdisjoint(i)
        assert f.isdisjoint(i)


# ===========================================================================
# Scenario definition
# ===========================================================================


class TestIntegratedScenarioDefinition:
    def test_scenario_exists(self):
        s = make_filesystem_integrated_multifault_scenario()
        assert s.scenario_id == "filesystem_integrated_multifault"

    def test_uses_two_bug_fixture(self):
        s = make_filesystem_integrated_multifault_scenario()
        assert s.fixture_id == "filesystem_calculator_two_bugs"

    def test_uses_multifault_plan(self):
        s = make_filesystem_integrated_multifault_scenario()
        assert s.fault_plan_id == "filesystem_integrated_multifault"

    def test_script_has_10_actions(self):
        actions = make_filesystem_integrated_multifault_actions()
        assert len(actions) == 10

    def test_fault_plan_has_three_faults(self):
        from evaluation.scenarios.filesystem_scenarios import (
            make_filesystem_standard_fault_plans,
        )
        plans = make_filesystem_standard_fault_plans()
        plan = plans["filesystem_integrated_multifault"]
        assert len(plan.faults) == 3

    def test_fault_plan_has_transient_timeout_interruption(self):
        from evaluation.faults import FaultType
        from evaluation.scenarios.filesystem_scenarios import (
            make_filesystem_standard_fault_plans,
        )
        plans = make_filesystem_standard_fault_plans()
        plan = plans["filesystem_integrated_multifault"]
        fault_types = [f.fault_type for f in plan.faults]
        assert FaultType.TRANSIENT_FAILURE in fault_types
        assert FaultType.TIMEOUT in fault_types
        assert FaultType.PROCESS_INTERRUPTION in fault_types


# ===========================================================================
# Naive behavior
# ===========================================================================


class TestIntegratedNaive:
    def test_naive_fails(self):
        s = make_filesystem_integrated_multifault_scenario()
        naive, _ = _make_integrated_runners()
        rec = _run(naive.run(s))
        assert not rec.completed

    def test_naive_failure_reason_transient(self):
        s = make_filesystem_integrated_multifault_scenario()
        naive, _ = _make_integrated_runners()
        rec = _run(naive.run(s))
        assert rec.failure_reason == "TRANSIENT"

    def test_naive_logical_action_count(self):
        s = make_filesystem_integrated_multifault_scenario()
        naive, _ = _make_integrated_runners()
        rec = _run(naive.run(s))
        assert rec.logical_action_count == 2

    def test_naive_no_retries(self):
        s = make_filesystem_integrated_multifault_scenario()
        naive, _ = _make_integrated_runners()
        rec = _run(naive.run(s))
        assert rec.retry_count == 0

    def test_naive_no_resume(self):
        s = make_filesystem_integrated_multifault_scenario()
        naive, _ = _make_integrated_runners()
        rec = _run(naive.run(s))
        assert rec.resume_count == 0

    def test_naive_no_checkpoints(self):
        s = make_filesystem_integrated_multifault_scenario()
        naive, _ = _make_integrated_runners()
        rec = _run(naive.run(s))
        assert rec.checkpoint_count == 0

    def test_naive_oracle_failed(self):
        s = make_filesystem_integrated_multifault_scenario()
        naive, _ = _make_integrated_runners()
        _run(naive.run(s))
        assert not naive.last_trial_state.oracle_result.passed

    def test_naive_repo_unrepaired(self):
        s = make_filesystem_integrated_multifault_scenario()
        naive, _ = _make_integrated_runners()
        _run(naive.run(s))
        snap = naive.last_trial_state.final_snapshot
        assert "return a - b" in snap["calculator.py"]


# ===========================================================================
# Reliable behavior
# ===========================================================================


class TestIntegratedReliable:
    def test_reliable_passes(self):
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        rec = _run(reliable.run(s))
        assert rec.completed

    def test_reliable_no_failure_reason(self):
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        rec = _run(reliable.run(s))
        assert rec.failure_reason is None

    def test_reliable_retry_count(self):
        """Two internal retries: transient + timeout."""
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        rec = _run(reliable.run(s))
        assert rec.retry_count == 2

    def test_reliable_resume_count(self):
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        rec = _run(reliable.run(s))
        assert rec.resume_count == 1

    def test_reliable_logical_action_count(self):
        """10 actions + 1 legal replay of interrupted action = 11."""
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        rec = _run(reliable.run(s))
        assert rec.logical_action_count == 11

    def test_reliable_tool_invocation_count(self):
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        rec = _run(reliable.run(s))
        assert rec.tool_invocation_count == 11

    def test_reliable_tool_attempt_count(self):
        """11 invocations + 2 internal retries = 13 attempts."""
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        rec = _run(reliable.run(s))
        assert rec.tool_attempt_count == 13

    def test_reliable_checkpoint_count(self):
        """Source: 6 (actions 0-5). Resume: 5 (actions 6-9 + COMPLETE). Total=11."""
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        rec = _run(reliable.run(s))
        assert rec.checkpoint_count == 11

    def test_reliable_oracle_passed(self):
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        _run(reliable.run(s))
        assert reliable.last_trial_state.oracle_result.passed

    def test_reliable_final_snapshot_repaired(self):
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        _run(reliable.run(s))
        snap = reliable.last_trial_state.final_snapshot
        assert "return a + b" in snap["calculator.py"]
        assert "return a * b" in snap["calculator.py"]


# ===========================================================================
# Fault consumption
# ===========================================================================


class TestIntegratedFaultConsumption:
    def test_all_three_faults_fired(self):
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        _run(reliable.run(s))
        inj = reliable.last_trial_state.injector
        assert len(inj.fired_fault_indices) == 3

    def test_run_tests_invocation_count(self):
        """run_tests called: source #1 (action 1), source #2 (action 5),
        resume #3 (action 9). Total = 3."""
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        _run(reliable.run(s))
        inj = reliable.last_trial_state.injector
        assert inj._invocation_counts.get("run_tests") == 3

    def test_read_file_invocation_count(self):
        """read_file called: source #1 (action 2), source #2 (action 3),
        source #3 (action 6, interrupted), resume #4 (action 6 replay),
        resume #5 (action 7). Total = 5."""
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        _run(reliable.run(s))
        inj = reliable.last_trial_state.injector
        assert inj._invocation_counts.get("read_file") == 5


# ===========================================================================
# Run lineage
# ===========================================================================


class TestIntegratedRunLineage:
    def test_source_run_interrupted(self):
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        _run(reliable.run(s))
        from harness.state import RunStatus
        assert reliable.last_trial_state.source_run.status == RunStatus.INTERRUPTED

    def test_resume_run_completed(self):
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        _run(reliable.run(s))
        from harness.state import RunStatus
        assert reliable.last_trial_state.resume_run.status == RunStatus.COMPLETED

    def test_different_run_ids(self):
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        _run(reliable.run(s))
        sr = reliable.last_trial_state.source_run
        rr = reliable.last_trial_state.resume_run
        assert sr.run_id != rr.run_id

    def test_interruption_occurred(self):
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        _run(reliable.run(s))
        assert reliable.last_trial_state.interruption_occurred


# ===========================================================================
# Process accounting
# ===========================================================================


class TestIntegratedProcessAccounting:
    def test_spawn_count(self):
        """Agent spawns: transient retry (action 1 attempt 2), timeout
        attempt (action 5 attempt 1, cancelled), timeout retry (action 5
        attempt 2), final run_tests (action 9). = 4."""
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        _run(reliable.run(s))
        repo = reliable.last_trial_state.repo
        assert repo.agent_pytest_spawn_count == 4

    def test_completed_count(self):
        """Agent completed: transient retry, timeout retry, final. = 3."""
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        _run(reliable.run(s))
        repo = reliable.last_trial_state.repo
        assert repo.agent_pytest_completed_count == 3

    def test_no_active_process(self):
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        _run(reliable.run(s))
        repo = reliable.last_trial_state.repo
        assert repo.active_process_count == 0

    def test_oracle_count(self):
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        _run(reliable.run(s))
        repo = reliable.last_trial_state.repo
        assert repo.oracle_pytest_run_count == 1


# ===========================================================================
# Source/resume executor history
# ===========================================================================


class TestIntegratedExecutorHistory:
    def test_source_executor_actions(self):
        """Source: actions 0-5 succeed + action 6 interrupted = 7."""
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        _run(reliable.run(s))
        se = reliable.last_trial_state.source_executor
        assert se.logical_action_count == 7

    def test_resume_executor_actions(self):
        """Resume: action 6 replay + actions 7-9 = 4."""
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        _run(reliable.run(s))
        re = reliable.last_trial_state.resume_executor
        assert re.logical_action_count == 4


# ===========================================================================
# Checkpoint state
# ===========================================================================


class TestIntegratedCheckpointState:
    def test_checkpoint_store_exists(self):
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        _run(reliable.run(s))
        assert reliable.last_trial_state.checkpoint_store is not None

    def test_source_last_checkpoint_not_none(self):
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        _run(reliable.run(s))
        assert reliable.last_trial_state.source_run.last_checkpoint_id is not None


# ===========================================================================
# No cache leak
# ===========================================================================


class TestIntegratedNoCacheLeak:
    def test_no_cache_in_snapshot(self):
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        _run(reliable.run(s))
        snap = reliable.last_trial_state.final_snapshot
        for key in snap:
            assert ".pytest_cache" not in key
            assert "__pycache__" not in key
            assert not key.endswith(".pyc")


# ===========================================================================
# Order independence
# ===========================================================================


class TestIntegratedOrderIndependence:
    def test_naive_first_then_reliable(self):
        s = make_filesystem_integrated_multifault_scenario()
        naive, reliable = _make_integrated_runners()
        n_rec = _run(naive.run(s))
        r_rec = _run(reliable.run(s))
        assert not n_rec.completed
        assert r_rec.completed

    def test_reliable_first_then_naive(self):
        s = make_filesystem_integrated_multifault_scenario()
        naive, reliable = _make_integrated_runners()
        r_rec = _run(reliable.run(s))
        n_rec = _run(naive.run(s))
        assert r_rec.completed
        assert not n_rec.completed


# ===========================================================================
# Fresh-state isolation
# ===========================================================================


class TestIntegratedFreshStateIsolation:
    def test_two_consecutive_reliable_runs(self):
        s = make_filesystem_integrated_multifault_scenario()
        for _ in range(2):
            _, reliable = _make_integrated_runners()
            rec = _run(reliable.run(s))
            assert rec.completed
            assert rec.retry_count == 2
            assert rec.resume_count == 1
            inj = reliable.last_trial_state.injector
            assert len(inj.fired_fault_indices) == 3

    def test_fresh_source_run_each_time(self):
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable1 = _make_integrated_runners()
        _run(reliable1.run(s))
        sr1 = reliable1.last_trial_state.source_run
        _, reliable2 = _make_integrated_runners()
        _run(reliable2.run(s))
        sr2 = reliable2.last_trial_state.source_run
        assert sr1 is not sr2
        from harness.state import RunStatus
        assert sr1.status == RunStatus.INTERRUPTED
        assert sr2.status == RunStatus.INTERRUPTED


# ===========================================================================
# Determinism
# ===========================================================================


class TestIntegratedDeterminism:
    def test_repeated_reliable_consistent(self):
        s = make_filesystem_integrated_multifault_scenario()
        results = []
        for _ in range(2):
            _, reliable = _make_integrated_runners()
            rec = _run(reliable.run(s))
            results.append((
                rec.completed,
                rec.logical_action_count,
                rec.tool_invocation_count,
                rec.tool_attempt_count,
                rec.retry_count,
                rec.checkpoint_count,
                rec.resume_count,
            ))
        assert results[0] == results[1]

    def test_repeated_reliable_snapshot_consistent(self):
        s = make_filesystem_integrated_multifault_scenario()
        snapshots = []
        for _ in range(2):
            _, reliable = _make_integrated_runners()
            _run(reliable.run(s))
            snapshots.append(reliable.last_trial_state.final_snapshot)
        assert snapshots[0] == snapshots[1]


# ===========================================================================
# Integrated suite aggregate
# ===========================================================================


class TestIntegratedSuiteAggregate:
    def test_naive_completion_count(self):
        """Naive completes 0 of 1."""
        s = make_filesystem_integrated_multifault_scenario()
        naive, _ = _make_integrated_runners()
        rec = _run(naive.run(s))
        assert not rec.completed

    def test_reliable_completion_count(self):
        """Reliable completes 1 of 1."""
        s = make_filesystem_integrated_multifault_scenario()
        _, reliable = _make_integrated_runners()
        rec = _run(reliable.run(s))
        assert rec.completed

    def test_advantage_count(self):
        """Reliable advantage: 1 scenario."""
        s = make_filesystem_integrated_multifault_scenario()
        naive, reliable = _make_integrated_runners()
        n_rec = _run(naive.run(s))
        r_rec = _run(reliable.run(s))
        assert r_rec.completed and not n_rec.completed


# ===========================================================================
# Existing suite regression
# ===========================================================================


_FS_SCENARIOS = [
    "filesystem_pytest_clean",
    "filesystem_pytest_transient",
    "filesystem_pytest_timeout",
    "filesystem_pytest_recovery",
]

_CONTROLLED_SCENARIOS = [
    "clean_success",
    "transient_failure",
    "timeout",
    "permanent_failure",
    "checkpoint_recovery",
    "loop_replan",
    "context_growth",
    "large_output_externalization",
]


class TestExistingFilesystemSuiteRegression:
    def test_filesystem_suite_size_unchanged(self):
        assert FILESYSTEM_PYTEST_V1.size == 4

    def test_filesystem_suite_does_not_contain_integrated(self):
        assert not FILESYSTEM_PYTEST_V1.contains("filesystem_integrated_multifault")

    def test_filesystem_naive_completion_count(self):
        """Naive still completes 1 of 4."""
        from evaluation.scenarios.filesystem_scenarios import (
            make_filesystem_standard_scenarios,
        )
        scenarios = make_filesystem_standard_scenarios()
        completed = 0
        for sid in _FS_SCENARIOS:
            naive, _ = _make_fs_runners()
            rec = _run(naive.run(scenarios[sid]))
            if rec.completed:
                completed += 1
        assert completed == 1

    def test_filesystem_reliable_completion_count(self):
        """Reliable still completes 4 of 4."""
        from evaluation.scenarios.filesystem_scenarios import (
            make_filesystem_standard_scenarios,
        )
        scenarios = make_filesystem_standard_scenarios()
        completed = 0
        for sid in _FS_SCENARIOS:
            _, reliable = _make_fs_runners()
            rec = _run(reliable.run(scenarios[sid]))
            if rec.completed:
                completed += 1
        assert completed == 4


class TestControlledSuiteRegression:
    def test_controlled_suite_size_unchanged(self):
        assert CONTROLLED_V1.size == 8

    def test_controlled_does_not_contain_integrated(self):
        assert not CONTROLLED_V1.contains("filesystem_integrated_multifault")

    def test_controlled_naive_completion_count(self):
        """Controlled Naive still completes 1 of 8."""
        from evaluation.scenarios.controlled_repo import make_standard_scenarios
        scenarios = make_standard_scenarios()
        completed = 0
        for sid in _CONTROLLED_SCENARIOS:
            naive, _ = _make_controlled_runners()
            rec = _run(naive.run(scenarios[sid]))
            if rec.completed:
                completed += 1
        assert completed == 1

    def test_controlled_reliable_completion_count(self):
        """Controlled Reliable still completes 7 of 8."""
        from evaluation.scenarios.controlled_repo import make_standard_scenarios
        scenarios = make_standard_scenarios()
        completed = 0
        for sid in _CONTROLLED_SCENARIOS:
            _, reliable = _make_controlled_runners()
            rec = _run(reliable.run(scenarios[sid]))
            if rec.completed:
                completed += 1
        assert completed == 7
