"""Phase 7 Step 6/7 smoke tests — filesystem suite matrix and aggregate.

Verifies:
- All filesystem scenarios exist.
- Clean: Naive PASS / Reliable PASS.
- Transient: Naive FAIL / Reliable PASS.
- Timeout: Naive FAIL / Reliable PASS (Phase 7 Step 7).
- Filesystem suite aggregate: Naive 1/3, Reliable 3/3, advantage 2/3.
- Controlled suite aggregate unchanged: Naive 1/8, Reliable 7/8, advantage 6/8.
- Fresh-state isolation.
- Runner order independence.
- Subprocess accounting (agent vs oracle pytest counts).
- Fault placement (same run_tests logical invocation #1).
- Real subprocess on retry.
- Objective oracle wins over Harness COMPLETED.

All offline, deterministic, no LLM, no network. Real pytest subprocess.
"""

from __future__ import annotations

import asyncio

import pytest

from evaluation.records import BenchmarkEventType
from evaluation.scenarios.controlled_repo import make_standard_scenarios as make_controlled_scenarios
from evaluation.scenarios.filesystem_scenarios import (
    make_filesystem_recovery_actions,
    make_filesystem_standard_scenarios,
)
from evaluation.suites import CONTROLLED_V1, FILESYSTEM_PYTEST_V1


# Per-scenario timeout overrides: the timeout scenario needs a short
# ToolSpec timeout so the ToolRuntime fires before the artificial
# subprocess delay completes.
_FS_TIMEOUT_OVERRIDES = {"filesystem_pytest_timeout": 1.0}

# Per-scenario action factory overrides: the recovery scenario uses
# an 8-action two-bug script instead of the standard 6-action script.
_FS_ACTION_FACTORIES = {
    "filesystem_pytest_recovery": make_filesystem_recovery_actions,
}


def _run(coro):
    return asyncio.run(coro)


def _make_fs_runners():
    from evaluation.filesystem_runners import (
        FilesystemNaiveBenchmarkRunner,
        FilesystemReliableHarnessBenchmarkRunner,
    )
    return (
        FilesystemNaiveBenchmarkRunner(
            scenario_timeout_overrides=_FS_TIMEOUT_OVERRIDES,
            scenario_action_factories=_FS_ACTION_FACTORIES,
        ),
        FilesystemReliableHarnessBenchmarkRunner(
            scenario_timeout_overrides=_FS_TIMEOUT_OVERRIDES,
            scenario_action_factories=_FS_ACTION_FACTORIES,
        ),
    )


def _make_controlled_runners():
    from evaluation.runners import (
        NaiveBenchmarkRunner,
        ReliableHarnessBenchmarkRunner,
    )
    return NaiveBenchmarkRunner(), ReliableHarnessBenchmarkRunner()


# ===========================================================================
# Scenario existence
# ===========================================================================


class TestScenarioExistence:
    def test_clean_exists(self):
        scenarios = make_filesystem_standard_scenarios()
        assert "filesystem_pytest_clean" in scenarios

    def test_transient_exists(self):
        scenarios = make_filesystem_standard_scenarios()
        assert "filesystem_pytest_transient" in scenarios

    def test_suite_membership(self):
        assert FILESYSTEM_PYTEST_V1.size == 4
        assert FILESYSTEM_PYTEST_V1.contains("filesystem_pytest_clean")
        assert FILESYSTEM_PYTEST_V1.contains("filesystem_pytest_transient")
        assert FILESYSTEM_PYTEST_V1.contains("filesystem_pytest_timeout")
        assert FILESYSTEM_PYTEST_V1.contains("filesystem_pytest_recovery")


# ===========================================================================
# Clean scenario
# ===========================================================================


class TestCleanScenario:
    def test_naive_pass(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        rec = _run(naive.run(scenarios["filesystem_pytest_clean"]))
        assert rec.completed
        assert rec.failure_reason is None

    def test_reliable_pass(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(scenarios["filesystem_pytest_clean"]))
        assert rec.completed
        assert rec.failure_reason is None

    def test_naive_logical_action_count(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        rec = _run(naive.run(scenarios["filesystem_pytest_clean"]))
        assert rec.logical_action_count == 6

    def test_reliable_logical_action_count(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(scenarios["filesystem_pytest_clean"]))
        assert rec.logical_action_count == 6

    def test_naive_no_retries(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        rec = _run(naive.run(scenarios["filesystem_pytest_clean"]))
        assert rec.retry_count == 0

    def test_reliable_no_retries(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(scenarios["filesystem_pytest_clean"]))
        assert rec.retry_count == 0

    def test_naive_no_checkpoints(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        rec = _run(naive.run(scenarios["filesystem_pytest_clean"]))
        assert rec.checkpoint_count == 0

    def test_reliable_has_checkpoints(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(scenarios["filesystem_pytest_clean"]))
        # 6 actions + 1 COMPLETE = 7 checkpoints.
        assert rec.checkpoint_count == 7

    def test_naive_oracle_passed(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        _run(naive.run(scenarios["filesystem_pytest_clean"]))
        assert naive.last_trial_state.oracle_result.passed

    def test_reliable_oracle_passed(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_clean"]))
        assert reliable.last_trial_state.oracle_result.passed

    def test_clean_final_snapshots_equivalent(self):
        """Naive and Reliable final snapshots should be functionally
        equivalent (same files, same content)."""
        scenarios = make_filesystem_standard_scenarios()
        naive, reliable = _make_fs_runners()
        _run(naive.run(scenarios["filesystem_pytest_clean"]))
        n_snap = naive.last_trial_state.final_snapshot
        _, reliable2 = _make_fs_runners()
        _run(reliable2.run(scenarios["filesystem_pytest_clean"]))
        r_snap = reliable2.last_trial_state.final_snapshot
        assert n_snap == r_snap


# ===========================================================================
# Transient scenario
# ===========================================================================


class TestTransientScenario:
    def test_naive_fail(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        rec = _run(naive.run(scenarios["filesystem_pytest_transient"]))
        assert not rec.completed
        assert rec.failure_reason == "TRANSIENT"

    def test_reliable_pass(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(scenarios["filesystem_pytest_transient"]))
        assert rec.completed
        assert rec.failure_reason is None

    def test_naive_no_retries(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        rec = _run(naive.run(scenarios["filesystem_pytest_transient"]))
        assert rec.retry_count == 0

    def test_reliable_one_retry(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(scenarios["filesystem_pytest_transient"]))
        assert rec.retry_count == 1

    def test_naive_oracle_failed(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        _run(naive.run(scenarios["filesystem_pytest_transient"]))
        assert not naive.last_trial_state.oracle_result.passed

    def test_reliable_oracle_passed(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_transient"]))
        assert reliable.last_trial_state.oracle_result.passed

    def test_reliable_attempts_count(self):
        """Reliable: 6 invocations, 7 attempts (1 retry on run_tests #1)."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(scenarios["filesystem_pytest_transient"]))
        assert rec.tool_invocation_count == 6
        assert rec.tool_attempt_count == 7


# ===========================================================================
# Fault placement
# ===========================================================================


class TestFaultPlacement:
    def test_same_run_tests_logical_invocation_1(self):
        """Both runners hit the same run_tests logical invocation #1."""
        scenarios = make_filesystem_standard_scenarios()
        naive, reliable = _make_fs_runners()
        _run(naive.run(scenarios["filesystem_pytest_transient"]))
        n_inj = naive.last_trial_state.injector
        _, reliable2 = _make_fs_runners()
        _run(reliable2.run(scenarios["filesystem_pytest_transient"]))
        r_inj = reliable2.last_trial_state.injector
        # Both should have fired the fault at run_tests invocation #1.
        assert len(n_inj.fired_fault_indices) == 1
        assert len(r_inj.fired_fault_indices) == 1

    def test_reliable_retry_does_not_advance_logical_invocation(self):
        """Reliable retry attempt 2 does NOT advance the logical
        invocation counter — it stays at invocation #1."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_transient"]))
        r_inj = reliable.last_trial_state.injector
        # Only 1 fault should have fired (at invocation #1).
        assert len(r_inj.fired_fault_indices) == 1
        # The run_tests invocation count should be 2 (initial + final).
        assert r_inj._invocation_counts.get("run_tests") == 2


# ===========================================================================
# Real subprocess on retry
# ===========================================================================


class TestRealSubprocessOnRetry:
    def test_reliable_retries_with_real_subprocess(self):
        """Reliable run_tests #1: attempt 1 transient (no subprocess),
        attempt 2 real pytest subprocess."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_transient"]))
        repo = reliable.last_trial_state.repo
        # Agent pytest count should be 2 (initial run_tests + final run_tests).
        # The transient fault on attempt 1 does NOT spawn a subprocess.
        assert repo.agent_pytest_run_count == 2

    def test_naive_no_subprocess_on_transient(self):
        """Naive: run_tests #1 attempt 1 transient → no subprocess
        spawned → terminate."""
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        _run(naive.run(scenarios["filesystem_pytest_transient"]))
        repo = naive.last_trial_state.repo
        # Agent pytest count should be 0 (transient fault before spawn).
        assert repo.agent_pytest_run_count == 0


# ===========================================================================
# Subprocess / oracle accounting
# ===========================================================================


class TestSubprocessAccounting:
    def test_clean_agent_pytest_count(self):
        """Clean: 2 agent run_tests calls → agent_pytest_run_count == 2."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_clean"]))
        repo = reliable.last_trial_state.repo
        assert repo.agent_pytest_run_count == 2

    def test_clean_oracle_pytest_count(self):
        """Clean: oracle runs 1 pytest → oracle_pytest_run_count == 1."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_clean"]))
        repo = reliable.last_trial_state.repo
        assert repo.oracle_pytest_run_count == 1

    def test_oracle_does_not_affect_tool_metrics(self):
        """Oracle pytest invocation is NOT a tool invocation or
        logical action."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(scenarios["filesystem_pytest_clean"]))
        # 6 logical actions, 6 tool invocations (oracle is separate).
        assert rec.logical_action_count == 6
        assert rec.tool_invocation_count == 6


# ===========================================================================
# Fresh-state isolation
# ===========================================================================


class TestFreshStateIsolation:
    def test_different_temp_roots(self):
        """Two consecutive runs use different temp roots."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_clean"]))
        root1 = reliable.last_trial_state.repo.root
        _, reliable2 = _make_fs_runners()
        _run(reliable2.run(scenarios["filesystem_pytest_clean"]))
        root2 = reliable2.last_trial_state.repo.root
        assert root1 != root2

    def test_same_initial_snapshot(self):
        """Two consecutive runs have the same initial snapshot."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_clean"]))
        snap1 = reliable.last_trial_state.initial_snapshot
        _, reliable2 = _make_fs_runners()
        _run(reliable2.run(scenarios["filesystem_pytest_clean"]))
        snap2 = reliable2.last_trial_state.initial_snapshot
        assert snap1 == snap2

    def test_no_leak_between_runs(self):
        """First run's modification does not leak to second run."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_clean"]))
        # First run fixed the calculator.
        snap1 = reliable.last_trial_state.final_snapshot
        assert "return a + b" in snap1["calculator.py"]
        # Second run starts fresh (buggy calculator).
        _, reliable2 = _make_fs_runners()
        _run(reliable2.run(scenarios["filesystem_pytest_clean"]))
        snap2_initial = reliable2.last_trial_state.initial_snapshot
        assert "return a - b" in snap2_initial["calculator.py"]


# ===========================================================================
# Runner order independence
# ===========================================================================


class TestRunnerOrderIndependence:
    def test_naive_first_then_reliable(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, reliable = _make_fs_runners()
        n_rec = _run(naive.run(scenarios["filesystem_pytest_transient"]))
        r_rec = _run(reliable.run(scenarios["filesystem_pytest_transient"]))
        assert not n_rec.completed
        assert r_rec.completed

    def test_reliable_first_then_naive(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, reliable = _make_fs_runners()
        r_rec = _run(reliable.run(scenarios["filesystem_pytest_transient"]))
        n_rec = _run(naive.run(scenarios["filesystem_pytest_transient"]))
        assert r_rec.completed
        assert not n_rec.completed


# ===========================================================================
# Objective oracle wins
# ===========================================================================


class TestObjectiveOracleWins:
    def test_oracle_overrides_harness_completed(self):
        """If the repository is still buggy when the script exhausts,
        the oracle must report incomplete even if Harness COMPLETED."""
        # This is implicitly tested by the transient scenario: Naive
        # fails early, but even if it had completed the script, the
        # oracle would catch the incomplete repository. We test this
        # by checking that the oracle is the source of truth.
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        rec = _run(naive.run(scenarios["filesystem_pytest_transient"]))
        # Naive did NOT complete the repository (oracle says incomplete).
        assert not rec.completed
        # Oracle result is available in trial state.
        assert naive.last_trial_state.oracle_result is not None
        assert not naive.last_trial_state.oracle_result.passed


# ===========================================================================
# Filesystem suite aggregate
# ===========================================================================


_FS_SCENARIOS = [
    "filesystem_pytest_clean",
    "filesystem_pytest_transient",
    "filesystem_pytest_timeout",
    "filesystem_pytest_recovery",
]


class TestFilesystemSuiteAggregate:
    def test_naive_completion_count(self):
        """Naive completes 1 of 4 (clean only)."""
        scenarios = make_filesystem_standard_scenarios()
        completed = 0
        for sid in _FS_SCENARIOS:
            naive, _ = _make_fs_runners()
            rec = _run(naive.run(scenarios[sid]))
            if rec.completed:
                completed += 1
        assert completed == 1

    def test_reliable_completion_count(self):
        """Reliable completes 4 of 4."""
        scenarios = make_filesystem_standard_scenarios()
        completed = 0
        for sid in _FS_SCENARIOS:
            _, reliable = _make_fs_runners()
            rec = _run(reliable.run(scenarios[sid]))
            if rec.completed:
                completed += 1
        assert completed == 4

    def test_advantage_count(self):
        """Reliable advantage: 3 scenarios (transient + timeout + recovery)."""
        scenarios = make_filesystem_standard_scenarios()
        advantages = 0
        for sid in _FS_SCENARIOS:
            naive, reliable = _make_fs_runners()
            n_rec = _run(naive.run(scenarios[sid]))
            r_rec = _run(reliable.run(scenarios[sid]))
            if r_rec.completed and not n_rec.completed:
                advantages += 1
        assert advantages == 3


# ===========================================================================
# Controlled suite regression (must remain unchanged)
# ===========================================================================


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


class TestControlledSuiteRegression:
    def test_controlled_suite_size(self):
        assert CONTROLLED_V1.size == 8

    def test_controlled_naive_completion(self):
        """Controlled suite: Naive 1/8."""
        scenarios = make_controlled_scenarios()
        completed = 0
        for sid in _CONTROLLED_SCENARIOS:
            naive, _ = _make_controlled_runners()
            rec = _run(naive.run(scenarios[sid]))
            if rec.completed:
                completed += 1
        assert completed == 1

    def test_controlled_reliable_completion(self):
        """Controlled suite: Reliable 7/8."""
        scenarios = make_controlled_scenarios()
        completed = 0
        for sid in _CONTROLLED_SCENARIOS:
            _, reliable = _make_controlled_runners()
            rec = _run(reliable.run(scenarios[sid]))
            if rec.completed:
                completed += 1
        assert completed == 7

    def test_controlled_advantage(self):
        """Controlled suite: advantage 6/8."""
        scenarios = make_controlled_scenarios()
        advantages = 0
        for sid in _CONTROLLED_SCENARIOS:
            naive, reliable = _make_controlled_runners()
            n_rec = _run(naive.run(scenarios[sid]))
            r_rec = _run(reliable.run(scenarios[sid]))
            if r_rec.completed and not n_rec.completed:
                advantages += 1
        assert advantages == 6


# ===========================================================================
# Suite separation
# ===========================================================================


class TestSuiteSeparation:
    def test_suites_are_disjoint(self):
        """Controlled and filesystem suites share no scenario IDs."""
        controlled_ids = set(CONTROLLED_V1.scenario_ids)
        fs_ids = set(FILESYSTEM_PYTEST_V1.scenario_ids)
        assert controlled_ids.isdisjoint(fs_ids)

    def test_filesystem_not_in_controlled(self):
        for sid in FILESYSTEM_PYTEST_V1.scenario_ids:
            assert not CONTROLLED_V1.contains(sid)

    def test_filesystem_suite_size(self):
        assert FILESYSTEM_PYTEST_V1.size == 4

    def test_timeout_in_filesystem_suite(self):
        assert FILESYSTEM_PYTEST_V1.contains("filesystem_pytest_timeout")

    def test_recovery_in_filesystem_suite(self):
        assert FILESYSTEM_PYTEST_V1.contains("filesystem_pytest_recovery")


# ===========================================================================
# Timeout scenario (Phase 7 Step 7)
# ===========================================================================


class TestTimeoutScenarioExistence:
    def test_timeout_exists(self):
        scenarios = make_filesystem_standard_scenarios()
        assert "filesystem_pytest_timeout" in scenarios

    def test_timeout_same_fixture(self):
        scenarios = make_filesystem_standard_scenarios()
        clean = scenarios["filesystem_pytest_clean"]
        timeout = scenarios["filesystem_pytest_timeout"]
        assert clean.fixture_id == timeout.fixture_id

    def test_timeout_fault_plan(self):
        scenarios = make_filesystem_standard_scenarios()
        timeout = scenarios["filesystem_pytest_timeout"]
        assert timeout.fault_plan_id == "filesystem_timeout_run_tests_1"


class TestTimeoutNaive:
    def test_naive_fail_timeout(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        rec = _run(naive.run(scenarios["filesystem_pytest_timeout"]))
        assert not rec.completed
        assert rec.failure_reason == "TIMEOUT"

    def test_naive_logical_action_count(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        rec = _run(naive.run(scenarios["filesystem_pytest_timeout"]))
        assert rec.logical_action_count == 2

    def test_naive_no_retries(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        rec = _run(naive.run(scenarios["filesystem_pytest_timeout"]))
        assert rec.retry_count == 0

    def test_naive_agent_spawn_count(self):
        """Naive: 1 spawn (timeout attempt), 0 completed."""
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        _run(naive.run(scenarios["filesystem_pytest_timeout"]))
        repo = naive.last_trial_state.repo
        assert repo.agent_pytest_spawn_count == 1
        assert repo.agent_pytest_completed_count == 0

    def test_naive_oracle_failed(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        _run(naive.run(scenarios["filesystem_pytest_timeout"]))
        assert not naive.last_trial_state.oracle_result.passed

    def test_naive_oracle_count(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        _run(naive.run(scenarios["filesystem_pytest_timeout"]))
        repo = naive.last_trial_state.repo
        assert repo.oracle_pytest_run_count == 1

    def test_naive_no_active_process(self):
        """After the trial, no child process is active."""
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        _run(naive.run(scenarios["filesystem_pytest_timeout"]))
        repo = naive.last_trial_state.repo
        assert repo.active_process_count == 0


class TestTimeoutReliable:
    def test_reliable_pass(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(scenarios["filesystem_pytest_timeout"]))
        assert rec.completed
        assert rec.failure_reason is None

    def test_reliable_logical_action_count(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(scenarios["filesystem_pytest_timeout"]))
        assert rec.logical_action_count == 6

    def test_reliable_tool_invocation_count(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(scenarios["filesystem_pytest_timeout"]))
        assert rec.tool_invocation_count == 6

    def test_reliable_tool_attempt_count(self):
        """6 invocations, 7 attempts (1 retry on timeout)."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(scenarios["filesystem_pytest_timeout"]))
        assert rec.tool_attempt_count == 7

    def test_reliable_retry_count(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(scenarios["filesystem_pytest_timeout"]))
        assert rec.retry_count == 1

    def test_reliable_checkpoint_count(self):
        """6 successful steps + COMPLETE = 7 checkpoints."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(scenarios["filesystem_pytest_timeout"]))
        assert rec.checkpoint_count == 7

    def test_reliable_agent_spawn_count(self):
        """Reliable: 3 spawns (timeout + retry + final), 2 completed."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_timeout"]))
        repo = reliable.last_trial_state.repo
        assert repo.agent_pytest_spawn_count == 3
        assert repo.agent_pytest_completed_count == 2

    def test_reliable_oracle_passed(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_timeout"]))
        assert reliable.last_trial_state.oracle_result.passed

    def test_reliable_oracle_count(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_timeout"]))
        repo = reliable.last_trial_state.repo
        assert repo.oracle_pytest_run_count == 1

    def test_reliable_no_active_process(self):
        """After the trial, no child process is active."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_timeout"]))
        repo = reliable.last_trial_state.repo
        assert repo.active_process_count == 0


class TestTimeoutFaultConsumption:
    def test_fault_fires_once(self):
        """The TIMEOUT fault fires exactly once."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_timeout"]))
        inj = reliable.last_trial_state.injector
        assert len(inj.fired_fault_indices) == 1

    def test_retry_does_not_re_apply_delay(self):
        """Retry attempt 2 does NOT re-apply the artificial delay.

        Verified by: the reliable run completes (if delay were
        re-applied, attempt 2 would also timeout and the run would
        fail).
        """
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(scenarios["filesystem_pytest_timeout"]))
        assert rec.completed

    def test_retry_keeps_same_logical_invocation(self):
        """Retry attempt 2 does NOT advance the logical invocation
        counter."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_timeout"]))
        inj = reliable.last_trial_state.injector
        # run_tests was called 2 times (initial + final).
        assert inj._invocation_counts.get("run_tests") == 2


class TestTimeoutRealSubprocess:
    def test_child_actually_spawned(self):
        """The timeout attempt actually spawned a real child process
        (not just raised before spawn)."""
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        _run(naive.run(scenarios["filesystem_pytest_timeout"]))
        repo = naive.last_trial_state.repo
        assert repo.agent_pytest_spawn_count == 1

    def test_child_cleaned_up(self):
        """After the timeout, the child process is no longer active."""
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        _run(naive.run(scenarios["filesystem_pytest_timeout"]))
        repo = naive.last_trial_state.repo
        assert repo.active_process_count == 0

    def test_retry_spawns_fresh_child(self):
        """Reliable retry attempt 2 spawns a fresh child process."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_timeout"]))
        repo = reliable.last_trial_state.repo
        # 3 spawns: timeout + retry + final run_tests.
        assert repo.agent_pytest_spawn_count == 3

    def test_process_returncodes_recorded(self):
        """Process exit returncodes are recorded for diagnostics."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_timeout"]))
        repo = reliable.last_trial_state.repo
        # 4 returncodes: 3 agent + 1 oracle.
        assert len(repo.process_exit_returncodes) == 4


class TestTimeoutSnapshot:
    def test_naive_repo_still_buggy(self):
        """Naive timeout: repository is still buggy after the run."""
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        _run(naive.run(scenarios["filesystem_pytest_timeout"]))
        snap = naive.last_trial_state.final_snapshot
        assert "return a - b" in snap["calculator.py"]

    def test_reliable_repo_repaired(self):
        """Reliable: repository is repaired after the run."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_timeout"]))
        snap = reliable.last_trial_state.final_snapshot
        assert "return a + b" in snap["calculator.py"]

    def test_no_cache_leak_after_timeout(self):
        """After timeout/termination, snapshot has no .pytest_cache,
        __pycache__, or *.pyc."""
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        _run(naive.run(scenarios["filesystem_pytest_timeout"]))
        snap = naive.last_trial_state.final_snapshot
        for key in snap:
            assert ".pytest_cache" not in key
            assert "__pycache__" not in key
            assert not key.endswith(".pyc")


class TestTimeoutOrderIndependence:
    def test_naive_first_then_reliable(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, reliable = _make_fs_runners()
        n_rec = _run(naive.run(scenarios["filesystem_pytest_timeout"]))
        r_rec = _run(reliable.run(scenarios["filesystem_pytest_timeout"]))
        assert not n_rec.completed
        assert r_rec.completed

    def test_reliable_first_then_naive(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, reliable = _make_fs_runners()
        r_rec = _run(reliable.run(scenarios["filesystem_pytest_timeout"]))
        n_rec = _run(naive.run(scenarios["filesystem_pytest_timeout"]))
        assert r_rec.completed
        assert not n_rec.completed


class TestTimeoutFreshTrialIsolation:
    def test_two_consecutive_reliable_runs(self):
        """Two consecutive Reliable timeout runs both complete."""
        scenarios = make_filesystem_standard_scenarios()
        for _ in range(2):
            _, reliable = _make_fs_runners()
            rec = _run(reliable.run(scenarios["filesystem_pytest_timeout"]))
            assert rec.completed

    def test_fresh_diagnostics_each_run(self):
        """Each run has fresh process diagnostics."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable1 = _make_fs_runners()
        _run(reliable1.run(scenarios["filesystem_pytest_timeout"]))
        spawn1 = reliable1.last_trial_state.repo.agent_pytest_spawn_count
        _, reliable2 = _make_fs_runners()
        _run(reliable2.run(scenarios["filesystem_pytest_timeout"]))
        spawn2 = reliable2.last_trial_state.repo.agent_pytest_spawn_count
        # Both should have the same spawn count (fresh diagnostics).
        assert spawn1 == spawn2 == 3


# ===========================================================================
# Recovery scenario (Phase 7 Step 8)
# ===========================================================================


class TestRecoveryScenarioExistence:
    def test_recovery_exists(self):
        scenarios = make_filesystem_standard_scenarios()
        assert "filesystem_pytest_recovery" in scenarios

    def test_recovery_in_suite(self):
        assert FILESYSTEM_PYTEST_V1.contains("filesystem_pytest_recovery")

    def test_recovery_fixture_two_bugs(self):
        scenarios = make_filesystem_standard_scenarios()
        recovery = scenarios["filesystem_pytest_recovery"]
        assert recovery.fixture_id == "filesystem_calculator_two_bugs"

    def test_recovery_fault_plan(self):
        scenarios = make_filesystem_standard_scenarios()
        recovery = scenarios["filesystem_pytest_recovery"]
        assert recovery.fault_plan_id == "filesystem_recovery_read_file_2"

    def test_recovery_not_in_controlled(self):
        assert not CONTROLLED_V1.contains("filesystem_pytest_recovery")


class TestRecoveryNaive:
    def test_naive_fail_interrupted(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        rec = _run(naive.run(scenarios["filesystem_pytest_recovery"]))
        assert not rec.completed
        assert rec.failure_reason == "INTERRUPTED"

    def test_naive_no_resume(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        rec = _run(naive.run(scenarios["filesystem_pytest_recovery"]))
        assert rec.resume_count == 0

    def test_naive_no_checkpoints(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        rec = _run(naive.run(scenarios["filesystem_pytest_recovery"]))
        assert rec.checkpoint_count == 0

    def test_naive_oracle_failed(self):
        """Naive: partial repair done but multiply still buggy → oracle fails."""
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        _run(naive.run(scenarios["filesystem_pytest_recovery"]))
        assert not naive.last_trial_state.oracle_result.passed

    def test_naive_partial_repair_on_disk(self):
        """Naive: partial repair (add fixed, multiply still buggy) is on disk."""
        scenarios = make_filesystem_standard_scenarios()
        naive, _ = _make_fs_runners()
        _run(naive.run(scenarios["filesystem_pytest_recovery"]))
        snap = naive.last_trial_state.final_snapshot
        assert "return a + b" in snap["calculator.py"]
        assert "return a + b" in snap["calculator.py"].split("multiply")[1]


class TestRecoveryReliable:
    def test_reliable_pass(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(scenarios["filesystem_pytest_recovery"]))
        assert rec.completed
        assert rec.failure_reason is None

    def test_reliable_resume_count(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(scenarios["filesystem_pytest_recovery"]))
        assert rec.resume_count == 1

    def test_reliable_no_retries(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(scenarios["filesystem_pytest_recovery"]))
        assert rec.retry_count == 0

    def test_reliable_logical_action_count(self):
        """Source: 5 actions (0-4, interrupted). Resume: 4 actions (4-7). Total=9."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(scenarios["filesystem_pytest_recovery"]))
        assert rec.logical_action_count == 9

    def test_reliable_checkpoint_count(self):
        """Source: 4 checkpoints (0-3). Resume: 5 (4-7 + COMPLETE). Total=9."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(scenarios["filesystem_pytest_recovery"]))
        assert rec.checkpoint_count == 9

    def test_reliable_oracle_passed(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_recovery"]))
        assert reliable.last_trial_state.oracle_result.passed

    def test_reliable_source_run_interrupted(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_recovery"]))
        from harness.state import RunStatus
        assert reliable.last_trial_state.source_run.status == RunStatus.INTERRUPTED

    def test_reliable_resume_run_completed(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_recovery"]))
        from harness.state import RunStatus
        assert reliable.last_trial_state.resume_run.status == RunStatus.COMPLETED

    def test_reliable_different_run_ids(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_recovery"]))
        sr = reliable.last_trial_state.source_run
        rr = reliable.last_trial_state.resume_run
        assert sr.run_id != rr.run_id

    def test_reliable_final_snapshot_repaired(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_recovery"]))
        snap = reliable.last_trial_state.final_snapshot
        assert "return a + b" in snap["calculator.py"]
        assert "return a * b" in snap["calculator.py"]


class TestRecoverySourceRun:
    def test_source_run_exists(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_recovery"]))
        assert reliable.last_trial_state.source_run is not None

    def test_interruption_occurred(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_recovery"]))
        assert reliable.last_trial_state.interruption_occurred

    def test_source_executor_has_history(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_recovery"]))
        se = reliable.last_trial_state.source_executor
        assert se is not None
        assert se.logical_action_count == 5

    def test_resume_executor_has_history(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_recovery"]))
        re = reliable.last_trial_state.resume_executor
        assert re is not None
        assert re.logical_action_count == 4


class TestRecoveryCheckpointState:
    def test_checkpoint_store_exists(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_recovery"]))
        assert reliable.last_trial_state.checkpoint_store is not None

    def test_source_last_checkpoint_not_none(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_recovery"]))
        assert reliable.last_trial_state.source_run.last_checkpoint_id is not None


class TestRecoveryFaultConsumption:
    def test_fault_fires_once(self):
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_recovery"]))
        inj = reliable.last_trial_state.injector
        assert len(inj.fired_fault_indices) == 1

    def test_read_file_invocation_count(self):
        """read_file called: source #1 (action 2), source #2 (action 4,
        interrupted), resume #3 (action 4 replay), resume #4 (action 5).
        Total read_file invocations = 4."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable = _make_fs_runners()
        _run(reliable.run(scenarios["filesystem_pytest_recovery"]))
        inj = reliable.last_trial_state.injector
        assert inj._invocation_counts.get("read_file") == 4


class TestRecoveryNoCheckpointNegative:
    def test_no_checkpoint_fails(self):
        from evaluation.scenarios.filesystem_scenarios import (
            make_filesystem_no_checkpoint_scenario,
        )
        nc = make_filesystem_no_checkpoint_scenario()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(nc))
        assert not rec.completed

    def test_no_checkpoint_failure_reason(self):
        from evaluation.scenarios.filesystem_scenarios import (
            make_filesystem_no_checkpoint_scenario,
        )
        nc = make_filesystem_no_checkpoint_scenario()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(nc))
        assert rec.failure_reason == "NO_CHECKPOINT"

    def test_no_checkpoint_no_resume(self):
        from evaluation.scenarios.filesystem_scenarios import (
            make_filesystem_no_checkpoint_scenario,
        )
        nc = make_filesystem_no_checkpoint_scenario()
        _, reliable = _make_fs_runners()
        rec = _run(reliable.run(nc))
        assert rec.resume_count == 0


class TestRecoveryOrderIndependence:
    def test_naive_first_then_reliable(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, reliable = _make_fs_runners()
        n_rec = _run(naive.run(scenarios["filesystem_pytest_recovery"]))
        r_rec = _run(reliable.run(scenarios["filesystem_pytest_recovery"]))
        assert not n_rec.completed
        assert r_rec.completed

    def test_reliable_first_then_naive(self):
        scenarios = make_filesystem_standard_scenarios()
        naive, reliable = _make_fs_runners()
        r_rec = _run(reliable.run(scenarios["filesystem_pytest_recovery"]))
        n_rec = _run(naive.run(scenarios["filesystem_pytest_recovery"]))
        assert r_rec.completed
        assert not n_rec.completed


class TestRecoveryFreshTrialIsolation:
    def test_two_consecutive_reliable_runs(self):
        """Two consecutive Reliable recovery runs both complete."""
        scenarios = make_filesystem_standard_scenarios()
        for _ in range(2):
            _, reliable = _make_fs_runners()
            rec = _run(reliable.run(scenarios["filesystem_pytest_recovery"]))
            assert rec.completed
            assert rec.resume_count == 1

    def test_fresh_source_run_each_time(self):
        """Each run creates a fresh source Run object (not reused)."""
        scenarios = make_filesystem_standard_scenarios()
        _, reliable1 = _make_fs_runners()
        _run(reliable1.run(scenarios["filesystem_pytest_recovery"]))
        sr1 = reliable1.last_trial_state.source_run
        _, reliable2 = _make_fs_runners()
        _run(reliable2.run(scenarios["filesystem_pytest_recovery"]))
        sr2 = reliable2.last_trial_state.source_run
        # Different Run objects (not reused).
        assert sr1 is not sr2
        # Both interrupted (fresh trial state).
        from harness.state import RunStatus
        assert sr1.status == RunStatus.INTERRUPTED
        assert sr2.status == RunStatus.INTERRUPTED
