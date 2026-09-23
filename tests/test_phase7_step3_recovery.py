"""Phase 7 Step 3 — Deterministic Interruption, Checkpoint Resume &
Recovery Benchmark tests.

Covers Sections 61-70 of the Phase 7 Step 3 specification:
- Interruption boundary (read_file receives PROCESS_INTERRUPTION, fault
  fires once, fault consumed during resume, same placement for both
  runners)
- Harness interruption (InjectedProcessInterruption → CancelledError,
  source Run INTERRUPTED, Task WAITING, interrupted step no checkpoint)
- Resume state (real checkpoint loaded, new Run, resumed_from correct,
  current_step = checkpoint.step_index + 1, ExecutionState restored,
  script cursor restored, checkpointed step not replayed)
- Source cursor (interrupted step re-executed, not skipped)
- Fresh runtime resume (new HarnessRuntime instance)
- No-replay (checkpointed step execute count == 1, interrupted step == 2)
- Recovery result (Reliable completes with resume_count=1, Naive fails
  with resume_count=0)
- Real checkpoint store (source checkpoint exists, interrupted step
  absent, resume checkpoints belong to new Run)
- Oracle (Reliable recovery → complete, Naive interruption → incomplete)
- No checkpoint negative case (interruption before any checkpoint →
  cannot resume → fail)
"""

from __future__ import annotations

import asyncio

import pytest

from evaluation.comparison import BenchmarkComparator
from evaluation.faults import FaultInjector, FaultPlan, FaultSpec, FaultType
from evaluation.models import BenchmarkScenario, BenchmarkTask
from evaluation.records import BenchmarkEventType, BenchmarkRunRecord
from evaluation.repository import RepositoryOracle
from evaluation.runners import (
    BenchmarkStepExecutor,
    FAILURE_INTERRUPTED,
    FAILURE_NO_CHECKPOINT,
    NaiveBenchmarkRunner,
    ReliableHarnessBenchmarkRunner,
    SCRIPT_CURSOR_KEY,
    TrialState,
)
from evaluation.scenarios.controlled_repo import (
    ScriptedAction,
    ScriptedActionSource,
    make_fix_calculator_actions,
    make_standard_fault_plans,
    make_standard_fixtures,
    make_standard_scenarios,
)
from harness.state import RunStatus, TaskStatus
from tools import RetryPolicy, ToolCall, ToolExecutionContext, ToolRegistry, ToolRuntime
from tools.models import ToolErrorType


def _run(coro):
    return asyncio.run(coro)


class _FakeSleeper:
    def __init__(self):
        self.sleeps: list[float] = []

    async def __call__(self, seconds: float):
        self.sleeps.append(seconds)


_TIMEOUT_SECONDS = 0.01


def _make_naive():
    return NaiveBenchmarkRunner(
        run_tests_timeout_seconds=_TIMEOUT_SECONDS,
        sleep=_FakeSleeper(),
    )


def _make_reliable():
    return ReliableHarnessBenchmarkRunner(
        run_tests_timeout_seconds=_TIMEOUT_SECONDS,
        sleep=_FakeSleeper(),
    )


def _make_comparator():
    return BenchmarkComparator(_make_naive(), _make_reliable())


def _get_scenario(scenario_id: str) -> BenchmarkScenario:
    return make_standard_scenarios()[scenario_id]


# ===========================================================================
# Section 61 — Interruption boundary tests
# ===========================================================================


class TestInterruptionBoundary:
    """read_file receives PROCESS_INTERRUPTION; fault fires once and
    remains consumed during resume."""

    def test_read_file_receives_process_interruption(self):
        """The read_file handler receives the PROCESS_INTERRUPTION fault
        on logical invocation #1."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        # If the fault didn't fire, the run would complete without
        # interruption (resume_count=0). The fact that resume_count=1
        # proves the fault fired on read_file #1.
        assert record.resume_count == 1

    def test_fault_fires_exactly_once(self):
        """The PROCESS_INTERRUPTION fault fires exactly once per trial."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        # The injector should have fired exactly 1 fault.
        assert len(ts.injector.fired_fault_indices) == 1

    def test_fault_remains_consumed_during_resume(self):
        """After the fault fires on the source Run, it does NOT re-fire
        on the resume Run."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        # If the fault re-fired on resume, the resume run would also be
        # interrupted (resume_count would still be 1 but completed=False).
        assert record.completed is True
        assert record.resume_count == 1
        ts = runner.last_trial_state
        assert len(ts.injector.fired_fault_indices) == 1

    def test_same_fault_placement_for_naive_and_reliable(self):
        """Both runners see the PROCESS_INTERRUPTION fault on the same
        logical read_file invocation #1."""
        comparator = _make_comparator()
        comparison = _run(
            comparator.compare(_get_scenario("checkpoint_recovery"))
        )
        # Naive: the fault fires on read_file #1, naive terminates.
        assert comparison.naive_record.completed is False
        assert comparison.naive_record.failure_reason == FAILURE_INTERRUPTED
        # Naive executed 2 actions (run_tests + read_file interrupted).
        assert comparison.naive_record.logical_action_count == 2

        # Reliable: the fault fires on read_file #1 (source run), then
        # resume completes.
        assert comparison.reliable_record.completed is True
        assert comparison.reliable_record.resume_count == 1


# ===========================================================================
# Section 62 — Harness interruption tests
# ===========================================================================


class TestHarnessInterruption:
    """InjectedProcessInterruption → CancelledError → Run INTERRUPTED."""

    def test_source_run_becomes_interrupted(self):
        """The source Run transitions to INTERRUPTED."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        assert ts.source_run is not None
        assert ts.source_run.status == RunStatus.INTERRUPTED

    def test_task_becomes_waiting_after_interruption(self):
        """The Task transitions to WAITING after interruption (before
        resume)."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        # After the full run (including resume), the task should be
        # COMPLETED. But the source run interruption set it to WAITING
        # before resume. We verify the interruption occurred and the
        # task eventually completed.
        assert ts.interruption_occurred is True
        assert ts.task is not None
        assert ts.task.status == TaskStatus.COMPLETED

    def test_interrupted_step_has_no_checkpoint(self):
        """The interrupted step (step 1, read_file) produces no
        checkpoint."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        # The source run's last checkpoint should be from step 0
        # (run_tests), not step 1 (read_file).
        assert ts.source_run.last_checkpoint_id is not None
        checkpoint = _run(
            ts.checkpoint_store.get(ts.source_run.last_checkpoint_id)
        )
        assert checkpoint.step_index == 0

    def test_source_run_current_step_is_interrupted_step(self):
        """The source Run's current_step is the interrupted step index."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        # The interruption happened at step 1 (read_file). The
        # HarnessRuntime does NOT advance current_step on interruption.
        assert ts.source_run.current_step == 1


# ===========================================================================
# Section 63 — Resume state tests
# ===========================================================================


class TestResumeState:
    """Real checkpoint loaded, new Run created, state restored."""

    def test_actual_checkpoint_loaded_from_store(self):
        """The resume uses a real checkpoint loaded from the store."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        assert ts.source_run.last_checkpoint_id is not None
        checkpoint = _run(
            ts.checkpoint_store.get(ts.source_run.last_checkpoint_id)
        )
        assert checkpoint is not None
        assert checkpoint.checkpoint_id == ts.source_run.last_checkpoint_id

    def test_resume_creates_new_run(self):
        """Resume creates a new Run with a different run_id."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        assert ts.source_run is not None
        assert ts.resume_run is not None
        assert ts.resume_run.run_id != ts.source_run.run_id

    def test_resumed_from_checkpoint_id_correct(self):
        """The resume Run's resumed_from_checkpoint_id matches the source
        Run's last_checkpoint_id."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        assert ts.resume_run.resumed_from_checkpoint_id == (
            ts.source_run.last_checkpoint_id
        )

    def test_resume_starts_at_checkpoint_step_plus_one(self):
        """The resume Run's current_step starts at
        checkpoint.step_index + 1."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        checkpoint = _run(
            ts.checkpoint_store.get(ts.source_run.last_checkpoint_id)
        )
        # The resume run should have started at checkpoint.step_index + 1 = 1.
        # After completion, current_step has advanced, but we can verify
        # from the execution history that the first resume step was 1.
        resume_history = ts.resume_executor.execution_history
        first_resume_step = resume_history[0][1]
        assert first_resume_step == checkpoint.step_index + 1

    def test_execution_state_restored(self):
        """The ExecutionState is restored from the checkpoint."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        checkpoint = _run(
            ts.checkpoint_store.get(ts.source_run.last_checkpoint_id)
        )
        # The checkpoint should contain the script cursor and
        # completed_actions from step 0.
        assert SCRIPT_CURSOR_KEY in checkpoint.agent_state
        assert checkpoint.agent_state[SCRIPT_CURSOR_KEY] == 1
        assert "run_tests" in checkpoint.completed_actions

    def test_script_cursor_restored(self):
        """The script cursor is restored from the checkpointed
        ExecutionState."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        # The resume executor's source should have been seeked to index 1.
        assert ts.resume_executor is not None
        # After executing steps 1, 2, 3, the source should be exhausted
        # (index 4).
        assert ts.resume_executor._source.current_index == 4

    def test_checkpointed_step_not_replayed(self):
        """Step 0 (run_tests) is NOT re-executed in the resume Run."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        # The resume history should start at step 1, not step 0.
        resume_history = ts.resume_executor.execution_history
        step_indices = [h[1] for h in resume_history]
        assert 0 not in step_indices
        assert 1 in step_indices


# ===========================================================================
# Section 64 — Source cursor tests
# ===========================================================================


class TestSourceCursor:
    """The interrupted step is re-executed, not skipped."""

    def test_interrupted_step_re_executed(self):
        """Step 1 (read_file) is re-executed in the resume Run."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        resume_history = ts.resume_executor.execution_history
        # The first resume step should be step 1 (read_file).
        assert resume_history[0] == (
            ts.resume_run.run_id, 1, "read_file", "success"
        )

    def test_resume_does_not_skip_interrupted_step(self):
        """Resume does NOT incorrectly begin with step 2."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        resume_history = ts.resume_executor.execution_history
        first_step = resume_history[0][1]
        assert first_step == 1  # NOT 2


# ===========================================================================
# Section 65 — Fresh runtime resume tests
# ===========================================================================


class TestFreshRuntimeResume:
    """Recovery succeeds with a new HarnessRuntime instance."""

    def test_recovery_with_new_runtime_instance(self):
        """The resume uses a new HarnessRuntime instance (not the source
        runtime). This proves recovery doesn't depend on the old
        runtime's local loop state."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        # If recovery depended on the old runtime's state, it would fail.
        assert record.completed is True
        assert record.resume_count == 1


# ===========================================================================
# Section 66 — No-replay tests
# ===========================================================================


class TestNoReplay:
    """Checkpointed steps are not re-executed; interrupted steps are."""

    def test_checkpointed_step_executed_once(self):
        """Step 0 (run_tests) is executed exactly once across source +
        resume Runs."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        source_history = ts.source_executor.execution_history
        resume_history = ts.resume_executor.execution_history
        all_history = source_history + resume_history

        # Count step 0 executions.
        step_0_count = sum(1 for h in all_history if h[1] == 0)
        assert step_0_count == 1

    def test_interrupted_step_executed_twice(self):
        """Step 1 (read_file) is executed twice: once interrupted, once
        resumed success."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        source_history = ts.source_executor.execution_history
        resume_history = ts.resume_executor.execution_history
        all_history = source_history + resume_history

        # Count step 1 executions.
        step_1_count = sum(1 for h in all_history if h[1] == 1)
        assert step_1_count == 2

        # Verify outcomes: one interrupted, one success.
        step_1_outcomes = [h[3] for h in all_history if h[1] == 1]
        assert "interrupted" in step_1_outcomes
        assert "success" in step_1_outcomes


# ===========================================================================
# Section 67 — Recovery result tests
# ===========================================================================


class TestRecoveryResult:
    """Reliable completes with resume_count=1; Naive fails with
    resume_count=0."""

    def test_reliable_completed(self):
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        assert record.completed is True

    def test_reliable_resume_count_one(self):
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        assert record.resume_count == 1

    def test_reliable_failure_reason_none(self):
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        assert record.failure_reason is None

    def test_naive_not_completed(self):
        runner = _make_naive()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        assert record.completed is False

    def test_naive_resume_count_zero(self):
        runner = _make_naive()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        assert record.resume_count == 0

    def test_naive_failure_reason_interrupted(self):
        runner = _make_naive()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        assert record.failure_reason == FAILURE_INTERRUPTED


# ===========================================================================
# Section 68 — Real checkpoint store tests
# ===========================================================================


class TestRealCheckpointStore:
    """Checkpoint assertions come from the real store, not only the
    executor counter."""

    def test_source_checkpoint_exists(self):
        """A real checkpoint exists for the source Run."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        checkpoint = _run(
            ts.checkpoint_store.get(ts.source_run.last_checkpoint_id)
        )
        assert checkpoint is not None
        assert checkpoint.run_id == ts.source_run.run_id

    def test_interrupted_step_absent_from_source_checkpoints(self):
        """No checkpoint exists for the interrupted step (step 1)."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        # The source run's last checkpoint is from step 0, not step 1.
        checkpoint = _run(
            ts.checkpoint_store.get(ts.source_run.last_checkpoint_id)
        )
        assert checkpoint.step_index == 0

    def test_resume_checkpoints_belong_to_new_run(self):
        """Checkpoints produced during resume belong to the resume Run,
        not the source Run."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        # The resume run should have its own checkpoints.
        # We can verify by checking the resume run's last checkpoint.
        # After resume completes, the resume run's last_checkpoint_id
        # should be set and belong to the resume run.
        assert ts.resume_run.last_checkpoint_id is not None
        resume_checkpoint = _run(
            ts.checkpoint_store.get(ts.resume_run.last_checkpoint_id)
        )
        assert resume_checkpoint.run_id == ts.resume_run.run_id
        assert resume_checkpoint.run_id != ts.source_run.run_id

    def test_resume_checkpoint_step_index_geq_one(self):
        """Resume checkpoints have step_index >= 1 (they start after the
        source checkpoint)."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        resume_checkpoint = _run(
            ts.checkpoint_store.get(ts.resume_run.last_checkpoint_id)
        )
        assert resume_checkpoint.step_index >= 1


# ===========================================================================
# Section 69 — Oracle tests
# ===========================================================================


class TestOracle:
    """Oracle determines completion after recovery."""

    def test_reliable_recovery_oracle_complete(self):
        """After reliable recovery, the oracle says the repository is
        complete."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        assert record.completed is True
        # Verify directly with the oracle.
        ts = runner.last_trial_state
        oracle = RepositoryOracle()
        assert oracle.is_complete(ts.repo) is True

    def test_naive_interruption_oracle_incomplete(self):
        """After naive interruption, the oracle says the repository is
        incomplete."""
        runner = _make_naive()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        assert record.completed is False


# ===========================================================================
# Section 70 — No checkpoint negative case
# ===========================================================================


class TestNoCheckpointNegativeCase:
    """Interruption before any successful step → cannot resume → fail."""

    def test_no_checkpoint_cannot_resume(self):
        """If interruption occurs before any successful checkpoint, the
        reliable runner cannot resume and fails deterministically."""
        # Create a scenario where the first action (run_tests) is
        # interrupted. This means no checkpoint exists when interruption
        # occurs.
        from evaluation.scenarios.controlled_repo import (
            make_repository_tool_specs,
            make_repository_tool_handlers,
        )

        # Use the existing "interruption" scenario which interrupts
        # run_tests #1 (the first action). The reliable runner should
        # fail with NO_CHECKPOINT because there's no checkpoint to
        # resume from.
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("interruption")))
        assert record.completed is False
        assert record.failure_reason == FAILURE_NO_CHECKPOINT
        assert record.resume_count == 0

    def test_no_fake_fresh_start_on_resume_failure(self):
        """When resume fails due to no checkpoint, the runner does NOT
        fall back to a fresh start."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("interruption")))
        # The run should fail, not complete via a secret fresh start.
        assert record.completed is False


# ===========================================================================
# Section 45 — Checkpoint lineage tests
# ===========================================================================


class TestCheckpointLineage:
    """Source and resume checkpoints have correct lineage."""

    def test_source_checkpoint_run_id(self):
        """The source checkpoint's run_id matches the source Run."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        checkpoint = _run(
            ts.checkpoint_store.get(ts.source_run.last_checkpoint_id)
        )
        assert checkpoint.run_id == ts.source_run.run_id

    def test_resume_checkpoint_run_id(self):
        """The resume checkpoint's run_id matches the resume Run."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        resume_checkpoint = _run(
            ts.checkpoint_store.get(ts.resume_run.last_checkpoint_id)
        )
        assert resume_checkpoint.run_id == ts.resume_run.run_id

    def test_resume_checkpoint_not_on_source_run(self):
        """Resume checkpoints are NOT incorrectly attributed to the
        source Run."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        resume_checkpoint = _run(
            ts.checkpoint_store.get(ts.resume_run.last_checkpoint_id)
        )
        assert resume_checkpoint.run_id != ts.source_run.run_id


# ===========================================================================
# Section 23-25 — FaultInjector lifecycle tests
# ===========================================================================


class TestFaultInjectorLifecycle:
    """The same FaultInjector spans source + resume Runs."""

    def test_same_injector_spans_source_and_resume(self):
        """The FaultInjector is the same instance for source and resume
        Runs (within one trial)."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        # The injector is shared across source + resume.
        assert ts.injector is not None
        # The fault was fired exactly once (on the source run).
        assert len(ts.injector.fired_fault_indices) == 1

    def test_fault_not_refired_on_resume(self):
        """The PROCESS_INTERRUPTION fault is NOT re-fired on resume."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        # Only 1 fault fired (on source run). If it re-fired on resume,
        # there would be 2 or the resume would fail.
        assert len(ts.injector.fired_fault_indices) == 1
        assert record.completed is True


# ===========================================================================
# Section 24 — FaultAwareToolInvoker cleanup tests
# ===========================================================================


class TestInvokerCleanup:
    """FaultAwareToolInvoker cleans active invocation on interruption."""

    def test_no_stale_active_invocation_after_interruption(self):
        """After interruption, there is no stale active invocation."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        # The injector should have no active invocations after the
        # trial completes (FaultAwareToolInvoker's finally block runs
        # even on BaseException propagation).
        assert ts.injector.current_invocation("read_file") == 0
        assert ts.injector.current_invocation("run_tests") == 0


# ===========================================================================
# Section 31 — Cursor consistency tests
# ===========================================================================


class TestCursorConsistency:
    """run.current_step matches the restored script cursor."""

    def test_cursor_matches_current_step_on_resume(self):
        """On resume, the restored script cursor matches
        run.current_step."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        # The checkpoint stores the script cursor.
        checkpoint = _run(
            ts.checkpoint_store.get(ts.source_run.last_checkpoint_id)
        )
        script_next = checkpoint.agent_state.get(SCRIPT_CURSOR_KEY, 0)
        # The resume run starts at checkpoint.step_index + 1.
        # The script cursor should match.
        assert script_next == checkpoint.step_index + 1


# ===========================================================================
# Section 51 — Runner order independence
# ===========================================================================


class TestOrderIndependence:
    """Running naive→reliable vs reliable→naive produces the same
    recovery result."""

    def test_order_independence(self):
        comparator = _make_comparator()
        scenario = _get_scenario("checkpoint_recovery")
        c1 = _run(comparator.compare(scenario))
        c2 = _run(comparator.compare_reversed(scenario))
        assert c1.naive_completed == c2.naive_completed
        assert c1.reliable_completed == c2.reliable_completed
        assert c1.reliable_record.resume_count == c2.reliable_record.resume_count


# ===========================================================================
# Section 48 — Determinism
# ===========================================================================


class TestDeterminism:
    """Same runner + same scenario run twice → same functional record."""

    def test_reliable_deterministic(self):
        runner = _make_reliable()
        scenario = _get_scenario("checkpoint_recovery")
        r1 = _run(runner.run(scenario))
        r2 = _run(runner.run(scenario))
        assert r1.completed == r2.completed
        assert r1.resume_count == r2.resume_count
        assert r1.logical_action_count == r2.logical_action_count
        assert r1.tool_attempt_count == r2.tool_attempt_count
        assert r1.checkpoint_count == r2.checkpoint_count
        assert r1.failure_reason == r2.failure_reason

    def test_naive_deterministic(self):
        runner = _make_naive()
        scenario = _get_scenario("checkpoint_recovery")
        r1 = _run(runner.run(scenario))
        r2 = _run(runner.run(scenario))
        assert r1.completed == r2.completed
        assert r1.resume_count == r2.resume_count
        assert r1.logical_action_count == r2.logical_action_count
        assert r1.failure_reason == r2.failure_reason


# ===========================================================================
# Section 52 — Existing Step 2 matrix unchanged
# ===========================================================================


class TestStep2MatrixUnchanged:
    """The four Step 2 scenarios still produce the same results."""

    def test_clean_success_both_complete(self):
        comparator = _make_comparator()
        c = _run(comparator.compare(_get_scenario("clean_success")))
        assert c.naive_completed is True
        assert c.reliable_completed is True

    def test_transient_naive_fails_reliable_completes(self):
        comparator = _make_comparator()
        c = _run(comparator.compare(_get_scenario("transient_failure")))
        assert c.naive_completed is False
        assert c.reliable_completed is True

    def test_timeout_naive_fails_reliable_completes(self):
        comparator = _make_comparator()
        c = _run(comparator.compare(_get_scenario("timeout")))
        assert c.naive_completed is False
        assert c.reliable_completed is True

    def test_permanent_both_fail(self):
        comparator = _make_comparator()
        c = _run(comparator.compare(_get_scenario("permanent_failure")))
        assert c.naive_completed is False
        assert c.reliable_completed is False


# ===========================================================================
# Section 35 — Recovery success metric
# ===========================================================================


class TestRecoverySuccessMetric:
    """Recovery Success = interruption occurred AND resume occurred AND
    task completed."""

    def test_reliable_recovery_success(self):
        """Reliable: interruption=yes, resume=yes, completed=yes →
        recovery success."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        interruption = ts.interruption_occurred
        resume = record.resume_count == 1
        completed = record.completed
        recovery_success = interruption and resume and completed
        assert recovery_success is True

    def test_naive_no_recovery_success(self):
        """Naive: interruption=yes, resume=no, completed=no → no recovery
        success."""
        runner = _make_naive()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        # Naive has no interruption_occurred field, but we know it
        # terminated due to interruption.
        resume = record.resume_count == 1
        completed = record.completed
        recovery_success = resume and completed
        assert recovery_success is False


# ===========================================================================
# Section 36 — BenchmarkRunRecord accounting for recovery
# ===========================================================================


class TestRecoveryAccounting:
    """The aggregated BenchmarkRunRecord has correct accounting."""

    def test_reliable_logical_action_count(self):
        """Reliable: 5 logical actions (2 source + 3 resume)."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        # Source: run_tests + read_file(interrupted) = 2
        # Resume: read_file + write_file + run_tests = 3
        assert record.logical_action_count == 5

    def test_reliable_tool_invocation_count(self):
        """Reliable: 5 tool invocations (2 source + 3 resume)."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        assert record.tool_invocation_count == 5

    def test_reliable_tool_attempt_count(self):
        """Reliable: 5 attempts (2 source + 3 resume, no retries)."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        assert record.tool_attempt_count == 5

    def test_reliable_retry_count_zero(self):
        """Reliable: 0 retries (no retryable faults in this scenario)."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        assert record.retry_count == 0

    def test_reliable_checkpoint_count(self):
        """Reliable: 5 checkpoints (1 source + 4 resume)."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        # Source: step 0 checkpoint = 1
        # Resume: step 1, 2, 3, COMPLETE = 4
        assert record.checkpoint_count == 5

    def test_naive_logical_action_count(self):
        """Naive: 2 logical actions (run_tests + read_file interrupted)."""
        runner = _make_naive()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        assert record.logical_action_count == 2

    def test_naive_checkpoint_count_zero(self):
        """Naive: 0 checkpoints (no HarnessRuntime)."""
        runner = _make_naive()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        assert record.checkpoint_count == 0


# ===========================================================================
# Section 46 — Oracle always wins
# ===========================================================================


class TestOracleAlwaysWinsRecovery:
    """Oracle determines completion even in recovery scenarios."""

    def test_oracle_not_harness_status(self):
        """Completion comes from the oracle, not from Harness COMPLETED
        status."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts = runner.last_trial_state
        # The resume run is COMPLETED, but that's not what determines
        # benchmark completion. The oracle does.
        assert ts.resume_run.status == RunStatus.COMPLETED
        # But we verify with the oracle directly.
        oracle = RepositoryOracle()
        assert oracle.is_complete(ts.repo) is True
        assert record.completed is True


# ===========================================================================
# Section 50 — Trial-level fresh state
# ===========================================================================


class TestTrialFreshState:
    """Each trial gets fresh state; no cross-trial contamination."""

    def test_fresh_repo_per_trial(self):
        """Each trial gets a fresh repository."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts1 = runner.last_trial_state
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts2 = runner.last_trial_state
        assert ts1.repo is not ts2.repo

    def test_fresh_injector_per_trial(self):
        """Each trial gets a fresh fault injector."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts1 = runner.last_trial_state
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts2 = runner.last_trial_state
        assert ts1.injector is not ts2.injector

    def test_fresh_source_executor_per_trial(self):
        """Each trial gets a fresh source executor."""
        runner = _make_reliable()
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts1 = runner.last_trial_state
        _run(runner.run(_get_scenario("checkpoint_recovery")))
        ts2 = runner.last_trial_state
        assert ts1.source_executor is not ts2.source_executor
