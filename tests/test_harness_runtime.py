"""Tests for the async harness run loop and resume (Phase 1 Step 2).

All tests are offline, deterministic, no LLM, no network, no MCP, no
external service. Async runtime methods are driven via ``asyncio.run``
so no ``pytest-asyncio`` plugin is required.

When a step fails or is cancelled, ``HarnessRuntime`` re-raises the
original exception with the affected ``Run`` attached as ``exc.run`` so
callers can inspect the final run state without holding a separate
reference.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from harness import (
    ExecutionState,
    HarnessRuntime,
    InvalidRuntimeStateError,
    ResumeError,
    RunStatus,
    Task,
    TaskStatus,
)
from harness.state import Checkpoint, Run
from storage import InMemoryCheckpointStore
from tests.fakes import FakeClock, FakeStepExecutor, make_id_factory

T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _make_task(task_id: str = "T-001") -> Task:
    return Task(task_id=task_id, goal="Diagnose the failing experiment", created_at=T0)


def _make_runtime(
    *,
    store=None,
    executor=None,
    run_prefix: str = "R",
    cp_prefix: str = "C",
):
    store = store or InMemoryCheckpointStore()
    executor = executor or FakeStepExecutor(complete_at_step=2)
    clock = FakeClock(T0)
    runtime = HarnessRuntime(
        checkpoint_store=store,
        step_executor=executor,
        clock=clock,
        run_id_factory=make_id_factory(run_prefix),
        checkpoint_id_factory=make_id_factory(cp_prefix),
    )
    return runtime, store, executor


# ---------------------------------------------------------------------------
# Section 36 — Normal completion
# ---------------------------------------------------------------------------


def test_normal_completion():
    runtime, store, executor = _make_runtime()
    task = _make_task()

    run = asyncio.run(runtime.start(task))

    assert executor.executed_steps == [0, 1, 2]
    assert run.current_step == 3
    assert run.last_checkpoint_id == "C-003"
    assert run.status == RunStatus.COMPLETED
    assert task.status == TaskStatus.COMPLETED
    assert run.resumed_from_checkpoint_id is None

    latest = asyncio.run(store.latest_for_run(run.run_id))
    assert latest is not None
    assert latest.step_index == 2


# ---------------------------------------------------------------------------
# Section 37 — every step checkpointed
# ---------------------------------------------------------------------------


def test_every_step_produces_a_checkpoint():
    runtime, store, _ = _make_runtime()
    task = _make_task()

    run = asyncio.run(runtime.start(task))

    c1 = asyncio.run(store.get("C-001"))
    c2 = asyncio.run(store.get("C-002"))
    c3 = asyncio.run(store.get("C-003"))
    assert c1 is not None and c1.step_index == 0
    assert c2 is not None and c2.step_index == 1
    assert c3 is not None and c3.step_index == 2
    # COMPLETE step is also checkpointed.
    assert run.last_checkpoint_id == "C-003"
    # All checkpoints belong to this run.
    for cp in (c1, c2, c3):
        assert cp.run_id == run.run_id


# ---------------------------------------------------------------------------
# Section 38 — interruption (CancelledError at step 2)
# ---------------------------------------------------------------------------


def test_interruption_on_cancellation():
    executor = FakeStepExecutor(complete_at_step=3, cancel_at_step=2)
    runtime, store, executor = _make_runtime(executor=executor)
    task = _make_task()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        asyncio.run(runtime.start(task))

    run = exc_info.value.run
    assert executor.executed_steps == [0, 1, 2]

    # step 0 and 1 checkpointed; step 2 not.
    assert asyncio.run(store.get("C-001")) is not None
    assert asyncio.run(store.get("C-002")) is not None
    assert asyncio.run(store.get("C-003")) is None

    assert run.status == RunStatus.INTERRUPTED
    assert task.status == TaskStatus.WAITING
    assert run.current_step == 2  # step 2 did not succeed, still next
    assert run.last_checkpoint_id == "C-002"


# ---------------------------------------------------------------------------
# Section 39 — cancellation is re-raised
# ---------------------------------------------------------------------------


def test_cancellation_is_reraised():
    executor = FakeStepExecutor(complete_at_step=3, cancel_at_step=1)
    runtime, _, _ = _make_runtime(executor=executor)
    task = _make_task()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runtime.start(task))


# ---------------------------------------------------------------------------
# Section 40 — basic resume
# ---------------------------------------------------------------------------


def test_resume_creates_new_run_and_continues():
    # First run: steps 0,1 succeed; step 2 cancelled.
    executor1 = FakeStepExecutor(complete_at_step=3, cancel_at_step=2)
    store = InMemoryCheckpointStore()
    clock = FakeClock(T0)
    run_ids = make_id_factory("R")
    cp_ids = make_id_factory("C")
    runtime1 = HarnessRuntime(
        checkpoint_store=store,
        step_executor=executor1,
        clock=clock,
        run_id_factory=run_ids,
        checkpoint_id_factory=cp_ids,
    )
    task = _make_task()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        asyncio.run(runtime1.start(task))
    old_run = exc_info.value.run

    # After interruption: latest checkpoint is C-002 (step_index=1).
    latest = asyncio.run(store.latest_for_run(old_run.run_id))
    assert latest is not None
    assert latest.checkpoint_id == "C-002"
    assert latest.step_index == 1

    assert old_run.status == RunStatus.INTERRUPTED
    assert task.status == TaskStatus.WAITING

    # Second run: resume from C-002. New executor completes at step 3.
    executor2 = FakeStepExecutor(complete_at_step=3)
    runtime2 = HarnessRuntime(
        checkpoint_store=store,
        step_executor=executor2,
        clock=clock,
        run_id_factory=run_ids,
        checkpoint_id_factory=cp_ids,
    )

    new_run = asyncio.run(
        runtime2.resume(task=task, source_run=old_run, checkpoint_id="C-002")
    )

    # New run is a different run.
    assert new_run.run_id != old_run.run_id
    assert new_run.run_id == "R-002"
    # Resumed from C-002.
    assert new_run.resumed_from_checkpoint_id == "C-002"
    # Began at step 2 (checkpoint.step_index + 1).
    assert executor2.executed_steps == [2, 3]
    # Completed.
    assert new_run.status == RunStatus.COMPLETED
    assert task.status == TaskStatus.COMPLETED
    assert new_run.current_step == 4


# ---------------------------------------------------------------------------
# Section 41 — completed_actions restored on resume
# ---------------------------------------------------------------------------


def test_resume_restores_completed_actions():
    executor1 = FakeStepExecutor(complete_at_step=3, cancel_at_step=2)
    store = InMemoryCheckpointStore()
    clock = FakeClock(T0)
    run_ids = make_id_factory("R")
    cp_ids = make_id_factory("C")
    runtime1 = HarnessRuntime(
        checkpoint_store=store,
        step_executor=executor1,
        clock=clock,
        run_id_factory=run_ids,
        checkpoint_id_factory=cp_ids,
    )
    task = _make_task()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        asyncio.run(runtime1.start(task))
    old_run = exc_info.value.run

    # The latest checkpoint should carry step-0 and step-1.
    latest = asyncio.run(store.latest_for_run(old_run.run_id))
    assert latest is not None
    assert latest.completed_actions == ("step-0", "step-1")

    # Capture the state seen by the second executor on its first invocation.
    seen_actions: list = []

    class ObservingExecutor:
        def __init__(self):
            self.executed_steps: list[int] = []

        async def execute_step(self, task, run, state):
            n = run.current_step
            self.executed_steps.append(n)
            if not seen_actions:
                seen_actions.append(state.completed_actions)
            new_actions = state.completed_actions + (f"step-{n}",)
            new_state = ExecutionState(
                agent_state=dict(state.agent_state),
                critical_context=dict(state.critical_context),
                tool_state=dict(state.tool_state),
                completed_actions=new_actions,
                pending_action=state.pending_action,
            )
            from harness.execution import StepOutcome, StepResult
            outcome = StepOutcome.COMPLETE if n == 3 else StepOutcome.CONTINUE
            return StepResult(outcome=outcome, state=new_state)

    executor2 = ObservingExecutor()
    runtime2 = HarnessRuntime(
        checkpoint_store=store,
        step_executor=executor2,
        clock=clock,
        run_id_factory=run_ids,
        checkpoint_id_factory=cp_ids,
    )

    new_run = asyncio.run(
        runtime2.resume(task=task, source_run=old_run, checkpoint_id="C-002")
    )

    # First invocation of the resume executor saw the restored actions.
    assert seen_actions == [("step-0", "step-1")]
    # Final completed_actions include the resumed steps.
    final_cp = asyncio.run(store.get(new_run.last_checkpoint_id))
    assert final_cp is not None
    assert final_cp.completed_actions == ("step-0", "step-1", "step-2", "step-3")


# ---------------------------------------------------------------------------
# Section 42 — historical run is not modified
# ---------------------------------------------------------------------------


def test_historical_run_not_modified_after_resume():
    executor1 = FakeStepExecutor(complete_at_step=3, cancel_at_step=2)
    store = InMemoryCheckpointStore()
    clock = FakeClock(T0)
    run_ids = make_id_factory("R")
    cp_ids = make_id_factory("C")
    runtime1 = HarnessRuntime(
        checkpoint_store=store,
        step_executor=executor1,
        clock=clock,
        run_id_factory=run_ids,
        checkpoint_id_factory=cp_ids,
    )
    task = _make_task()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        asyncio.run(runtime1.start(task))
    old_run = exc_info.value.run

    executor2 = FakeStepExecutor(complete_at_step=3)
    runtime2 = HarnessRuntime(
        checkpoint_store=store,
        step_executor=executor2,
        clock=clock,
        run_id_factory=run_ids,
        checkpoint_id_factory=cp_ids,
    )

    new_run = asyncio.run(
        runtime2.resume(task=task, source_run=old_run, checkpoint_id="C-002")
    )

    assert old_run.status == RunStatus.INTERRUPTED
    assert new_run.status == RunStatus.COMPLETED


# ---------------------------------------------------------------------------
# Section 43 — lineage fields
# ---------------------------------------------------------------------------


def test_resume_lineage_fields():
    executor1 = FakeStepExecutor(complete_at_step=3, cancel_at_step=2)
    store = InMemoryCheckpointStore()
    clock = FakeClock(T0)
    run_ids = make_id_factory("R")
    cp_ids = make_id_factory("C")
    runtime1 = HarnessRuntime(
        checkpoint_store=store,
        step_executor=executor1,
        clock=clock,
        run_id_factory=run_ids,
        checkpoint_id_factory=cp_ids,
    )
    task = _make_task()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        asyncio.run(runtime1.start(task))
    old_run = exc_info.value.run

    executor2 = FakeStepExecutor(complete_at_step=3)
    runtime2 = HarnessRuntime(
        checkpoint_store=store,
        step_executor=executor2,
        clock=clock,
        run_id_factory=run_ids,
        checkpoint_id_factory=cp_ids,
    )

    new_run = asyncio.run(
        runtime2.resume(task=task, source_run=old_run, checkpoint_id="C-002")
    )

    # The new run executed step 2 (C-003) and step 3 COMPLETE (C-004).
    # last_checkpoint_id is the new run's own most recent checkpoint.
    assert new_run.resumed_from_checkpoint_id == "C-002"
    assert new_run.last_checkpoint_id == "C-004"
    # The two fields are distinct and not confused.
    assert new_run.resumed_from_checkpoint_id != new_run.last_checkpoint_id


def test_resume_new_run_starts_with_no_last_checkpoint():
    """Right after creation (before the loop runs) last_checkpoint_id is None.

    Verified by resuming into an executor that immediately cancels, so
    the new run produces no checkpoint of its own.
    """
    executor1 = FakeStepExecutor(complete_at_step=3, cancel_at_step=2)
    store = InMemoryCheckpointStore()
    clock = FakeClock(T0)
    run_ids = make_id_factory("R")
    cp_ids = make_id_factory("C")
    runtime1 = HarnessRuntime(
        checkpoint_store=store,
        step_executor=executor1,
        clock=clock,
        run_id_factory=run_ids,
        checkpoint_id_factory=cp_ids,
    )
    task = _make_task()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        asyncio.run(runtime1.start(task))
    old_run = exc_info.value.run

    # New executor cancels immediately on its first (and only) step.
    executor2 = FakeStepExecutor(complete_at_step=3, cancel_at_step=2)
    runtime2 = HarnessRuntime(
        checkpoint_store=store,
        step_executor=executor2,
        clock=clock,
        run_id_factory=run_ids,
        checkpoint_id_factory=cp_ids,
    )

    with pytest.raises(asyncio.CancelledError) as exc_info:
        asyncio.run(
            runtime2.resume(task=task, source_run=old_run, checkpoint_id="C-002")
        )
    new_run = exc_info.value.run

    # The new run was resumed from C-002 but produced no checkpoint of its
    # own before being interrupted.
    assert new_run.resumed_from_checkpoint_id == "C-002"
    assert new_run.last_checkpoint_id is None
    assert new_run.status == RunStatus.INTERRUPTED


# ---------------------------------------------------------------------------
# Section 44 — invalid resume
# ---------------------------------------------------------------------------


def test_resume_missing_checkpoint():
    runtime, store, _ = _make_runtime()
    task = _make_task()
    old_run = Run(run_id="R-001", task_id=task.task_id, status=RunStatus.INTERRUPTED)
    task.transition_to(TaskStatus.RUNNING)
    task.transition_to(TaskStatus.WAITING)

    with pytest.raises(ResumeError, match="not found"):
        asyncio.run(
            runtime.resume(task=task, source_run=old_run, checkpoint_id="C-999")
        )


def test_resume_wrong_checkpoint_run():
    runtime, store, _ = _make_runtime()
    task = _make_task()
    cp = Checkpoint(
        checkpoint_id="C-X", run_id="R-OTHER", step_index=0, created_at=T0
    )
    asyncio.run(store.save(cp))
    old_run = Run(run_id="R-001", task_id=task.task_id, status=RunStatus.INTERRUPTED)
    task.transition_to(TaskStatus.RUNNING)
    task.transition_to(TaskStatus.WAITING)

    with pytest.raises(ResumeError, match="belongs to run"):
        asyncio.run(
            runtime.resume(task=task, source_run=old_run, checkpoint_id="C-X")
        )


def test_resume_wrong_run_task():
    runtime, store, _ = _make_runtime()
    task = _make_task(task_id="T-001")
    cp = Checkpoint(
        checkpoint_id="C-001", run_id="R-001", step_index=0, created_at=T0
    )
    asyncio.run(store.save(cp))
    old_run = Run(
        run_id="R-001", task_id="T-OTHER", status=RunStatus.INTERRUPTED
    )
    task.transition_to(TaskStatus.RUNNING)
    task.transition_to(TaskStatus.WAITING)

    with pytest.raises(ResumeError, match="belongs to task"):
        asyncio.run(
            runtime.resume(task=task, source_run=old_run, checkpoint_id="C-001")
        )


def test_resume_source_run_not_interrupted():
    runtime, store, _ = _make_runtime()
    task = _make_task()
    cp = Checkpoint(
        checkpoint_id="C-001", run_id="R-001", step_index=0, created_at=T0
    )
    asyncio.run(store.save(cp))
    old_run = Run(run_id="R-001", task_id=task.task_id, status=RunStatus.COMPLETED)
    task.transition_to(TaskStatus.RUNNING)
    task.transition_to(TaskStatus.WAITING)

    with pytest.raises(ResumeError, match="must be INTERRUPTED"):
        asyncio.run(
            runtime.resume(task=task, source_run=old_run, checkpoint_id="C-001")
        )


# ---------------------------------------------------------------------------
# Section 45 — ordinary failure
# ---------------------------------------------------------------------------


def test_ordinary_failure():
    executor = FakeStepExecutor(complete_at_step=3, fail_at_step=1)
    runtime, store, _ = _make_runtime(executor=executor)
    task = _make_task()

    with pytest.raises(RuntimeError, match="boom") as exc_info:
        asyncio.run(runtime.start(task))
    run = exc_info.value.run

    # step 0 checkpoint exists; step 1 does not.
    assert asyncio.run(store.get("C-001")) is not None
    assert asyncio.run(store.get("C-002")) is None

    assert run.status == RunStatus.FAILED
    assert task.status == TaskStatus.FAILED


# ---------------------------------------------------------------------------
# Section 46 — failed/cancelled steps consume no checkpoint id
# ---------------------------------------------------------------------------


def test_failed_step_consumes_no_checkpoint_id():
    """A failed step must not consume a checkpoint id.

    After step 0 succeeds (C-001) and step 1 fails, C-002 was never
    generated — the failed step burned no id.
    """
    executor = FakeStepExecutor(complete_at_step=3, fail_at_step=1)
    runtime, store, _ = _make_runtime(executor=executor)
    task = _make_task()

    with pytest.raises(RuntimeError):
        asyncio.run(runtime.start(task))

    assert asyncio.run(store.get("C-001")) is not None
    assert asyncio.run(store.get("C-002")) is None


def test_cancelled_step_consumes_no_checkpoint_id():
    executor = FakeStepExecutor(complete_at_step=3, cancel_at_step=2)
    runtime, store, _ = _make_runtime(executor=executor)
    task = _make_task()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runtime.start(task))

    assert asyncio.run(store.get("C-001")) is not None
    assert asyncio.run(store.get("C-002")) is not None
    assert asyncio.run(store.get("C-003")) is None  # cancelled step 2 burned none


# ---------------------------------------------------------------------------
# Fresh start requires PENDING task
# ---------------------------------------------------------------------------


def test_start_requires_pending_task():
    runtime, _, _ = _make_runtime()
    task = _make_task()
    task.transition_to(TaskStatus.RUNNING)

    with pytest.raises(InvalidRuntimeStateError, match="PENDING"):
        asyncio.run(runtime.start(task))
