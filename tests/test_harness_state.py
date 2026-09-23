"""Tests for the harness runtime state models (Phase 1 Step 1)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from harness import (
    Checkpoint,
    InvalidStateTransitionError,
    Run,
    RunStatus,
    Task,
    TaskStatus,
)

T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
T2 = datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)
T3 = datetime(2026, 1, 1, 0, 0, 3, tzinfo=timezone.utc)


def _make_task() -> Task:
    return Task(
        task_id="T-001",
        goal="Diagnose the failing experiment",
        created_at=T0,
    )


def _make_run(task: Task | None = None) -> Run:
    task = task or _make_task()
    return Run(run_id="R-001", task_id=task.task_id)


def _make_checkpoint(run: Run | None = None, step: int = 0) -> Checkpoint:
    run = run or _make_run()
    return Checkpoint(
        checkpoint_id=f"C-{step:03d}",
        run_id=run.run_id,
        step_index=step,
        created_at=T0,
    )


# ---------------------------------------------------------------------------
# Test 1 — independent IDs and correct associations
# ---------------------------------------------------------------------------


def test_independent_ids_and_associations():
    task = _make_task()
    run = _make_run(task)
    checkpoint = _make_checkpoint(run)

    # IDs are semantically independent (distinct values).
    assert task.task_id != run.run_id != checkpoint.checkpoint_id
    assert {task.task_id, run.run_id, checkpoint.checkpoint_id} == {
        "T-001",
        "R-001",
        "C-000",
    }

    # Each model stores its own ID.
    assert task.task_id == "T-001"
    assert run.run_id == "R-001"
    assert checkpoint.checkpoint_id == "C-000"

    # Associations are explicit.
    assert run.task_id == task.task_id
    assert checkpoint.run_id == run.run_id


# ---------------------------------------------------------------------------
# Test 2 — valid Run transitions
# ---------------------------------------------------------------------------


def test_run_pending_to_running_to_completed():
    run = _make_run()
    run.transition_to(RunStatus.RUNNING, now=T1)
    assert run.status == RunStatus.RUNNING
    assert run.started_at == T1
    assert run.ended_at is None

    run.transition_to(RunStatus.COMPLETED, now=T2)
    assert run.status == RunStatus.COMPLETED
    assert run.ended_at == T2


# ---------------------------------------------------------------------------
# Test 3 — waiting transition
# ---------------------------------------------------------------------------


def test_run_waiting_round_trip():
    run = _make_run()
    run.transition_to(RunStatus.RUNNING, now=T1)
    run.transition_to(RunStatus.WAITING, now=T2)
    assert run.status == RunStatus.WAITING
    run.transition_to(RunStatus.RUNNING, now=T3)
    assert run.status == RunStatus.RUNNING
    run.transition_to(RunStatus.COMPLETED, now=T3)
    assert run.status == RunStatus.COMPLETED


# ---------------------------------------------------------------------------
# Test 4 — interrupted Run is terminal
# ---------------------------------------------------------------------------


def test_interrupted_run_cannot_restart():
    run = _make_run()
    run.transition_to(RunStatus.RUNNING, now=T1)
    run.transition_to(RunStatus.INTERRUPTED, now=T2)
    assert run.status == RunStatus.INTERRUPTED
    assert run.is_terminal
    assert run.ended_at == T2

    with pytest.raises(InvalidStateTransitionError) as exc:
        run.transition_to(RunStatus.RUNNING, now=T3)
    assert "interrupted -> running" in str(exc.value)
    assert exc.value.entity == "Run"
    assert exc.value.current == "interrupted"
    assert exc.value.target == "running"


# ---------------------------------------------------------------------------
# Test 5 — completed Run cannot restart
# ---------------------------------------------------------------------------


def test_completed_run_cannot_restart():
    run = _make_run()
    run.transition_to(RunStatus.RUNNING, now=T1)
    run.transition_to(RunStatus.COMPLETED, now=T2)
    with pytest.raises(InvalidStateTransitionError):
        run.transition_to(RunStatus.RUNNING, now=T3)


# ---------------------------------------------------------------------------
# Test 6 — failed Run cannot restart
# ---------------------------------------------------------------------------


def test_failed_run_cannot_restart():
    run = _make_run()
    run.transition_to(RunStatus.RUNNING, now=T1)
    run.transition_to(RunStatus.FAILED, now=T2)
    with pytest.raises(InvalidStateTransitionError):
        run.transition_to(RunStatus.RUNNING, now=T3)


# ---------------------------------------------------------------------------
# Test 7 — invalid direct transition
# ---------------------------------------------------------------------------


def test_pending_cannot_jump_to_completed():
    run = _make_run()
    with pytest.raises(InvalidStateTransitionError):
        run.transition_to(RunStatus.COMPLETED, now=T1)


def test_pending_cannot_jump_to_waiting():
    run = _make_run()
    with pytest.raises(InvalidStateTransitionError):
        run.transition_to(RunStatus.WAITING, now=T1)


# ---------------------------------------------------------------------------
# Test 8 — lifecycle timestamps
# ---------------------------------------------------------------------------


def test_run_lifecycle_timestamps_set_once():
    run = _make_run()
    assert run.started_at is None
    assert run.ended_at is None

    run.transition_to(RunStatus.RUNNING, now=T1)
    assert run.started_at == T1

    # Re-entering RUNNING from WAITING must not overwrite the original start.
    run.transition_to(RunStatus.WAITING, now=T2)
    run.transition_to(RunStatus.RUNNING, now=T3)
    assert run.started_at == T1

    run.transition_to(RunStatus.FAILED, now=T3)
    assert run.ended_at == T3


def test_run_started_at_not_overwritten_when_provided():
    run = Run(run_id="R-002", task_id="T-001", started_at=T0)
    run.transition_to(RunStatus.RUNNING, now=T1)
    # Pre-existing started_at is preserved.
    assert run.started_at == T0


# ---------------------------------------------------------------------------
# Test 9 — Task valid transitions and terminal rejection
# ---------------------------------------------------------------------------


def test_task_pending_to_running_to_completed():
    task = _make_task()
    task.transition_to(TaskStatus.RUNNING)
    assert task.status == TaskStatus.RUNNING
    task.transition_to(TaskStatus.COMPLETED)
    assert task.status == TaskStatus.COMPLETED


def test_task_completed_is_terminal():
    task = _make_task()
    task.transition_to(TaskStatus.RUNNING)
    task.transition_to(TaskStatus.COMPLETED)
    with pytest.raises(InvalidStateTransitionError):
        task.transition_to(TaskStatus.RUNNING)


def test_task_failed_is_terminal():
    task = _make_task()
    task.transition_to(TaskStatus.RUNNING)
    task.transition_to(TaskStatus.FAILED)
    with pytest.raises(InvalidStateTransitionError):
        task.transition_to(TaskStatus.RUNNING)


def test_task_waiting_round_trip():
    task = _make_task()
    task.transition_to(TaskStatus.RUNNING)
    task.transition_to(TaskStatus.WAITING)
    task.transition_to(TaskStatus.RUNNING)
    task.transition_to(TaskStatus.COMPLETED)


def test_task_pending_cannot_jump_to_completed():
    task = _make_task()
    with pytest.raises(InvalidStateTransitionError):
        task.transition_to(TaskStatus.COMPLETED)


# ---------------------------------------------------------------------------
# Test 10 — model validation
# ---------------------------------------------------------------------------


def test_task_empty_goal_rejected():
    with pytest.raises(ValueError):
        Task(task_id="T-x", goal="", created_at=T0)
    with pytest.raises(ValueError):
        Task(task_id="T-x", goal="   ", created_at=T0)


def test_task_empty_id_rejected():
    with pytest.raises(ValueError):
        Task(task_id="", goal="goal", created_at=T0)


def test_run_empty_ids_rejected():
    with pytest.raises(ValueError):
        Run(run_id="", task_id="T-001")
    with pytest.raises(ValueError):
        Run(run_id="R-001", task_id="")


def test_run_negative_fields_rejected():
    with pytest.raises(ValueError):
        Run(run_id="R-001", task_id="T-001", current_step=-1)
    with pytest.raises(ValueError):
        Run(run_id="R-001", task_id="T-001", retry_count=-1)


def test_checkpoint_validation():
    run = _make_run()
    with pytest.raises(ValueError):
        Checkpoint(checkpoint_id="", run_id=run.run_id, step_index=0, created_at=T0)
    with pytest.raises(ValueError):
        Checkpoint(checkpoint_id="C-1", run_id="", step_index=0, created_at=T0)
    with pytest.raises(ValueError):
        Checkpoint(
            checkpoint_id="C-1", run_id=run.run_id, step_index=-1, created_at=T0
        )


def test_naive_datetime_rejected():
    naive = datetime(2026, 1, 1, 0, 0, 0)
    with pytest.raises(ValueError):
        Task(task_id="T-1", goal="g", created_at=naive)
    with pytest.raises(ValueError):
        Checkpoint(
            checkpoint_id="C-1", run_id="R-1", step_index=0, created_at=naive
        )
