"""Smoke test for Phase 1 Step 2 — full interrupt + resume lifecycle.

Proves the end-to-end guarantee of this phase:

    Task
      -> R-001
        -> step 0 success  -> C-001
        -> step 1 success  -> C-002
        -> step 2 interruption
        -> R-001 INTERRUPTED, Task WAITING
      -> resume into R-002 from C-002
        -> step 2 success  -> C-003
        -> step 3 COMPLETE  -> C-004
        -> R-002 COMPLETED, Task COMPLETED

with R-001 != R-002 and steps 0 / 1 never re-executed.

Fully offline, deterministic, no LLM, no network, no MCP.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from harness import HarnessRuntime, RunStatus, Task, TaskStatus
from storage import InMemoryCheckpointStore
from tests.fakes import FakeClock, FakeStepExecutor, make_id_factory

T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def test_phase1_step2_smoke():
    store = InMemoryCheckpointStore()
    clock = FakeClock(T0)
    run_ids = make_id_factory("R")
    cp_ids = make_id_factory("C")

    task = Task(
        task_id="T-001",
        goal="Reliably diagnose and repair the failing experiment pipeline",
        created_at=T0,
        constraints=("no network", "deterministic"),
    )

    # --- First run: interrupt at step 2. ---
    executor1 = FakeStepExecutor(complete_at_step=3, cancel_at_step=2)
    runtime1 = HarnessRuntime(
        checkpoint_store=store,
        step_executor=executor1,
        clock=clock,
        run_id_factory=run_ids,
        checkpoint_id_factory=cp_ids,
    )

    with pytest.raises(asyncio.CancelledError) as exc_info:
        asyncio.run(runtime1.start(task))
    run1 = exc_info.value.run

    # First run executed steps 0, 1, 2 (step 2 cancelled).
    assert executor1.executed_steps == [0, 1, 2]
    assert run1.run_id == "R-001"
    assert run1.status == RunStatus.INTERRUPTED
    assert task.status == TaskStatus.WAITING
    assert run1.current_step == 2
    assert run1.last_checkpoint_id == "C-002"
    assert run1.resumed_from_checkpoint_id is None

    # Checkpoints C-001 (step 0) and C-002 (step 1) exist; C-003 does not.
    c1 = asyncio.run(store.get("C-001"))
    c2 = asyncio.run(store.get("C-002"))
    assert c1 is not None and c1.step_index == 0
    assert c2 is not None and c2.step_index == 1
    assert asyncio.run(store.get("C-003")) is None

    # --- Resume into a new run from C-002. ---
    executor2 = FakeStepExecutor(complete_at_step=3)
    runtime2 = HarnessRuntime(
        checkpoint_store=store,
        step_executor=executor2,
        clock=clock,
        run_id_factory=run_ids,
        checkpoint_id_factory=cp_ids,
    )

    run2 = asyncio.run(
        runtime2.resume(task=task, source_run=run1, checkpoint_id="C-002")
    )

    # New run is distinct from the old run.
    assert run2.run_id != run1.run_id
    assert run2.run_id == "R-002"

    # Resume did NOT re-execute steps 0 or 1; it started at step 2.
    assert executor2.executed_steps == [2, 3]

    # Lineage.
    assert run2.resumed_from_checkpoint_id == "C-002"
    assert run2.last_checkpoint_id == "C-004"

    # Completion.
    assert run2.status == RunStatus.COMPLETED
    assert task.status == TaskStatus.COMPLETED
    assert run2.current_step == 4

    # The old run remains INTERRUPTED — never revived.
    assert run1.status == RunStatus.INTERRUPTED

    # The resumed steps produced their own checkpoints.
    c3 = asyncio.run(store.get("C-003"))
    c4 = asyncio.run(store.get("C-004"))
    assert c3 is not None and c3.step_index == 2 and c3.run_id == run2.run_id
    assert c4 is not None and c4.step_index == 3 and c4.run_id == run2.run_id

    # Execution state (completed_actions) was restored and extended.
    assert c3.completed_actions == ("step-0", "step-1", "step-2")
    assert c4.completed_actions == ("step-0", "step-1", "step-2", "step-3")
