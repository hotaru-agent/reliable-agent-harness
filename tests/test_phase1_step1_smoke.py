"""Smoke test for Phase 1 Step 1 — Task / Run / Checkpoint lifecycle.

This test wires the three state models together with the in-memory
checkpoint store to prove the foundation can cooperate end to end. It
uses no Agent, LLM, tool, or runtime loop.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from harness import Checkpoint, Run, RunStatus, Task, TaskStatus
from storage import InMemoryCheckpointStore

T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
T2 = datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)
T3 = datetime(2026, 1, 1, 0, 0, 3, tzinfo=timezone.utc)
T4 = datetime(2026, 1, 1, 0, 0, 4, tzinfo=timezone.utc)


def test_phase1_step1_smoke():
    store = InMemoryCheckpointStore()

    # 1. Create the long-horizon Task.
    task = Task(
        task_id="T-001",
        goal="Reliably diagnose and repair the failing experiment pipeline",
        created_at=T0,
        constraints=("no network", "deterministic"),
        metadata={"owner": "research"},
    )
    assert task.status == TaskStatus.PENDING

    # 2. Task PENDING -> RUNNING.
    task.transition_to(TaskStatus.RUNNING)
    assert task.status == TaskStatus.RUNNING

    # 3. Create a Run for this Task.
    run = Run(run_id="R-001", task_id=task.task_id)
    assert run.task_id == task.task_id
    assert run.status == RunStatus.PENDING

    # 4. Run PENDING -> RUNNING.
    run.transition_to(RunStatus.RUNNING, now=T2)
    assert run.status == RunStatus.RUNNING
    assert run.started_at == T2

    # 5. Checkpoint C-001 at step 0.
    c1 = Checkpoint(
        checkpoint_id="C-001",
        run_id=run.run_id,
        step_index=0,
        created_at=T2,
        agent_state={"phase": "explore"},
        critical_context={"plan": ["gather", "analyze"]},
        completed_actions=(),
        pending_action={"name": "gather"},
    )
    asyncio.run(store.save(c1))
    run.last_checkpoint_id = c1.checkpoint_id
    run.current_step = 1

    # 6. Checkpoint C-002 at step 1.
    c2 = Checkpoint(
        checkpoint_id="C-002",
        run_id=run.run_id,
        step_index=1,
        created_at=T3,
        agent_state={"phase": "analyze"},
        critical_context={"plan": ["gather", "analyze", "report"]},
        completed_actions=("gather",),
        pending_action={"name": "analyze"},
    )
    asyncio.run(store.save(c2))
    run.last_checkpoint_id = c2.checkpoint_id
    run.current_step = 2

    # 7. latest_for_run returns C-002.
    latest = asyncio.run(store.latest_for_run(run.run_id))
    assert latest is not None
    assert latest.checkpoint_id == "C-002"
    assert latest.step_index == 1
    assert latest.run_id == run.run_id

    # 8. Run -> INTERRUPTED (terminal).
    run.transition_to(RunStatus.INTERRUPTED, now=T4)
    assert run.status == RunStatus.INTERRUPTED
    assert run.is_terminal
    assert run.ended_at == T4

    # 9. Task -> WAITING (Run interrupted does not end the Task).
    task.transition_to(TaskStatus.WAITING)
    assert task.status == TaskStatus.WAITING

    # The interrupted Run cannot be revived — a future resume must start
    # a fresh Run from the latest checkpoint rather than flipping this
    # Run back to RUNNING.
    from harness import InvalidStateTransitionError

    raised = False
    try:
        run.transition_to(RunStatus.RUNNING, now=T4)
    except InvalidStateTransitionError:
        raised = True
    assert raised, "INTERRUPTED Run must not be restartable"
