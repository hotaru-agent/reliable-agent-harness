"""Minimal async harness run loop with basic resume (Phase 1 Step 2).

The ``HarnessRuntime`` drives the explicit Task / Run / Checkpoint state
from Step 1 through a real async loop. It depends on an injectable
``StepExecutor`` (a fake boundary in this phase — NOT a Tool Runtime),
an injectable ``CheckpointStore``, an injectable clock and injectable
ID factories so tests stay fully deterministic, offline and free of real
wall-clock / UUID dependencies.

Core guarantees proven here:

* every successful step (including the COMPLETE step) is checkpointed
  before ``current_step`` advances;
* a failed or cancelled step produces no checkpoint and consumes no
  checkpoint id;
* on ``asyncio.CancelledError`` the Run becomes INTERRUPTED, the Task
  becomes WAITING, and the cancellation is re-raised (never swallowed);
* on an ordinary exception the Run becomes FAILED, the Task becomes
  FAILED, and the original exception is re-raised (no retry);
* resume creates a brand-new Run from a checkpoint — the old Run is
  never revived — and continues from ``checkpoint.step_index + 1`` so
  already-checkpointed steps are never re-executed.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

from opentelemetry.trace import Tracer
from opentelemetry.trace.status import Status, StatusCode

from harness.errors import InvalidRuntimeStateError, ResumeError
from harness.execution import ExecutionState, StepOutcome, StepResult
from harness.state import (
    Checkpoint,
    Run,
    RunStatus,
    Task,
    TaskStatus,
    transition_run_status,
    transition_task_status,
)
from observability.tracing import (
    ATTR_CHECKPOINT_ID,
    ATTR_RESUME_FROM_CHECKPOINT_ID,
    ATTR_RUN_ID,
    ATTR_RUN_RESUME,
    ATTR_RUN_STATUS,
    ATTR_STEP_INDEX,
    ATTR_TASK_ID,
    EVENT_CHECKPOINT_SAVED,
    EVENT_RUN_INTERRUPTED,
    EVENT_RUN_RESUMED,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_INTERRUPTED,
    SPAN_HARNESS_RUN,
    SPAN_HARNESS_STEP,
    resolve_tracer,
    safe_add_event,
    safe_record_exception,
    safe_set_attribute,
    safe_set_status,
    safe_span,
)

# Injectable factory types.
Clock = Callable[[], datetime]
IdFactory = Callable[[], str]


class HarnessRuntime:
    """Drives a Task through Runs, steps and checkpoints.

    All collaborators are injected so the runtime has no hidden
    singletons, no real wall-clock and no random IDs.

    An optional ``tracer`` may be injected for OpenTelemetry tracing.
    When ``None`` (default), a no-op tracer is used so runtime behaviour
    is unaffected. Telemetry failure never becomes business failure.
    """

    def __init__(
        self,
        *,
        checkpoint_store: "CheckpointStore",  # noqa: F821
        step_executor: "StepExecutor",  # noqa: F821
        clock: Clock,
        run_id_factory: IdFactory,
        checkpoint_id_factory: IdFactory,
        tracer: Optional[Tracer] = None,
    ) -> None:
        self._checkpoint_store = checkpoint_store
        self._step_executor = step_executor
        self._clock = clock
        self._run_id_factory = run_id_factory
        self._checkpoint_id_factory = checkpoint_id_factory
        self._tracer = resolve_tracer(tracer)

    # ------------------------------------------------------------------
    # Fresh start
    # ------------------------------------------------------------------

    async def start(self, task: Task) -> Run:
        """Start a fresh Run for a PENDING Task.

        Creates a new Run with empty execution state and drives the run
        loop to completion (or interruption / failure).
        """
        if task.status != TaskStatus.PENDING:
            raise InvalidRuntimeStateError(
                f"start requires a PENDING task, got {task.status.value}"
            )

        # Task PENDING -> RUNNING.
        transition_task_status(task, TaskStatus.RUNNING)

        # Create a fresh Run and start it.
        run = Run(
            run_id=self._run_id_factory(),
            task_id=task.task_id,
            resumed_from_checkpoint_id=None,
        )
        transition_run_status(run, RunStatus.RUNNING, now=self._clock())

        state = ExecutionState()
        with safe_span(self._tracer, SPAN_HARNESS_RUN) as run_span:
            safe_set_attribute(run_span, ATTR_TASK_ID, task.task_id)
            safe_set_attribute(run_span, ATTR_RUN_ID, run.run_id)
            safe_set_attribute(run_span, ATTR_RUN_RESUME, False)
            await self._run_loop(task, run, state, run_span)
        return run

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    async def resume(
        self,
        *,
        task: Task,
        source_run: Run,
        checkpoint_id: str,
    ) -> Run:
        """Resume execution from a checkpoint into a brand-new Run.

        The old Run is never modified: it stays INTERRUPTED. A new Run is
        created with ``current_step = checkpoint.step_index + 1`` and the
        execution state restored from the checkpoint, so already-
        checkpointed steps are not re-executed.
        """
        checkpoint = await self._checkpoint_store.get(checkpoint_id)
        if checkpoint is None:
            raise ResumeError(f"checkpoint {checkpoint_id!r} not found")
        if checkpoint.run_id != source_run.run_id:
            raise ResumeError(
                f"checkpoint {checkpoint_id!r} belongs to run "
                f"{checkpoint.run_id!r}, not {source_run.run_id!r}"
            )
        if source_run.task_id != task.task_id:
            raise ResumeError(
                f"source run {source_run.run_id!r} belongs to task "
                f"{source_run.task_id!r}, not {task.task_id!r}"
            )
        if source_run.status != RunStatus.INTERRUPTED:
            raise ResumeError(
                f"source run {source_run.run_id!r} is "
                f"{source_run.status.value!r}, must be INTERRUPTED to resume"
            )

        # Build the new Run. The old Run is left untouched (stays INTERRUPTED).
        new_run = Run(
            run_id=self._run_id_factory(),
            task_id=task.task_id,
            current_step=checkpoint.step_index + 1,
            resumed_from_checkpoint_id=checkpoint.checkpoint_id,
            # last_checkpoint_id stays None until this Run produces its own.
        )

        # Restore execution state from the (already deep-copied) checkpoint.
        state = ExecutionState.from_checkpoint(checkpoint)

        # Task WAITING -> RUNNING (resume re-enters execution).
        transition_task_status(task, TaskStatus.RUNNING)
        transition_run_status(new_run, RunStatus.RUNNING, now=self._clock())

        with safe_span(self._tracer, SPAN_HARNESS_RUN) as run_span:
            safe_set_attribute(run_span, ATTR_TASK_ID, task.task_id)
            safe_set_attribute(run_span, ATTR_RUN_ID, new_run.run_id)
            safe_set_attribute(run_span, ATTR_RUN_RESUME, True)
            safe_set_attribute(
                run_span, ATTR_RESUME_FROM_CHECKPOINT_ID, checkpoint.checkpoint_id
            )
            safe_add_event(
                run_span,
                EVENT_RUN_RESUMED,
                {
                    ATTR_RESUME_FROM_CHECKPOINT_ID: checkpoint.checkpoint_id,
                    ATTR_STEP_INDEX: new_run.current_step,
                },
            )
            await self._run_loop(task, new_run, state, run_span)
        return new_run

    # ------------------------------------------------------------------
    # The async run loop
    # ------------------------------------------------------------------

    async def _run_loop(
        self,
        task: Task,
        run: Run,
        state: ExecutionState,
        run_span: Any,
    ) -> None:
        """Execute steps sequentially, checkpointing each successful one.

        Strictly sequential: step N must be checkpointed before step N+1
        starts. No concurrency, no gather, no DAG.

        ``run_span`` is the active ``harness.run`` span; step spans are
        created as children of it.
        """
        while run.status == RunStatus.RUNNING:
            step_index = run.current_step

            with safe_span(self._tracer, SPAN_HARNESS_STEP) as step_span:
                safe_set_attribute(step_span, ATTR_TASK_ID, task.task_id)
                safe_set_attribute(step_span, ATTR_RUN_ID, run.run_id)
                safe_set_attribute(step_span, ATTR_STEP_INDEX, step_index)

                try:
                    result: StepResult = await self._step_executor.execute_step(
                        task, run, state
                    )
                except asyncio.CancelledError as exc:
                    # Step produced no checkpoint; consume no checkpoint id.
                    transition_run_status(
                        run, RunStatus.INTERRUPTED, now=self._clock()
                    )
                    transition_task_status(task, TaskStatus.WAITING)
                    # Attach the run so callers that catch the re-raised
                    # cancellation can still inspect the final run state.
                    exc.run = run  # type: ignore[attr-defined]
                    safe_add_event(
                        step_span,
                        EVENT_RUN_INTERRUPTED,
                        {ATTR_STEP_INDEX: step_index},
                    )
                    safe_add_event(
                        run_span,
                        EVENT_RUN_INTERRUPTED,
                        {ATTR_STEP_INDEX: step_index},
                    )
                    safe_set_attribute(run_span, ATTR_RUN_STATUS, RUN_STATUS_INTERRUPTED)
                    safe_set_status(step_span, Status(StatusCode.UNSET))
                    safe_set_status(run_span, Status(StatusCode.UNSET))
                    raise
                except Exception as exc:
                    # Ordinary failure: no retry in this phase.
                    transition_run_status(run, RunStatus.FAILED, now=self._clock())
                    transition_task_status(task, TaskStatus.FAILED)
                    exc.run = run  # type: ignore[attr-defined]
                    safe_record_exception(step_span, exc)
                    safe_set_status(step_span, Status(StatusCode.ERROR))
                    safe_record_exception(run_span, exc)
                    safe_set_attribute(run_span, ATTR_RUN_STATUS, RUN_STATUS_FAILED)
                    safe_set_status(run_span, Status(StatusCode.ERROR))
                    raise

                # Step succeeded — generate a checkpoint id only now, build the
                # checkpoint, persist it, then advance current_step.
                checkpoint_id = self._checkpoint_id_factory()
                fields = result.state.to_checkpoint_fields()
                checkpoint = Checkpoint(
                    checkpoint_id=checkpoint_id,
                    run_id=run.run_id,
                    step_index=step_index,
                    created_at=self._clock(),
                    agent_state=fields["agent_state"],
                    critical_context=fields["critical_context"],
                    tool_state=fields["tool_state"],
                    completed_actions=fields["completed_actions"],
                    pending_action=fields["pending_action"],
                )
                await self._checkpoint_store.save(checkpoint)
                run.last_checkpoint_id = checkpoint_id
                run.current_step = step_index + 1
                state = result.state

                safe_add_event(
                    step_span,
                    EVENT_CHECKPOINT_SAVED,
                    {
                        ATTR_CHECKPOINT_ID: checkpoint_id,
                        ATTR_STEP_INDEX: step_index,
                    },
                )
                safe_set_status(step_span, Status(StatusCode.OK))

                if result.outcome == StepOutcome.COMPLETE:
                    transition_run_status(
                        run, RunStatus.COMPLETED, now=self._clock()
                    )
                    transition_task_status(task, TaskStatus.COMPLETED)
                    safe_set_attribute(run_span, ATTR_RUN_STATUS, RUN_STATUS_COMPLETED)
                    safe_set_status(run_span, Status(StatusCode.OK))
                    return
