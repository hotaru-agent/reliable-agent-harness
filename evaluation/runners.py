"""Comparable benchmark runners (Phase 7 Step 2).

Provides two runners that execute the same ``BenchmarkScenario`` under
different reliability policies:

- ``NaiveBenchmarkRunner`` — a controlled ablation baseline. It runs a
  sequential scripted action loop through the existing ``ToolRegistry`` /
  ``ToolRuntime`` contract but with ``max_attempts=1`` (no automatic
  retry). The first failed tool action terminates the run. No
  HarnessRuntime, no checkpoints, no resume, no loop detection, no
  replan, no context budget, no externalization.

- ``ReliableHarnessBenchmarkRunner`` — uses the real ``HarnessRuntime``
  with ``ToolRuntime`` configured for normal retry. A thin
  ``BenchmarkStepExecutor`` adapts the scripted action source to the
  existing ``StepExecutor`` protocol. Checkpoints come from real
  HarnessRuntime execution. Retry comes only from the existing
  ``RetryPolicy`` — no benchmark-specific retry loops.

Both runners:
- receive the same immutable ``BenchmarkScenario`` blueprint
- create fresh repository / fault injector / scripted action source
- use ``FaultAwareToolInvoker`` for non-empty fault plans
- evaluate completion through ``RepositoryOracle`` (never self-report)
- return a ``BenchmarkRunRecord`` built from real execution results

This is a controlled mechanism validation, NOT a real-world benchmark.
``ScriptedActionSource`` is a deterministic driver, NOT an AI agent.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol

from evaluation.faults import FaultInjector, FaultPlan, InjectedProcessInterruption
from evaluation.invoker import FaultAwareToolInvoker
from evaluation.models import BenchmarkScenario, BenchmarkTask
from evaluation.records import BenchmarkEvent, BenchmarkEventType, BenchmarkRunRecord
from evaluation.repository import RepositoryFixture, RepositoryOracle
from evaluation.scenarios.controlled_repo import (
    ScriptedActionSource,
    make_fix_calculator_actions,
    make_repository_tool_handlers,
    make_repository_tool_specs,
    make_standard_fault_plans,
    make_standard_fixtures,
)
from evaluation.workloads import (
    ToolStack,
    WorkloadRegistry,
    make_default_workload_registry,
)
from evaluation.context_workload import (
    ContextBenchmarkStepExecutor,
    ContextLimitExceededError as _ContextLimitFailure,
    ContextWorkload,
)
from harness.execution import ExecutionState, StepOutcome, StepResult
from harness.runtime import HarnessRuntime
from harness.state import Run, RunStatus, Task, TaskStatus
from storage.checkpoint_store import InMemoryCheckpointStore
from tools import (
    RetryPolicy,
    ToolCall,
    ToolExecutionContext,
    ToolRegistry,
    ToolRuntime,
)
from tools.models import ToolErrorType


# ---------------------------------------------------------------------------
# Injectable sleep / clock / id factories
# ---------------------------------------------------------------------------

SleepFn = Callable[[float], "Any"]


class _DeterministicClock:
    """Injectable clock returning strictly increasing UTC datetimes.

    Each call advances by one second so timestamps are distinct and
    deterministic.
    """

    def __init__(self, start: datetime | None = None) -> None:
        from datetime import timedelta

        self._t = start or datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        self._delta = timedelta(seconds=1)

    def __call__(self) -> datetime:
        result = self._t
        self._t = self._t + self._delta
        return result


class _FakeSleeper:
    """Deterministic sleep that records durations without sleeping.

    Used for ToolRuntime backoff so retries are instant in tests. The
    timeout fault inside the handler uses real ``asyncio.sleep`` so the
    ``asyncio.wait_for`` timeout boundary is exercised for real.
    """

    def __init__(self) -> None:
        self.sleeps: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def _make_id_factory(prefix: str, start: int = 1) -> Callable[[], str]:
    """Return a callable producing ids like ``R-001``, ``R-002``, ..."""
    counter = start - 1

    def _factory() -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}-{counter:03d}"

    return _factory


# ---------------------------------------------------------------------------
# Failure reasons (stable, structured, no raw exception messages)
# ---------------------------------------------------------------------------

FAILURE_TRANSIENT = "TRANSIENT"
FAILURE_PERMANENT = "PERMANENT"
FAILURE_TIMEOUT = "TIMEOUT"
FAILURE_EXECUTION = "EXECUTION"
FAILURE_MAX_ACTIONS = "MAX_ACTIONS"
FAILURE_ORACLE_INCOMPLETE = "ORACLE_INCOMPLETE"
FAILURE_INTERRUPTED = "INTERRUPTED"
FAILURE_NO_CHECKPOINT = "NO_CHECKPOINT"
FAILURE_CONTEXT_LIMIT = "CONTEXT_LIMIT"

# The namespace key used to store the scripted action source cursor in
# ``ExecutionState.agent_state`` so it survives checkpoint / resume.
SCRIPT_CURSOR_KEY = "benchmark.script_next_index"

# Map ToolErrorType to stable failure reason strings.
_ERROR_TYPE_TO_FAILURE: dict[ToolErrorType, str] = {
    ToolErrorType.TRANSIENT: FAILURE_TRANSIENT,
    ToolErrorType.PERMANENT: FAILURE_PERMANENT,
    ToolErrorType.TIMEOUT: FAILURE_TIMEOUT,
    ToolErrorType.EXECUTION: FAILURE_EXECUTION,
    ToolErrorType.NOT_FOUND: "NOT_FOUND",
    ToolErrorType.VALIDATION: "VALIDATION",
    ToolErrorType.PERMISSION: "PERMISSION",
}


def _failure_reason_from_error(error_type: ToolErrorType) -> str:
    return _ERROR_TYPE_TO_FAILURE.get(error_type, error_type.value.upper())


# ---------------------------------------------------------------------------
# Trial state (for deterministic test verification of recovery semantics)
# ---------------------------------------------------------------------------


@dataclass
class TrialState:
    """Internal state from the last benchmark trial.

    This is NOT part of the benchmark contract — it exists for
    deterministic test verification of recovery semantics (source Run
    status, resume Run lineage, checkpoint store contents, etc.). It is
    populated by ``ReliableHarnessBenchmarkRunner`` after each ``run()``
    and reset at the start of the next ``run()``.
    """

    source_run: Optional[Run] = None
    resume_run: Optional[Run] = None
    task: Optional[Task] = None
    checkpoint_store: Optional[InMemoryCheckpointStore] = None
    source_executor: Optional["BenchmarkStepExecutor"] = None
    resume_executor: Optional["BenchmarkStepExecutor"] = None
    injector: Optional[FaultInjector] = None
    repo: Optional[Any] = None
    interruption_occurred: bool = False


# ---------------------------------------------------------------------------
# Benchmark runner protocol
# ---------------------------------------------------------------------------


class BenchmarkRunner(Protocol):
    """Protocol for a comparable benchmark runner.

    A runner receives an immutable ``BenchmarkScenario`` blueprint and
    returns a ``BenchmarkRunRecord`` built from real execution. The
    runner must NOT mutate the scenario. Completion is always determined
    by ``RepositoryOracle``, never self-reported.
    """

    async def run(self, scenario: BenchmarkScenario) -> BenchmarkRunRecord:
        ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_fixture(
    fixtures: dict[str, RepositoryFixture],
    scenario: BenchmarkScenario,
) -> RepositoryFixture:
    fixture = fixtures.get(scenario.fixture_id)
    if fixture is None:
        raise KeyError(f"unknown fixture_id: {scenario.fixture_id!r}")
    return fixture


def _resolve_fault_plan(
    fault_plans: dict[str, FaultPlan],
    scenario: BenchmarkScenario,
) -> FaultPlan:
    plan = fault_plans.get(scenario.fault_plan_id)
    if plan is None:
        raise KeyError(f"unknown fault_plan_id: {scenario.fault_plan_id!r}")
    return plan


def _build_tool_stack(
    repo,
    injector: FaultInjector,
    *,
    run_tests_retry_policy: Optional[RetryPolicy],
    run_tests_timeout_seconds: float,
    sleep: SleepFn,
) -> tuple[ToolRuntime, FaultAwareToolInvoker, ToolExecutionContext]:
    """Build a ToolRegistry + ToolRuntime + FaultAwareToolInvoker.

    When ``run_tests_retry_policy`` is None, the standard retry policy
    from ``make_repository_tool_specs`` is used (max_attempts=3).
    """
    registry = ToolRegistry()
    specs = make_repository_tool_specs(
        run_tests_retry_policy=run_tests_retry_policy,
        run_tests_timeout_seconds=run_tests_timeout_seconds,
    )
    handlers = make_repository_tool_handlers(repo, injector=injector)
    for name in specs:
        registry.register(specs[name], handlers[name])
    runtime = ToolRuntime(registry=registry, sleep=sleep)
    invoker = FaultAwareToolInvoker(runtime, injector=injector)
    ctx = ToolExecutionContext()
    return runtime, invoker, ctx


# ---------------------------------------------------------------------------
# Naive benchmark runner
# ---------------------------------------------------------------------------


class NaiveBenchmarkRunner:
    """Controlled ablation baseline runner.

    Executes a scripted action loop through the existing ``ToolRegistry``
    / ``ToolRuntime`` contract with ``max_attempts=1`` (no automatic
    retry). The first failed tool action terminates the run.

    This is NOT a bad implementation — it is a deliberate ablation that
    models a sequential tool-using loop without reliability recovery.
    It still goes through the full tool contract (validation, permissions,
    timeout boundary, error normalization) so the comparison is about
    reliability features, not tool protocol differences.

    Semantics:
    - No HarnessRuntime, no checkpoints, no resume.
    - No loop detection, no replan, no context budget, no externalization.
    - ``max_attempts=1`` for all benchmark tools — no automatic retry.
    - First failed ``ToolExecutionResult`` terminates the run.
    - ``checkpoint_count == 0`` always.
    - ``retry_count == 0`` always.
    - Completion determined by ``RepositoryOracle``.
    - ``max_logical_actions`` respected.
    """

    def __init__(
        self,
        *,
        fixtures: Optional[dict[str, RepositoryFixture]] = None,
        fault_plans: Optional[dict[str, FaultPlan]] = None,
        action_factory: Optional[Callable[[], list]] = None,
        run_tests_timeout_seconds: float = 5.0,
        sleep: Optional[SleepFn] = None,
        workload_registry: Optional[WorkloadRegistry] = None,
    ) -> None:
        self._fixtures = fixtures if fixtures is not None else make_standard_fixtures()
        self._fault_plans = (
            fault_plans if fault_plans is not None else make_standard_fault_plans()
        )
        self._action_factory = action_factory or make_fix_calculator_actions
        self._run_tests_timeout_seconds = run_tests_timeout_seconds
        self._sleep = sleep if sleep is not None else asyncio.sleep
        self._workload_registry = workload_registry or make_default_workload_registry(
            self._action_factory
        )

    async def run(self, scenario: BenchmarkScenario) -> BenchmarkRunRecord:
        fixture = _resolve_fixture(self._fixtures, scenario)
        fault_plan = _resolve_fault_plan(self._fault_plans, scenario)

        # Fresh state per run.
        repo = fixture.create_repository()
        injector = FaultInjector(fault_plan)
        workload = self._workload_registry.get(scenario.scenario_id)
        oracle = RepositoryOracle()

        # Naive retry policy: max_attempts=1 (no automatic retry).
        naive_policy = RetryPolicy(max_attempts=1)
        _, invoker, ctx = _build_tool_stack(
            repo,
            injector,
            run_tests_retry_policy=naive_policy,
            run_tests_timeout_seconds=self._run_tests_timeout_seconds,
            sleep=self._sleep,
        )

        max_actions = scenario.task.max_logical_actions

        # Context scenarios use the NaiveContextExecutor (append-all /
        # inline-all ablation) with the same context-capacity parameters
        # as the Reliable executor.
        from evaluation.context_workload import ContextWorkload

        if isinstance(workload, ContextWorkload):
            return await self._run_context_scenario(
                scenario=scenario,
                workload=workload,
                invoker=invoker,
                ctx=ctx,
                repo=repo,
                oracle=oracle,
                max_actions=max_actions,
            )

        # Standard scripted loop (non-context scenarios).
        source = workload.create_source()
        events: list[BenchmarkEvent] = []
        seq = 0
        logical_action_count = 0
        tool_invocation_count = 0
        tool_attempt_count = 0
        retry_count = 0
        failure_reason: Optional[str] = None

        while source.has_next():
            if logical_action_count >= max_actions:
                failure_reason = FAILURE_MAX_ACTIONS
                break

            call = source.next_call()
            logical_action_count += 1
            tool_invocation_count += 1

            try:
                result = await invoker.execute(call, ctx)
            except InjectedProcessInterruption:
                # Process interruption is a control signal, NOT a tool
                # error. The naive runner has no Harness checkpoint /
                # resume mechanism, so it terminates. Do NOT convert
                # this to a PERMANENT tool error — it is an independent
                # control signal.
                tool_attempt_count += 1  # one handler attempt was made
                events.append(BenchmarkEvent(
                    event_type=BenchmarkEventType.INTERRUPTION,
                    sequence=seq,
                    tool_name=call.tool_name,
                    success=False,
                    details={},
                ))
                seq += 1
                failure_reason = FAILURE_INTERRUPTED
                break

            tool_attempt_count += result.attempt_count
            retry_count += max(0, result.attempt_count - 1)

            events.append(BenchmarkEvent(
                event_type=BenchmarkEventType.LOGICAL_ACTION,
                sequence=seq,
                tool_name=call.tool_name,
                success=result.success,
                details={"attempt_count": result.attempt_count},
            ))
            seq += 1

            if not result.success:
                failure_reason = _failure_reason_from_error(
                    result.error.error_type
                )
                break

        # Oracle determines completion (never self-reported).
        completed = oracle.is_complete(repo)
        if not completed and failure_reason is None:
            failure_reason = FAILURE_ORACLE_INCOMPLETE

        return BenchmarkRunRecord(
            scenario_id=scenario.scenario_id,
            task_id=scenario.task.task_id,
            completed=completed,
            logical_action_count=logical_action_count,
            tool_invocation_count=tool_invocation_count,
            tool_attempt_count=tool_attempt_count,
            retry_count=retry_count,
            loop_detection_count=0,
            replan_count=0,
            checkpoint_count=0,
            failure_reason=failure_reason if not completed else None,
            events=tuple(events),
            wall_clock_seconds=0.0,
        )

    async def _run_context_scenario(
        self,
        *,
        scenario: BenchmarkScenario,
        workload: "ContextWorkload",
        invoker: FaultAwareToolInvoker,
        ctx: ToolExecutionContext,
        repo,
        oracle: RepositoryOracle,
        max_actions: int,
    ) -> BenchmarkRunRecord:
        """Run a context scenario with the NaiveContextExecutor."""
        from evaluation.context_workload import ContextLimitExceededError

        source = workload.create_source()
        executor = workload.create_naive_executor(
            source=source,
            invoker=invoker,
            ctx=ctx,
            max_logical_actions=max_actions,
        )

        failure_reason: Optional[str] = None
        try:
            while source.has_next():
                if executor.logical_action_count >= max_actions:
                    failure_reason = FAILURE_MAX_ACTIONS
                    break
                call = source.next_call()
                executor.logical_action_count += 1
                executor.tool_invocation_count += 1

                # Check context capacity BEFORE the tool call (decision
                # boundary). This mirrors the executor's own check but
                # is done here because the naive runner does not use the
                # StepExecutor protocol.
                total_tokens = executor._total_context_tokens()
                if total_tokens > executor.trial_state.peak_context_tokens:
                    executor.trial_state.peak_context_tokens = total_tokens
                if total_tokens > executor._max_context_tokens:
                    executor.trial_state.context_limit_exceeded = True
                    executor.events.append(BenchmarkEvent(
                        event_type=BenchmarkEventType.CONTEXT_LIMIT_EXCEEDED,
                        sequence=len(executor.events),
                        tool_name=None,
                        success=False,
                        details={
                            "total_tokens": total_tokens,
                            "max_tokens": executor._max_context_tokens,
                        },
                    ))
                    failure_reason = FAILURE_CONTEXT_LIMIT
                    break

                try:
                    result = await invoker.execute(call, ctx)
                except InjectedProcessInterruption:
                    executor.tool_attempt_count += 1
                    executor.events.append(BenchmarkEvent(
                        event_type=BenchmarkEventType.INTERRUPTION,
                        sequence=len(executor.events),
                        tool_name=call.tool_name,
                        success=False,
                        details={},
                    ))
                    failure_reason = FAILURE_INTERRUPTED
                    break

                executor.tool_attempt_count += result.attempt_count
                executor.retry_count += max(0, result.attempt_count - 1)
                executor.events.append(BenchmarkEvent(
                    event_type=BenchmarkEventType.LOGICAL_ACTION,
                    sequence=len(executor.events),
                    tool_name=call.tool_name,
                    success=result.success,
                    details={"attempt_count": result.attempt_count},
                ))

                if not result.success:
                    failure_reason = _failure_reason_from_error(
                        result.error.error_type
                    )
                    break

                # Append full output inline (append-all).
                rendered = executor._renderer.render(result.output)
                tokens = executor._estimator.estimate(rendered)
                if not rendered:
                    tokens = 0
                executor._context_item_id_counter += 1
                from harness.context import ContextItem, ContextKind, ContextPriority
                inline_item = ContextItem(
                    item_id=f"naive_ctx_{executor._context_item_id_counter:04d}",
                    kind=ContextKind.RECENT_INTERACTION,
                    content=rendered,
                    estimated_tokens=tokens,
                    sequence_index=executor._next_seq,
                    priority=ContextPriority.NORMAL,
                    must_keep=False,
                )
                executor._next_seq += 1
                executor._context_items.append(inline_item)
                executor.trial_state.processed_output_count += 1
                executor.trial_state.context_items = list(executor._context_items)
        except ContextLimitExceededError:
            failure_reason = FAILURE_CONTEXT_LIMIT

        completed = oracle.is_complete(repo)
        if not completed and failure_reason is None:
            failure_reason = FAILURE_ORACLE_INCOMPLETE

        ts = executor.trial_state
        return BenchmarkRunRecord(
            scenario_id=scenario.scenario_id,
            task_id=scenario.task.task_id,
            completed=completed,
            logical_action_count=executor.logical_action_count,
            tool_invocation_count=executor.tool_invocation_count,
            tool_attempt_count=executor.tool_attempt_count,
            retry_count=executor.retry_count,
            loop_detection_count=0,
            replan_count=0,
            checkpoint_count=0,
            externalized_output_count=ts.externalized_output_count,
            processed_output_count=ts.processed_output_count,
            peak_context_tokens=ts.peak_context_tokens,
            failure_reason=failure_reason if not completed else None,
            events=tuple(executor.events),
            wall_clock_seconds=0.0,
        )


# ---------------------------------------------------------------------------
# Reliable harness runner — step executor adapter
# ---------------------------------------------------------------------------


class BenchmarkStepFailure(Exception):
    """Raised by ``BenchmarkStepExecutor`` when a tool logical invocation
    fails after all allowed retry attempts.

    Carries the structured ``ToolErrorType`` so the runner can map it to
    a stable failure reason without parsing exception messages.
    """

    def __init__(self, error_type: ToolErrorType, tool_name: str) -> None:
        self.error_type = error_type
        self.tool_name = tool_name
        super().__init__(f"tool {tool_name!r} failed: {error_type.value}")


class MaxActionsExceededError(Exception):
    """Raised when ``max_logical_actions`` is exceeded during a run."""


class BenchmarkStepExecutor:
    """Adapts ``ScriptedActionSource`` to the ``StepExecutor`` protocol.

    One scripted action = one Harness logical step. ToolRuntime retry
    attempts within a single ``FaultAwareToolInvoker.execute()`` call
    belong to the SAME logical step — they never become new steps.

    On tool success: returns ``StepResult(CONTINUE, state)`` (or
    ``COMPLETE`` when the script is exhausted). The script cursor
    (``benchmark.script_next_index``) is stored in
    ``ExecutionState.agent_state`` so it survives checkpoint / resume.

    On tool failure (after all retries): raises ``BenchmarkStepFailure``
    so the HarnessRuntime marks the Run FAILED and re-raises. The run
    terminates deterministically — remaining scripted repair actions are
    NOT executed.

    On ``InjectedProcessInterruption`` (a benchmark control signal, NOT
    a tool error): maps it to ``asyncio.CancelledError`` so the
    HarnessRuntime's existing cancellation boundary fires (Run →
    INTERRUPTED, Task → WAITING). The interrupted ``Run`` reference is
    captured in ``self.interrupted_run`` for the runner to use.

    On ``max_logical_actions`` exceeded: raises ``MaxActionsExceededError``.

    Accumulates execution stats (logical_action_count, tool_attempt_count,
    etc.) and benchmark events for the runner to read after the run.

    Resume support:
    - ``initial_script_index`` seeks the fresh source to the restored
      cursor (from a checkpointed ``ExecutionState``).
    - At the start of each step, the executor verifies that
      ``run.current_step`` matches the restored script cursor
      (``state.agent_state[SCRIPT_CURSOR_KEY]``). If they disagree, it
      fails fast rather than silently picking one.
    """

    def __init__(
        self,
        *,
        source: ScriptedActionSource,
        invoker: FaultAwareToolInvoker,
        ctx: ToolExecutionContext,
        max_logical_actions: int,
        initial_script_index: int = 0,
    ) -> None:
        self._source = source
        # Seek the fresh source to the restored cursor (for resume).
        if initial_script_index > 0:
            self._source.seek(initial_script_index)
        self._invoker = invoker
        self._ctx = ctx
        self._max_logical_actions = max_logical_actions
        # Accumulated stats (read by the runner after the run).
        self.logical_action_count: int = 0
        self.tool_invocation_count: int = 0
        self.tool_attempt_count: int = 0
        self.retry_count: int = 0
        self.successful_step_count: int = 0
        # Loop/replan stats (always 0 for scripted executor; present so
        # the runner can aggregate uniformly across executor types).
        self.loop_detection_count: int = 0
        self.replan_count: int = 0
        self.blocked_call_count: int = 0
        self.events: list[BenchmarkEvent] = []
        self._seq: int = 0
        # Set when InjectedProcessInterruption is mapped to CancelledError.
        self.interrupted_run: Optional[Run] = None
        # Execution history: (run_id, step_index, tool_name, outcome)
        # for deterministic verification. This is benchmark accounting,
        # NOT a second tracing system.
        self.execution_history: list[tuple[str, int, str, str]] = []

    async def execute_step(self, task, run: Run, state: ExecutionState) -> StepResult:
        # Verify cursor consistency: the restored script cursor (from
        # checkpointed agent_state) must match the Harness run's
        # current_step. If they disagree, fail fast.
        restored_cursor = state.agent_state.get(SCRIPT_CURSOR_KEY, 0)
        if restored_cursor != run.current_step:
            raise RuntimeError(
                f"cursor mismatch: restored script_next_index={restored_cursor}, "
                f"run.current_step={run.current_step}"
            )

        # Script exhausted → COMPLETE (this is a real Harness step that
        # gets checkpointed by the HarnessRuntime, just like CONTINUE
        # steps). Count it as a successful step so checkpoint_count
        # reflects real Harness execution.
        if not self._source.has_next():
            self.successful_step_count += 1
            return StepResult(StepOutcome.COMPLETE, state)

        # Max actions exceeded (there are more actions, but we hit the
        # limit). This terminates the run with MAX_ACTIONS.
        if self.logical_action_count >= self._max_logical_actions:
            raise MaxActionsExceededError(
                f"max_logical_actions ({self._max_logical_actions}) exceeded"
            )

        call = self._source.next_call()
        self.logical_action_count += 1
        self.tool_invocation_count += 1

        try:
            result = await self._invoker.execute(call, self._ctx)
        except InjectedProcessInterruption:
            # Map the benchmark process-control signal to the Harness
            # cancellation boundary. The HarnessRuntime catches
            # asyncio.CancelledError and transitions Run → INTERRUPTED,
            # Task → WAITING. This is evaluation-adapter behavior —
            # production HarnessRuntime is NOT modified to recognize
            # benchmark exceptions.
            self.interrupted_run = run
            # The handler was called once (it raised the control signal).
            # Count it as one attempt for accounting.
            self.tool_attempt_count += 1
            self.events.append(BenchmarkEvent(
                event_type=BenchmarkEventType.INTERRUPTION,
                sequence=self._seq,
                tool_name=call.tool_name,
                success=False,
                details={"step_index": run.current_step},
            ))
            self._seq += 1
            self.execution_history.append(
                (run.run_id, run.current_step, call.tool_name, "interrupted")
            )
            raise asyncio.CancelledError()

        self.tool_attempt_count += result.attempt_count
        self.retry_count += max(0, result.attempt_count - 1)

        self.events.append(BenchmarkEvent(
            event_type=BenchmarkEventType.LOGICAL_ACTION,
            sequence=self._seq,
            tool_name=call.tool_name,
            success=result.success,
            details={"attempt_count": result.attempt_count},
        ))
        self._seq += 1

        if not result.success:
            self.execution_history.append(
                (run.run_id, run.current_step, call.tool_name, "failed")
            )
            raise BenchmarkStepFailure(result.error.error_type, call.tool_name)

        # Success — grow completed_actions, store the script cursor,
        # and return CONTINUE. The cursor (next index to execute) is
        # stored in agent_state so it survives checkpoint / resume.
        new_agent_state = dict(state.agent_state)
        new_agent_state[SCRIPT_CURSOR_KEY] = self._source.current_index
        new_state = ExecutionState(
            agent_state=new_agent_state,
            critical_context=dict(state.critical_context),
            tool_state=dict(state.tool_state),
            completed_actions=state.completed_actions + (call.tool_name,),
            pending_action=state.pending_action,
        )
        self.successful_step_count += 1
        self.execution_history.append(
            (run.run_id, run.current_step, call.tool_name, "success")
        )
        return StepResult(StepOutcome.CONTINUE, new_state)


# ---------------------------------------------------------------------------
# Reliable harness benchmark runner
# ---------------------------------------------------------------------------


class ReliableHarnessBenchmarkRunner:
    """Reliable runner that genuinely uses ``HarnessRuntime``.

    Executes the scripted maintenance workload through the real
    ``HarnessRuntime`` with ``ToolRuntime`` configured for normal retry.
    A thin ``BenchmarkStepExecutor`` adapts the scripted action source to
    the existing ``StepExecutor`` protocol.

    Retry comes ONLY from the existing ``RetryPolicy`` / ``ToolError.retryable``
    / ``ToolSideEffect`` safety mechanism. There is no benchmark-specific
    retry loop.

    Recovery (Phase 7 Step 3):
    - On ``InjectedProcessInterruption``, the executor maps it to
      ``asyncio.CancelledError`` so the HarnessRuntime transitions the
      source Run to INTERRUPTED and the Task to WAITING.
    - The runner catches the cancellation, verifies the source Run is
      INTERRUPTED, loads the real checkpoint from the store, and calls
      the real ``HarnessRuntime.resume()`` with a NEW HarnessRuntime
      instance (same CheckpointStore) to prove recovery does not depend
      on the old runtime's local loop state.
    - The same ``FaultInjector`` spans source + resume Runs so the
      one-shot ``PROCESS_INTERRUPTION`` fault is NOT re-fired.
    - The same ``ControlledRepository`` spans source + resume Runs so
      checkpoint state and workspace state stay consistent.
    - A fresh ``ScriptedActionSource`` is seeked to the restored cursor
      (from the checkpointed ``ExecutionState``) for the resume Run.
    - Already-checkpointed steps are NOT re-executed; the interrupted
      step IS re-executed (it had no checkpoint).

    Semantics:
    - Uses real ``HarnessRuntime`` with ``InMemoryCheckpointStore``.
    - One scripted action = one Harness logical step.
    - ToolRuntime retries within one logical step do NOT become new steps.
    - Successful logical steps produce real checkpoints (counted from
      real execution, not faked).
    - On tool failure (after all retries): the step raises, the Run
      becomes FAILED, and the run terminates.
    - Permanent failures are NOT retried (existing RetryPolicy semantics).
    - ``checkpoint_count`` comes from real ``successful_step_count``
      aggregated across source + resume Runs.
    - Completion determined by ``RepositoryOracle`` — Harness COMPLETED
      is NOT oracle completion.
    - ``max_logical_actions`` respected.
    - ``AgentActionController`` integration deferred to loop/replan step.
    """

    def __init__(
        self,
        *,
        fixtures: Optional[dict[str, RepositoryFixture]] = None,
        fault_plans: Optional[dict[str, FaultPlan]] = None,
        action_factory: Optional[Callable[[], list]] = None,
        run_tests_timeout_seconds: float = 5.0,
        sleep: Optional[SleepFn] = None,
        clock: Optional[Callable[[], datetime]] = None,
        run_id_factory: Optional[Callable[[], str]] = None,
        checkpoint_id_factory: Optional[Callable[[], str]] = None,
        workload_registry: Optional[WorkloadRegistry] = None,
    ) -> None:
        self._fixtures = fixtures if fixtures is not None else make_standard_fixtures()
        self._fault_plans = (
            fault_plans if fault_plans is not None else make_standard_fault_plans()
        )
        self._action_factory = action_factory or make_fix_calculator_actions
        self._run_tests_timeout_seconds = run_tests_timeout_seconds
        self._sleep = sleep if sleep is not None else asyncio.sleep
        self._clock = clock if clock is not None else _DeterministicClock()
        self._run_id_factory = run_id_factory or _make_id_factory("R")
        self._checkpoint_id_factory = checkpoint_id_factory or _make_id_factory("C")
        self._workload_registry = workload_registry or make_default_workload_registry(
            self._action_factory
        )
        # Internal state from the last trial (for test verification).
        self.last_trial_state: TrialState = TrialState()

    async def run(self, scenario: BenchmarkScenario) -> BenchmarkRunRecord:
        # Reset trial state for this run.
        self.last_trial_state = TrialState()
        fixture = _resolve_fixture(self._fixtures, scenario)
        fault_plan = _resolve_fault_plan(self._fault_plans, scenario)

        # Fresh trial state. The repo and injector span source + resume
        # Runs within this trial (they must NOT be recreated on resume).
        repo = fixture.create_repository()
        injector = FaultInjector(fault_plan)
        oracle = RepositoryOracle()

        # Populate trial state for test inspection.
        self.last_trial_state.injector = injector
        self.last_trial_state.repo = repo

        # Normal retry policy (max_attempts=3) — the standard from
        # make_repository_tool_specs. Pass None to use the default.
        runtime, invoker, ctx = _build_tool_stack(
            repo,
            injector,
            run_tests_retry_policy=None,
            run_tests_timeout_seconds=self._run_tests_timeout_seconds,
            sleep=self._sleep,
        )

        # Build the source HarnessRuntime with real collaborators.
        checkpoint_store = InMemoryCheckpointStore()
        workload = self._workload_registry.get(scenario.scenario_id)
        source = workload.create_source()
        tool_stack = ToolStack(
            runtime=runtime,
            invoker=invoker,
            ctx=ctx,
        )
        source_executor = workload.create_step_executor(
            source=source,
            tool_stack=tool_stack,
            repo=repo,
            max_logical_actions=scenario.task.max_logical_actions,
        )
        source_harness = HarnessRuntime(
            checkpoint_store=checkpoint_store,
            step_executor=source_executor,
            clock=self._clock,
            run_id_factory=self._run_id_factory,
            checkpoint_id_factory=self._checkpoint_id_factory,
        )

        # Create the harness Task (PENDING).
        task = Task(
            task_id=scenario.task.task_id,
            goal=scenario.task.goal,
            created_at=self._clock(),
        )

        # Populate trial state with task / store / source executor.
        self.last_trial_state.task = task
        self.last_trial_state.checkpoint_store = checkpoint_store
        self.last_trial_state.source_executor = source_executor

        # Execute the source Run through the real HarnessRuntime.
        failure_reason: Optional[str] = None
        interruption_occurred = False
        source_run: Optional[Run] = None
        try:
            await source_harness.start(task)
        except asyncio.CancelledError:
            # The executor mapped InjectedProcessInterruption to
            # CancelledError. The HarnessRuntime has already transitioned
            # the source Run to INTERRUPTED and the Task to WAITING.
            interruption_occurred = True
            source_run = source_executor.interrupted_run
            self.last_trial_state.source_run = source_run
            self.last_trial_state.interruption_occurred = True
        except BenchmarkStepFailure as exc:
            failure_reason = _failure_reason_from_error(exc.error_type)
        except MaxActionsExceededError:
            failure_reason = FAILURE_MAX_ACTIONS
        except _ContextLimitFailure:
            failure_reason = FAILURE_CONTEXT_LIMIT

        # Attempt recovery if interruption occurred.
        resume_count = 0
        resume_executor: Optional[BenchmarkStepExecutor] = None
        if interruption_occurred and source_run is not None:
            # Verify the source Run is INTERRUPTED (the HarnessRuntime
            # must have completed the state transition before we resume).
            if source_run.status != RunStatus.INTERRUPTED:
                failure_reason = FAILURE_INTERRUPTED
            elif task.status != TaskStatus.WAITING:
                failure_reason = FAILURE_INTERRUPTED
            else:
                # Load the real checkpoint from the store.
                checkpoint_id = source_run.last_checkpoint_id
                if checkpoint_id is None:
                    # No checkpoint exists — cannot resume. Fail
                    # deterministically; do NOT fake a fresh start.
                    failure_reason = FAILURE_NO_CHECKPOINT
                else:
                    checkpoint = await checkpoint_store.get(checkpoint_id)
                    if checkpoint is None:
                        failure_reason = FAILURE_NO_CHECKPOINT
                    else:
                        # Verify checkpoint lineage (checkpoint → run).
                        # The HarnessRuntime.resume() validates the
                        # run → task association separately.
                        assert checkpoint.run_id == source_run.run_id

                        # Restore the script cursor from the checkpointed
                        # ExecutionState.
                        script_next = checkpoint.agent_state.get(
                            SCRIPT_CURSOR_KEY, 0
                        )

                        # Create a fresh source and seek it to the
                        # restored cursor. Do NOT reuse the old source's
                        # mutable cursor.
                        resume_source = ScriptedActionSource(
                            self._action_factory()
                        )
                        resume_source.seek(script_next)

                        # Create a fresh executor for the resume Run.
                        resume_executor = BenchmarkStepExecutor(
                            source=resume_source,
                            invoker=invoker,
                            ctx=ctx,
                            max_logical_actions=scenario.task.max_logical_actions,
                            initial_script_index=script_next,
                        )
                        self.last_trial_state.resume_executor = resume_executor

                        # Create a NEW HarnessRuntime instance (same
                        # CheckpointStore) to prove recovery does not
                        # depend on the old runtime's local loop state.
                        resume_harness = HarnessRuntime(
                            checkpoint_store=checkpoint_store,
                            step_executor=resume_executor,
                            clock=self._clock,
                            run_id_factory=self._run_id_factory,
                            checkpoint_id_factory=self._checkpoint_id_factory,
                        )

                        # Real resume — this creates a new Run with
                        # current_step = checkpoint.step_index + 1 and
                        # restores the ExecutionState from the checkpoint.
                        try:
                            resume_run = await resume_harness.resume(
                                task=task,
                                source_run=source_run,
                                checkpoint_id=checkpoint_id,
                            )
                            resume_count = 1
                            self.last_trial_state.resume_run = resume_run
                        except BenchmarkStepFailure as exc:
                            failure_reason = _failure_reason_from_error(
                                exc.error_type
                            )
                        except MaxActionsExceededError:
                            failure_reason = FAILURE_MAX_ACTIONS
                        except asyncio.CancelledError:
                            # A second interruption during resume. We do
                            # not recursively resume again in this phase.
                            failure_reason = FAILURE_INTERRUPTED

        # Aggregate stats from source + resume executors.
        executors = [source_executor]
        if resume_executor is not None:
            executors.append(resume_executor)

        total_logical = sum(e.logical_action_count for e in executors)
        total_invocations = sum(e.tool_invocation_count for e in executors)
        total_attempts = sum(e.tool_attempt_count for e in executors)
        total_retries = sum(e.retry_count for e in executors)
        total_checkpoints = sum(e.successful_step_count for e in executors)
        total_loop_detections = sum(
            getattr(e, "loop_detection_count", 0) for e in executors
        )
        total_replans = sum(getattr(e, "replan_count", 0) for e in executors)
        total_externalized = sum(
            getattr(getattr(e, "trial_state", None), "externalized_output_count", 0)
            for e in executors
        )
        total_processed = sum(
            getattr(getattr(e, "trial_state", None), "processed_output_count", 0)
            for e in executors
        )
        peak_context = max(
            (
                getattr(getattr(e, "trial_state", None), "peak_context_tokens", 0)
                for e in executors
            ),
            default=0,
        )
        all_events: list[BenchmarkEvent] = []
        all_history: list[tuple[str, int, str, str]] = []
        for e in executors:
            all_events.extend(e.events)
            all_history.extend(e.execution_history)

        # Oracle determines completion — Harness COMPLETED is NOT oracle
        # completion.
        completed = oracle.is_complete(repo)
        if not completed and failure_reason is None:
            failure_reason = FAILURE_ORACLE_INCOMPLETE

        return BenchmarkRunRecord(
            scenario_id=scenario.scenario_id,
            task_id=scenario.task.task_id,
            completed=completed,
            logical_action_count=total_logical,
            tool_invocation_count=total_invocations,
            tool_attempt_count=total_attempts,
            retry_count=total_retries,
            loop_detection_count=total_loop_detections,
            replan_count=total_replans,
            checkpoint_count=total_checkpoints,
            resume_count=resume_count,
            externalized_output_count=total_externalized,
            processed_output_count=total_processed,
            peak_context_tokens=peak_context,
            failure_reason=failure_reason if not completed else None,
            events=tuple(all_events),
            wall_clock_seconds=0.0,
        )
