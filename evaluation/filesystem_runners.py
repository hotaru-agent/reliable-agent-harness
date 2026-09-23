"""Filesystem benchmark runners (Phase 7 Step 6).

Two runners that execute the same deterministic scripted action
sequence on a REAL temporary filesystem repository with REAL
``sys.executable -m pytest`` subprocess execution:

- ``FilesystemNaiveBenchmarkRunner`` — sequential scripted loop with
  ``max_attempts=1`` (no retry). First tool failure terminates.
  No HarnessRuntime, no checkpoints.

- ``FilesystemReliableHarnessBenchmarkRunner`` — real
  ``HarnessRuntime`` with ``ToolRuntime`` configured for normal retry.
  Reuses the existing ``BenchmarkStepExecutor`` (which only depends on
  ``ScriptedActionSource`` + ``FaultAwareToolInvoker``, NOT on
  ``ControlledRepository``).

Both runners:
- materialize the same immutable ``FilesystemRepositoryFixture`` into
  a FRESH temp directory per trial (no sharing).
- use the same ``FaultAwareToolInvoker`` for non-empty fault plans.
- evaluate completion through ``FilesystemRepositoryOracle`` (real
  pytest subprocess, bypassing fault injection).
- clean up the temp directory after capturing the final snapshot.

All offline, deterministic functional behavior, no LLM, no network.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from evaluation.filesystem_repository import (
    FilesystemRepository,
    FilesystemRepositoryFixture,
    FilesystemRepositoryOracle,
    PytestSubprocessResult,
)
from evaluation.filesystem_tools import (
    make_filesystem_tool_handlers,
    make_filesystem_tool_specs,
)
from evaluation.faults import FaultInjector, FaultPlan, InjectedProcessInterruption
from evaluation.invoker import FaultAwareToolInvoker
from evaluation.models import BenchmarkScenario
from evaluation.records import BenchmarkEvent, BenchmarkEventType, BenchmarkRunRecord
from evaluation.runners import (
    BenchmarkStepExecutor,
    BenchmarkStepFailure,
    MaxActionsExceededError,
    SCRIPT_CURSOR_KEY,
    _DeterministicClock,
    _failure_reason_from_error,
    _make_id_factory,
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
# Failure reasons (consistent with controlled benchmark)
# ---------------------------------------------------------------------------

FAILURE_TRANSIENT = "TRANSIENT"
FAILURE_PERMANENT = "PERMANENT"
FAILURE_TIMEOUT = "TIMEOUT"
FAILURE_EXECUTION = "EXECUTION"
FAILURE_MAX_ACTIONS = "MAX_ACTIONS"
FAILURE_ORACLE_INCOMPLETE = "ORACLE_INCOMPLETE"
FAILURE_INTERRUPTED = "INTERRUPTED"
FAILURE_NO_CHECKPOINT = "NO_CHECKPOINT"


# ---------------------------------------------------------------------------
# Trial state (for deterministic test verification)
# ---------------------------------------------------------------------------


@dataclass
class FilesystemTrialState:
    """Internal state from the last filesystem benchmark trial.

    This is NOT part of the benchmark contract — it exists for
    deterministic test verification (initial/final snapshots, agent/oracle
    pytest counts, temp root identity for isolation tests).
    """

    repo: Optional[FilesystemRepository] = None
    injector: Optional[FaultInjector] = None
    initial_snapshot: Optional[dict[str, str]] = None
    final_snapshot: Optional[dict[str, str]] = None
    oracle_result: Optional[PytestSubprocessResult] = None
    source_executor: Optional[BenchmarkStepExecutor] = None
    resume_executor: Optional[BenchmarkStepExecutor] = None
    task: Optional[Task] = None
    checkpoint_store: Optional[InMemoryCheckpointStore] = None
    source_run: Optional[Run] = None
    resume_run: Optional[Run] = None
    interruption_occurred: bool = False


# ---------------------------------------------------------------------------
# Tool stack builder
# ---------------------------------------------------------------------------


def _build_filesystem_tool_stack(
    repo: FilesystemRepository,
    injector: FaultInjector,
    *,
    run_tests_retry_policy: Optional[RetryPolicy],
    run_tests_timeout_seconds: float,
    sleep: Callable[[float], Any],
) -> tuple[ToolRuntime, FaultAwareToolInvoker, ToolExecutionContext]:
    """Build a ToolRegistry + ToolRuntime + FaultAwareToolInvoker for
    a filesystem repository."""
    registry = ToolRegistry()
    specs = make_filesystem_tool_specs(
        run_tests_retry_policy=run_tests_retry_policy,
        run_tests_timeout_seconds=run_tests_timeout_seconds,
    )
    handlers = make_filesystem_tool_handlers(repo, injector=injector)
    for name in specs:
        registry.register(specs[name], handlers[name])
    runtime = ToolRuntime(registry=registry, sleep=sleep)
    invoker = FaultAwareToolInvoker(runtime, injector=injector)
    ctx = ToolExecutionContext()
    return runtime, invoker, ctx


# ---------------------------------------------------------------------------
# Filesystem naive benchmark runner
# ---------------------------------------------------------------------------


class FilesystemNaiveBenchmarkRunner:
    """Controlled ablation baseline runner for filesystem scenarios.

    Executes a scripted action loop through the real ``ToolRegistry`` /
    ``ToolRuntime`` contract with ``max_attempts=1`` (no automatic
    retry). The first failed tool action terminates the run. No
    HarnessRuntime, no checkpoints, no resume.

    Uses real filesystem tools and real pytest subprocess.
    """

    def __init__(
        self,
        *,
        fixtures: Optional[dict[str, FilesystemRepositoryFixture]] = None,
        fault_plans: Optional[dict[str, FaultPlan]] = None,
        action_factory: Optional[Callable[[], list]] = None,
        run_tests_timeout_seconds: float = 30.0,
        sleep: Optional[Callable[[float], Any]] = None,
        scenario_timeout_overrides: Optional[dict[str, float]] = None,
        scenario_action_factories: Optional[dict[str, Callable[[], list]]] = None,
    ) -> None:
        from evaluation.scenarios.filesystem_scenarios import (
            make_filesystem_standard_fixtures,
            make_filesystem_standard_fault_plans,
            make_filesystem_fix_calculator_actions,
        )
        self._fixtures = (
            fixtures
            if fixtures is not None
            else make_filesystem_standard_fixtures()
        )
        self._fault_plans = (
            fault_plans
            if fault_plans is not None
            else make_filesystem_standard_fault_plans()
        )
        self._action_factory = (
            action_factory or make_filesystem_fix_calculator_actions
        )
        self._run_tests_timeout_seconds = run_tests_timeout_seconds
        self._scenario_timeout_overrides = scenario_timeout_overrides or {}
        self._scenario_action_factories = scenario_action_factories or {}
        self._sleep = sleep if sleep is not None else asyncio.sleep
        self.last_trial_state: FilesystemTrialState = FilesystemTrialState()

    def _get_action_factory(self, scenario_id: str) -> Callable[[], list]:
        return self._scenario_action_factories.get(
            scenario_id, self._action_factory
        )

    async def run(self, scenario: BenchmarkScenario) -> BenchmarkRunRecord:
        self.last_trial_state = FilesystemTrialState()
        fixture = self._fixtures[scenario.fixture_id]
        fault_plan = self._fault_plans.get(
            scenario.fault_plan_id, FaultPlan.no_faults()
        )
        # Per-scenario timeout override (for timeout benchmark).
        run_tests_timeout = self._scenario_timeout_overrides.get(
            scenario.scenario_id, self._run_tests_timeout_seconds
        )

        # Fresh temp repository per trial.
        repo = fixture.materialize()
        injector = FaultInjector(fault_plan)
        oracle = FilesystemRepositoryOracle()

        self.last_trial_state.repo = repo
        self.last_trial_state.injector = injector
        self.last_trial_state.initial_snapshot = repo.snapshot()

        # Naive retry policy: max_attempts=1.
        naive_policy = RetryPolicy(max_attempts=1)
        _, invoker, ctx = _build_filesystem_tool_stack(
            repo,
            injector,
            run_tests_retry_policy=naive_policy,
            run_tests_timeout_seconds=run_tests_timeout,
            sleep=self._sleep,
        )

        from evaluation.scenarios.controlled_repo import ScriptedActionSource

        action_factory = self._get_action_factory(scenario.scenario_id)
        source = ScriptedActionSource(action_factory())
        max_actions = scenario.task.max_logical_actions

        events: list[BenchmarkEvent] = []
        seq = 0
        logical_action_count = 0
        tool_invocation_count = 0
        tool_attempt_count = 0
        retry_count = 0
        failure_reason: Optional[str] = None

        start_time = time.monotonic()

        try:
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
                    tool_attempt_count += 1
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
        finally:
            # Capture final snapshot before cleanup.
            self.last_trial_state.final_snapshot = repo.snapshot()
            # Oracle verification (real pytest, bypasses fault injection).
            oracle_result = await oracle.get_result(repo)
            self.last_trial_state.oracle_result = oracle_result
            completed = oracle_result.passed
            # Cleanup temp directory.
            repo.cleanup()

        elapsed = time.monotonic() - start_time

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
            wall_clock_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Filesystem reliable harness benchmark runner
# ---------------------------------------------------------------------------


class FilesystemReliableHarnessBenchmarkRunner:
    """Reliable runner for filesystem scenarios.

    Uses the real ``HarnessRuntime`` with ``ToolRuntime`` configured for
    normal retry. Reuses the existing ``BenchmarkStepExecutor`` (which
    only depends on ``ScriptedActionSource`` +
    ``FaultAwareToolInvoker``, NOT on ``ControlledRepository``).

    Retry comes ONLY from the existing ``RetryPolicy`` — no
    benchmark-specific retry loops.
    """

    def __init__(
        self,
        *,
        fixtures: Optional[dict[str, FilesystemRepositoryFixture]] = None,
        fault_plans: Optional[dict[str, FaultPlan]] = None,
        action_factory: Optional[Callable[[], list]] = None,
        run_tests_timeout_seconds: float = 30.0,
        sleep: Optional[Callable[[float], Any]] = None,
        clock: Optional[Callable[[], datetime]] = None,
        run_id_factory: Optional[Callable[[], str]] = None,
        checkpoint_id_factory: Optional[Callable[[], str]] = None,
        scenario_timeout_overrides: Optional[dict[str, float]] = None,
        scenario_action_factories: Optional[dict[str, Callable[[], list]]] = None,
    ) -> None:
        from evaluation.scenarios.filesystem_scenarios import (
            make_filesystem_standard_fixtures,
            make_filesystem_standard_fault_plans,
            make_filesystem_fix_calculator_actions,
        )
        self._fixtures = (
            fixtures
            if fixtures is not None
            else make_filesystem_standard_fixtures()
        )
        self._fault_plans = (
            fault_plans
            if fault_plans is not None
            else make_filesystem_standard_fault_plans()
        )
        self._action_factory = (
            action_factory or make_filesystem_fix_calculator_actions
        )
        self._run_tests_timeout_seconds = run_tests_timeout_seconds
        self._scenario_timeout_overrides = scenario_timeout_overrides or {}
        self._scenario_action_factories = scenario_action_factories or {}
        self._sleep = sleep if sleep is not None else asyncio.sleep
        self._clock = clock if clock is not None else _DeterministicClock()
        self._run_id_factory = run_id_factory or _make_id_factory("R")
        self._checkpoint_id_factory = (
            checkpoint_id_factory or _make_id_factory("C")
        )
        self.last_trial_state: FilesystemTrialState = FilesystemTrialState()

    def _get_action_factory(self, scenario_id: str) -> Callable[[], list]:
        return self._scenario_action_factories.get(
            scenario_id, self._action_factory
        )

    async def run(self, scenario: BenchmarkScenario) -> BenchmarkRunRecord:
        self.last_trial_state = FilesystemTrialState()
        fixture = self._fixtures[scenario.fixture_id]
        fault_plan = self._fault_plans.get(
            scenario.fault_plan_id, FaultPlan.no_faults()
        )
        # Per-scenario timeout override (for timeout benchmark).
        run_tests_timeout = self._scenario_timeout_overrides.get(
            scenario.scenario_id, self._run_tests_timeout_seconds
        )

        # Fresh temp repository per trial.
        repo = fixture.materialize()
        injector = FaultInjector(fault_plan)
        oracle = FilesystemRepositoryOracle()

        self.last_trial_state.repo = repo
        self.last_trial_state.injector = injector
        self.last_trial_state.initial_snapshot = repo.snapshot()

        # Normal retry policy (max_attempts=3) — pass None for default.
        runtime, invoker, ctx = _build_filesystem_tool_stack(
            repo,
            injector,
            run_tests_retry_policy=None,
            run_tests_timeout_seconds=run_tests_timeout,
            sleep=self._sleep,
        )

        from evaluation.scenarios.controlled_repo import ScriptedActionSource

        action_factory = self._get_action_factory(scenario.scenario_id)
        checkpoint_store = InMemoryCheckpointStore()
        source = ScriptedActionSource(action_factory())
        source_executor = BenchmarkStepExecutor(
            source=source,
            invoker=invoker,
            ctx=ctx,
            max_logical_actions=scenario.task.max_logical_actions,
        )
        source_harness = HarnessRuntime(
            checkpoint_store=checkpoint_store,
            step_executor=source_executor,
            clock=self._clock,
            run_id_factory=self._run_id_factory,
            checkpoint_id_factory=self._checkpoint_id_factory,
        )

        task = Task(
            task_id=scenario.task.task_id,
            goal=scenario.task.goal,
            created_at=self._clock(),
        )

        self.last_trial_state.task = task
        self.last_trial_state.checkpoint_store = checkpoint_store
        self.last_trial_state.source_executor = source_executor

        start_time = time.monotonic()
        failure_reason: Optional[str] = None
        interruption_occurred = False
        source_run: Optional[Run] = None

        try:
            await source_harness.start(task)
        except asyncio.CancelledError:
            interruption_occurred = True
            source_run = source_executor.interrupted_run
            self.last_trial_state.source_run = source_run
            self.last_trial_state.interruption_occurred = True
        except BenchmarkStepFailure as exc:
            failure_reason = _failure_reason_from_error(exc.error_type)
        except MaxActionsExceededError:
            failure_reason = FAILURE_MAX_ACTIONS

        # Resume logic (Phase 7 Step 8): if interruption occurred and
        # a checkpoint exists, create a NEW HarnessRuntime instance
        # (same CheckpointStore, same filesystem workspace, same
        # FaultInjector) and call the real HarnessRuntime.resume().
        resume_count = 0
        resume_executor: Optional[BenchmarkStepExecutor] = None
        if interruption_occurred and source_run is not None:
            if source_run.status != RunStatus.INTERRUPTED:
                failure_reason = FAILURE_INTERRUPTED
            elif task.status != TaskStatus.WAITING:
                failure_reason = FAILURE_INTERRUPTED
            else:
                checkpoint_id = source_run.last_checkpoint_id
                if checkpoint_id is None:
                    failure_reason = FAILURE_NO_CHECKPOINT
                else:
                    checkpoint = await checkpoint_store.get(checkpoint_id)
                    if checkpoint is None:
                        failure_reason = FAILURE_NO_CHECKPOINT
                    else:
                        assert checkpoint.run_id == source_run.run_id
                        script_next = checkpoint.agent_state.get(
                            SCRIPT_CURSOR_KEY, 0
                        )
                        # Fresh source seeked to restored cursor.
                        resume_source = ScriptedActionSource(
                            action_factory()
                        )
                        resume_source.seek(script_next)
                        resume_executor = BenchmarkStepExecutor(
                            source=resume_source,
                            invoker=invoker,
                            ctx=ctx,
                            max_logical_actions=scenario.task.max_logical_actions,
                            initial_script_index=script_next,
                        )
                        self.last_trial_state.resume_executor = resume_executor
                        # NEW HarnessRuntime (same CheckpointStore).
                        resume_harness = HarnessRuntime(
                            checkpoint_store=checkpoint_store,
                            step_executor=resume_executor,
                            clock=self._clock,
                            run_id_factory=self._run_id_factory,
                            checkpoint_id_factory=self._checkpoint_id_factory,
                        )
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
        all_events: list[BenchmarkEvent] = []
        for e in executors:
            all_events.extend(e.events)

        # Capture final snapshot before cleanup.
        self.last_trial_state.final_snapshot = repo.snapshot()
        # Oracle verification (real pytest, bypasses fault injection).
        oracle_result = await oracle.get_result(repo)
        self.last_trial_state.oracle_result = oracle_result
        completed = oracle_result.passed
        # Cleanup temp directory.
        repo.cleanup()

        elapsed = time.monotonic() - start_time

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
            loop_detection_count=0,
            replan_count=0,
            checkpoint_count=total_checkpoints,
            resume_count=resume_count,
            failure_reason=failure_reason if not completed else None,
            events=tuple(all_events),
            wall_clock_seconds=elapsed,
        )
