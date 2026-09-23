"""Loop detection, replan gate & loop escape benchmark workload
(Phase 7 Step 4).

This module provides benchmark-only, deterministic, non-AI components
for the ``loop_replan`` scenario:

- ``ReplanAwareActionSource`` — a two-phase deterministic action source.
  In ``LOOPING`` phase it produces the same ``read_file("calculator.py")``
  call repeatedly. After ``apply_replan()`` it switches to ``REPAIR``
  phase and produces ``write_file`` + ``run_tests``.

- ``RepositoryProgressProvider`` — implements the production
  ``ProgressProvider`` protocol. Returns a progress token derived from
  the repository's current test failure count (e.g.
  ``"failing-tests:1"``). This reflects real business state, NOT action
  index.

- ``DeterministicReplanStrategy`` — receives a ``ReplanSignal`` and
  produces a deterministic non-AI replan plan. The plan is a marker that
  tells the action source to switch to the repair branch. It does NOT
  look at hidden expected results.

- ``LoopBenchmarkStepExecutor`` — adapts the loop action source to the
  ``StepExecutor`` protocol using the real ``AgentActionController`` +
  ``LoopDetector``. On loop detection it probes the replan gate (tries
  the same call again, expects ``ReplanRequiredError``), applies the
  deterministic replan, and calls ``acknowledge_replan()``.

- ``LoopWorkload`` — a ``Workload`` implementation that creates fresh
  instances of the above for each trial.

All offline, deterministic, no LLM, no network, no real filesystem.
Production runtime code (``HarnessRuntime``, ``ToolRuntime``,
``AgentActionController``, ``LoopDetector``) is NOT modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from evaluation.records import BenchmarkEvent, BenchmarkEventType
from evaluation.repository import ControlledRepository
from harness.action_controller import (
    ActionControllerState,
    AgentActionController,
    AgentActionResult,
    ReplanRequiredError,
    ReplanSignal,
)
from harness.execution import ExecutionState, StepOutcome, StepResult
from harness.loop_detection import (
    ActionHistory,
    LoopDetector,
    LoopDetectorConfig,
)
from tools import ToolCall, ToolExecutionContext, ToolRuntime
from tools.models import ToolErrorType

# The namespace key used to store the scripted action source cursor in
# ``ExecutionState.agent_state`` so it survives checkpoint / resume.
# This mirrors the constant in ``evaluation.runners``; defined locally
# to avoid a circular import (runners → workloads → loop_workload →
# runners).
SCRIPT_CURSOR_KEY = "benchmark.script_next_index"


# ===========================================================================
# Replan plan (deterministic, non-AI)
# ===========================================================================


@dataclass(frozen=True)
class ReplanPlan:
    """Deterministic non-AI replan plan produced by
    ``DeterministicReplanStrategy``.

    This is a marker that tells ``ReplanAwareActionSource`` to switch
    from the ``LOOPING`` phase to the ``REPAIR`` phase. It does NOT
    contain the fix content — the repair branch is part of the action
    source's blueprint, not the strategy's output.

    The strategy does NOT look at hidden expected results or scenario
    IDs. It simply receives a ``ReplanSignal`` and returns a plan that
    says "switch to repair".
    """

    plan_id: str = "deterministic_repair"


# ===========================================================================
# Deterministic replan strategy
# ===========================================================================


class DeterministicReplanStrategy:
    """Deterministic, non-AI replan strategy.

    Receives a ``ReplanSignal`` and produces a ``ReplanPlan``. The plan
    is always the same: "switch to repair". This is NOT an LLM, NOT a
    model API call, and NOT a prompt-based planner.

    The strategy does NOT:
    - call an LLM;
    - look at hidden expected results;
    - branch on scenario_id;
    - modify the repository directly;
    - produce ToolCalls itself.

    It only signals the action source to switch phases. The actual
    repair actions (``write_file``, ``run_tests``) come from the action
    source's repair branch, executed through the real ``ToolRuntime``.
    """

    def replan(self, signal: ReplanSignal) -> ReplanPlan:
        """Return a deterministic replan plan for the given signal."""
        if not isinstance(signal, ReplanSignal):
            raise ValueError("signal must be a ReplanSignal")
        return ReplanPlan()


# ===========================================================================
# Replan-aware action source
# ===========================================================================


class ReplanAwareActionSource:
    """Deterministic, non-AI action source with two phases.

    Phase ``LOOPING``: produces the same ``read_file("calculator.py")``
    call repeatedly. This models an agent stuck in a no-progress loop.

    Phase ``REPAIR``: produces a fixed sequence of repair actions
    (``write_file`` + ``run_tests``). This models the post-replan plan.

    Phase transition happens ONLY when ``apply_replan(plan)`` is called.
    The Naive runner never calls ``apply_replan``, so it stays in
    ``LOOPING`` forever (until ``max_logical_actions``).

    This is NOT an AI agent. It is benchmark control logic.
    """

    def __init__(
        self,
        *,
        looping_call: ToolCall,
        repair_calls: list[ToolCall],
    ) -> None:
        if not isinstance(looping_call, ToolCall):
            raise ValueError("looping_call must be a ToolCall")
        if not isinstance(repair_calls, list):
            raise ValueError("repair_calls must be a list")
        self._looping_call = looping_call
        self._repair_calls: tuple[ToolCall, ...] = tuple(repair_calls)
        self._phase: str = "LOOPING"
        self._repair_index: int = 0

    @property
    def phase(self) -> str:
        """Current phase: ``'LOOPING'`` or ``'REPAIR'``."""
        return self._phase

    @property
    def current_index(self) -> int:
        """Cursor position within the current phase.

        In ``LOOPING`` phase this is the number of looping calls
        produced so far. In ``REPAIR`` phase this is the repair action
        index.
        """
        if self._phase == "LOOPING":
            return self._repair_index  # 0 until apply_replan
        return self._repair_index

    def has_next(self) -> bool:
        """Return True if there are more actions to produce.

        In ``LOOPING`` phase this is always True (infinite loop).
        In ``REPAIR`` phase this is True until repair actions are
        exhausted.
        """
        if self._phase == "LOOPING":
            return True
        return self._repair_index < len(self._repair_calls)

    def next_call(self) -> ToolCall:
        """Return the next ``ToolCall`` and advance the internal pointer.

        In ``LOOPING`` phase: returns the same looping call each time.
        In ``REPAIR`` phase: returns repair calls in order.

        Raises ``StopIteration`` if exhausted (only in ``REPAIR`` phase).
        """
        if self._phase == "LOOPING":
            # Produce the same looping call. We don't advance a counter
            # here because the looping call is identical each time.
            return ToolCall(
                tool_name=self._looping_call.tool_name,
                arguments=dict(self._looping_call.arguments),
            )
        if self._repair_index >= len(self._repair_calls):
            raise StopIteration("repair action source exhausted")
        call = self._repair_calls[self._repair_index]
        self._repair_index += 1
        return ToolCall(
            tool_name=call.tool_name,
            arguments=dict(call.arguments),
        )

    def apply_replan(self, plan: ReplanPlan) -> None:
        """Switch from ``LOOPING`` to ``REPAIR`` phase.

        After this call, ``next_call()`` produces repair actions.
        The repair cursor starts at 0.
        """
        if not isinstance(plan, ReplanPlan):
            raise ValueError("plan must be a ReplanPlan")
        if self._phase != "LOOPING":
            raise RuntimeError(
                f"apply_replan() can only be called in LOOPING phase; "
                f"current phase is {self._phase}"
            )
        self._phase = "REPAIR"
        self._repair_index = 0


# ===========================================================================
# Repository progress provider
# ===========================================================================


class RepositoryProgressProvider:
    """Progress provider that derives progress tokens from the
    repository's current test failure count.

    Implements the production ``ProgressProvider`` protocol. Called
    exactly once per logical action (after ``ToolRuntime.execute()``
    returns a ``ToolExecutionResult``).

    The token format is ``"failing-tests:N"`` where N is the number of
    failing tests. Before the fix: ``"failing-tests:1"``. After
    ``write_file`` with corrected content: ``"failing-tests:0"``.

    This reflects real business state, NOT action index. Repeated
    ``read_file`` calls on an unchanged repository produce the same
    token — this is what makes the loop detectable as no-progress.
    """

    def __init__(self, repo: ControlledRepository) -> None:
        self._repo = repo
        self.call_count: int = 0

    def get_progress_token(
        self,
        call: ToolCall,
        result: Any,
    ) -> Optional[str]:
        """Return ``"failing-tests:N"`` based on current repo state."""
        self.call_count += 1
        test_result = self._repo.run_tests()
        return f"failing-tests:{test_result.failed_count}"


# ===========================================================================
# Loop benchmark step executor
# ===========================================================================


class LoopBenchmarkStepExecutor:
    """Adapts ``ReplanAwareActionSource`` to the ``StepExecutor``
    protocol using the real ``AgentActionController`` + ``LoopDetector``.

    One executed controller action = one Harness logical step.

    On loop detection (third duplicate action):
    1. Records ``LOOP_DETECTED`` event.
    2. Probes the replan gate: tries the same call again, expects
       ``ReplanRequiredError``. The blocked call does NOT consume a
       source action, does NOT increment action_index, does NOT enter
       ActionHistory, and does NOT count as a logical action.
    3. Records ``REPLAN_BLOCKED`` event.
    4. Applies the deterministic replan: ``strategy.replan(signal)`` →
       ``source.apply_replan(plan)``.
    5. Calls ``controller.acknowledge_replan()`` → controller READY,
       detector episode reset.
    6. Records ``REPLAN_ACKNOWLEDGED`` event.

    On tool failure: raises ``BenchmarkStepFailure`` (same as
    ``BenchmarkStepExecutor``).

    On ``max_logical_actions`` exceeded: raises
    ``MaxActionsExceededError``.

    Accumulates execution stats for the runner to read after the run.
    """

    def __init__(
        self,
        *,
        source: ReplanAwareActionSource,
        controller: AgentActionController,
        ctx: ToolExecutionContext,
        strategy: DeterministicReplanStrategy,
        max_logical_actions: int,
    ) -> None:
        self._source = source
        self._controller = controller
        self._ctx = ctx
        self._strategy = strategy
        self._max_logical_actions = max_logical_actions
        # Accumulated stats (read by the runner after the run).
        self.logical_action_count: int = 0
        self.tool_invocation_count: int = 0
        self.tool_attempt_count: int = 0
        self.retry_count: int = 0
        self.successful_step_count: int = 0
        self.loop_detection_count: int = 0
        self.replan_count: int = 0
        self.blocked_call_count: int = 0
        self.events: list[BenchmarkEvent] = []
        self._seq: int = 0
        # Execution history: (run_id, step_index, tool_name, outcome)
        self.execution_history: list[tuple[str, int, str, str]] = []
        # The last replan signal captured (for test verification).
        self.last_replan_signal: Optional[ReplanSignal] = None

    async def execute_step(self, task, run, state: ExecutionState) -> StepResult:
        # Lazy imports to avoid circular import (runners → workloads →
        # loop_workload → runners).
        from evaluation.runners import (
            BenchmarkStepFailure,
            MaxActionsExceededError,
        )

        # Script exhausted → COMPLETE.
        if not self._source.has_next():
            self.successful_step_count += 1
            return StepResult(StepOutcome.COMPLETE, state)

        # Max actions exceeded.
        if self.logical_action_count >= self._max_logical_actions:
            raise MaxActionsExceededError(
                f"max_logical_actions ({self._max_logical_actions}) exceeded"
            )

        call = self._source.next_call()
        self.logical_action_count += 1
        self.tool_invocation_count += 1

        # Execute through the real AgentActionController.
        result: AgentActionResult = await self._controller.execute(
            call, self._ctx
        )

        self.tool_attempt_count += result.tool_result.attempt_count
        self.retry_count += max(0, result.tool_result.attempt_count - 1)

        # Record logical action event.
        self.events.append(BenchmarkEvent(
            event_type=BenchmarkEventType.LOGICAL_ACTION,
            sequence=self._seq,
            tool_name=call.tool_name,
            success=result.tool_result.success,
            details={
                "action_index": result.action_record.action_index,
                "attempt_count": result.tool_result.attempt_count,
            },
        ))
        self._seq += 1

        if not result.tool_result.success:
            self.execution_history.append(
                (run.run_id, run.current_step, call.tool_name, "failed")
            )
            raise BenchmarkStepFailure(
                result.tool_result.error.error_type, call.tool_name
            )

        self.execution_history.append(
            (run.run_id, run.current_step, call.tool_name, "success")
        )

        # Check for loop detection.
        if result.replan_signal is not None:
            self.loop_detection_count += 1
            self.last_replan_signal = result.replan_signal

            self.events.append(BenchmarkEvent(
                event_type=BenchmarkEventType.LOOP_DETECTED,
                sequence=self._seq,
                tool_name=call.tool_name,
                success=False,
                details={
                    "reason": result.replan_signal.reason.value,
                    "evidence_action_indices": list(
                        result.replan_signal.evidence_action_indices
                    ),
                    "action_index": result.action_record.action_index,
                },
            ))
            self._seq += 1

            # Verify controller is in REPLAN_REQUIRED.
            assert self._controller.state is ActionControllerState.REPLAN_REQUIRED

            # Gate probe: try the same call again — must be blocked.
            try:
                await self._controller.execute(call, self._ctx)
                # If we get here, the gate did NOT block — that's a bug.
                raise RuntimeError(
                    "replan gate did not block after loop detection"
                )
            except ReplanRequiredError:
                pass  # Expected — gate blocked the call.

            self.blocked_call_count += 1
            self.events.append(BenchmarkEvent(
                event_type=BenchmarkEventType.REPLAN_BLOCKED,
                sequence=self._seq,
                tool_name=call.tool_name,
                success=False,
                details={},
            ))
            self._seq += 1

            # Deterministic replan: strategy produces plan, source
            # switches to REPAIR phase.
            new_plan = self._strategy.replan(result.replan_signal)
            self._source.apply_replan(new_plan)

            # Acknowledge replan — controller returns to READY,
            # detector episode reset.
            self._controller.acknowledge_replan()
            self.replan_count += 1

            self.events.append(BenchmarkEvent(
                event_type=BenchmarkEventType.REPLAN_ACKNOWLEDGED,
                sequence=self._seq,
                tool_name=call.tool_name,
                success=True,
                details={},
            ))
            self._seq += 1

            # Verify controller is back to READY.
            assert self._controller.state is ActionControllerState.READY
            assert self._controller.pending_replan_signal is None

        # Update execution state and return CONTINUE.
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
        return StepResult(StepOutcome.CONTINUE, new_state)


# ===========================================================================
# Loop workload blueprint
# ===========================================================================


def make_loop_repair_calls(corrected_content: str) -> list[ToolCall]:
    """Return the repair branch ToolCalls for the loop scenario.

    After replan:
    1. write_file calculator.py with corrected content.
    2. run_tests to verify.
    """
    return [
        ToolCall(
            tool_name="write_file",
            arguments={
                "path": "calculator.py",
                "content": corrected_content,
            },
        ),
        ToolCall(
            tool_name="run_tests",
            arguments={},
        ),
    ]


def make_loop_looping_call() -> ToolCall:
    """Return the looping ToolCall for the loop scenario."""
    return ToolCall(
        tool_name="read_file",
        arguments={"path": "calculator.py"},
    )
