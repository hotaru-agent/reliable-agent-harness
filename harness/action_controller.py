"""Agent action controller & structured replan gate (Phase 3 Step 2).

The ``AgentActionController`` is the governance boundary for logical
Agent tool actions. It sits above ``ToolRuntime`` (which handles tool
execution + retry) and uses ``LoopDetector`` (which handles detection)
to enforce a replan gate:

    Agent logical ToolCall
        ↓
    AgentActionController.execute()
        ↓
    [gate: if REPLAN_REQUIRED → ReplanRequiredError]
        ↓
    ToolRuntime.execute()
        ↓
    ToolExecutionResult
        ↓
    ProgressProvider.get_progress_token()
        ↓
    ActionRecord
        ↓
    LoopDetector.observe()
        ↓
    no loop → CONTINUE (return AgentActionResult, stay READY)
    loop    → ReplanSignal → controller enters REPLAN_REQUIRED
        ↓
    further ToolCalls blocked until acknowledge_replan()

The controller does NOT:

* call an LLM;
* modify Task / Run / Checkpoint state;
* generate a new plan;
* implement abort policy;
* integrate with ``HarnessRuntime._run_loop``.

It only governs logical action flow: execute, observe, detect, gate.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol

from opentelemetry.trace import Tracer
from opentelemetry.trace.status import Status, StatusCode

from harness.errors import HarnessRuntimeError
from harness.loop_detection import (
    ActionRecord,
    LoopDetectionReason,
    LoopDetectionResult,
    LoopDetector,
)
from observability.tracing import (
    ATTR_AGENT_ACTION_INDEX,
    ATTR_AGENT_ACTION_LOOP_DETECTED,
    ATTR_AGENT_ACTION_NEXT_INDEX,
    ATTR_AGENT_ACTION_SUCCESS,
    ATTR_AGENT_CONTROLLER_STATE,
    ATTR_AGENT_LOOP_EVIDENCE_COUNT,
    ATTR_AGENT_LOOP_REASON,
    ATTR_AGENT_REPLAN_EVIDENCE_COUNT,
    ATTR_TOOL_NAME,
    EVENT_AGENT_LOOP_DETECTED,
    EVENT_AGENT_REPLAN_BLOCKED,
    EVENT_AGENT_REPLAN_REQUIRED,
    SPAN_AGENT_ACTION,
    SPAN_AGENT_REPLAN_ACKNOWLEDGE,
    resolve_tracer,
    safe_add_event,
    safe_record_exception,
    safe_set_attribute,
    safe_set_status,
    safe_span,
)
from tools.models import ToolCall, ToolExecutionContext, ToolExecutionResult
from tools.runtime import ToolRuntime


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ReplanRequiredError(HarnessRuntimeError):
    """Raised when a ToolCall is attempted while the controller is in
    ``REPLAN_REQUIRED`` state.

    The caller must ``acknowledge_replan()`` before submitting more
    ToolCalls.
    """


class InvalidActionControllerStateError(HarnessRuntimeError):
    """Raised when an operation is invoked in the wrong controller state.

    For example, calling ``acknowledge_replan()`` while ``READY``.
    """


# ---------------------------------------------------------------------------
# Progress provider
# ---------------------------------------------------------------------------


class ProgressProvider(Protocol):
    """Provides a progress token after a tool action completes.

    Called exactly once per logical action (after ``ToolRuntime.execute``
    returns a ``ToolExecutionResult``), NOT per internal retry attempt.

    Returns ``str`` for explicit progress evidence, or ``None`` for
    unknown / not-provided progress.
    """

    def get_progress_token(
        self,
        call: ToolCall,
        result: ToolExecutionResult,
    ) -> Optional[str]:
        ...


# ---------------------------------------------------------------------------
# Controller state
# ---------------------------------------------------------------------------


class ActionControllerState(Enum):
    """State of the ``AgentActionController``."""

    READY = "ready"
    REPLAN_REQUIRED = "replan_required"


# ---------------------------------------------------------------------------
# Replan signal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplanSignal:
    """Structured replan signal emitted when a loop is detected.

    This is data, not a prompt. A future Agent adapter can render it
    into a prompt; the controller itself never calls an LLM.
    """

    reason: LoopDetectionReason
    evidence_action_indices: tuple[int, ...]
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, LoopDetectionReason):
            raise ValueError("reason must be a LoopDetectionReason")
        if not isinstance(self.evidence_action_indices, tuple):
            raise ValueError("evidence_action_indices must be a tuple")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("message must be a non-empty str")


# ---------------------------------------------------------------------------
# Agent action result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentActionResult:
    """Unified result of one logical Agent action through the controller.

    * ``tool_result`` — the raw ``ToolExecutionResult`` from
      ``ToolRuntime.execute()``.
    * ``action_record`` — the ``ActionRecord`` appended to history.
    * ``loop_result`` — the ``LoopDetectionResult`` from the detector.
    * ``replan_signal`` — non-None iff a loop was detected and the
      controller transitioned to ``REPLAN_REQUIRED``.
    """

    tool_result: ToolExecutionResult
    action_record: ActionRecord
    loop_result: LoopDetectionResult
    replan_signal: Optional[ReplanSignal] = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool_result, ToolExecutionResult):
            raise ValueError("tool_result must be a ToolExecutionResult")
        if not isinstance(self.action_record, ActionRecord):
            raise ValueError("action_record must be an ActionRecord")
        if not isinstance(self.loop_result, LoopDetectionResult):
            raise ValueError("loop_result must be a LoopDetectionResult")
        if self.replan_signal is not None and not isinstance(
            self.replan_signal, ReplanSignal
        ):
            raise ValueError("replan_signal must be a ReplanSignal or None")


# ---------------------------------------------------------------------------
# Agent action controller
# ---------------------------------------------------------------------------


class AgentActionController:
    """Governance boundary for logical Agent tool actions.

    Combines ``ToolRuntime`` (execution + retry), ``LoopDetector``
    (detection), and ``ProgressProvider`` (progress evidence) into a
    single logical-action pipeline with a replan gate.

    The controller owns ``action_index`` (the logical Agent action
    sequence number). Callers do NOT specify it.

    Dependencies are injected; the controller never creates its own
    ``ToolRuntime`` or registry.
    """

    def __init__(
        self,
        *,
        tool_runtime: ToolRuntime,
        loop_detector: LoopDetector,
        progress_provider: ProgressProvider,
        tracer: Optional[Tracer] = None,
    ) -> None:
        self._tool_runtime = tool_runtime
        self._loop_detector = loop_detector
        self._progress_provider = progress_provider
        self._state: ActionControllerState = ActionControllerState.READY
        self._next_action_index: int = 0
        self._pending_replan_signal: Optional[ReplanSignal] = None
        self._tracer = resolve_tracer(tracer)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> ActionControllerState:
        return self._state

    @property
    def next_action_index(self) -> int:
        return self._next_action_index

    @property
    def pending_replan_signal(self) -> Optional[ReplanSignal]:
        return self._pending_replan_signal

    @property
    def loop_detector(self) -> LoopDetector:
        return self._loop_detector

    # ------------------------------------------------------------------
    # Execute a logical action
    # ------------------------------------------------------------------

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> AgentActionResult:
        """Execute one logical Agent tool action through the governance gate.

        If the controller is in ``REPLAN_REQUIRED``, raises
        ``ReplanRequiredError`` without touching ``ToolRuntime`` or
        incrementing ``action_index``.

        If ``ToolRuntime.execute()`` raises ``asyncio.CancelledError``
        (external cancellation), the controller re-raises it without
        producing an ``ActionRecord`` or incrementing ``action_index``.
        """
        with safe_span(self._tracer, SPAN_AGENT_ACTION) as action_span:
            safe_set_attribute(
                action_span, ATTR_AGENT_CONTROLLER_STATE, self._state.value
            )
            safe_set_attribute(
                action_span, ATTR_AGENT_ACTION_NEXT_INDEX, self._next_action_index
            )
            safe_set_attribute(action_span, ATTR_TOOL_NAME, call.tool_name)

            if self._state is ActionControllerState.REPLAN_REQUIRED:
                # Replan gate block: do NOT execute the tool, do NOT
                # increment action_index. Record the block event.
                safe_add_event(action_span, EVENT_AGENT_REPLAN_BLOCKED)
                safe_set_status(action_span, Status(StatusCode.ERROR))
                raise ReplanRequiredError(
                    "controller is in REPLAN_REQUIRED state; call "
                    "acknowledge_replan() before submitting more ToolCalls"
                )

            # Execute the tool. CancelledError propagates out — no record,
            # no index increment, no progress provider call.
            try:
                tool_result = await self._tool_runtime.execute(call, context)
            except asyncio.CancelledError:
                safe_set_status(action_span, Status(StatusCode.UNSET))
                raise

            # Progress evidence is computed AFTER execution, once.
            progress_token = self._progress_provider.get_progress_token(
                call, tool_result
            )

            # Build the action record. The controller owns action_index.
            action_record = ActionRecord.from_tool_result(
                action_index=self._next_action_index,
                call=call,
                result=tool_result,
                progress_token=progress_token,
            )

            # Observe through the loop detector.
            loop_result = self._loop_detector.observe(action_record)

            # Advance the action index — this logical action is committed.
            self._next_action_index += 1

            # Record action outcome on the span.
            safe_set_attribute(
                action_span, ATTR_AGENT_ACTION_INDEX, action_record.action_index
            )
            safe_set_attribute(
                action_span, ATTR_AGENT_ACTION_SUCCESS, tool_result.success
            )

            # If a loop was detected, emit a replan signal and enter the
            # replan gate.
            replan_signal: Optional[ReplanSignal] = None
            if loop_result.detected:
                replan_signal = ReplanSignal(
                    reason=loop_result.reason,  # type: ignore[arg-type]
                    evidence_action_indices=loop_result.evidence_action_indices,
                    message=loop_result.message or "",
                )
                self._state = ActionControllerState.REPLAN_REQUIRED
                self._pending_replan_signal = replan_signal

                # Record loop detection event.
                safe_set_attribute(
                    action_span, ATTR_AGENT_ACTION_LOOP_DETECTED, True
                )
                safe_add_event(
                    action_span,
                    EVENT_AGENT_LOOP_DETECTED,
                    {
                        ATTR_AGENT_LOOP_REASON: loop_result.reason.value,  # type: ignore[union-attr]
                        ATTR_AGENT_LOOP_EVIDENCE_COUNT: len(
                            loop_result.evidence_action_indices
                        ),
                    },
                )
                safe_add_event(
                    action_span,
                    EVENT_AGENT_REPLAN_REQUIRED,
                    {
                        ATTR_AGENT_LOOP_REASON: loop_result.reason.value,  # type: ignore[union-attr]
                        ATTR_AGENT_REPLAN_EVIDENCE_COUNT: len(
                            loop_result.evidence_action_indices
                        ),
                    },
                )
                safe_set_status(action_span, Status(StatusCode.ERROR))
            else:
                safe_set_attribute(
                    action_span, ATTR_AGENT_ACTION_LOOP_DETECTED, False
                )
                if tool_result.success:
                    safe_set_status(action_span, Status(StatusCode.OK))
                else:
                    safe_set_status(action_span, Status(StatusCode.ERROR))

            return AgentActionResult(
                tool_result=tool_result,
                action_record=action_record,
                loop_result=loop_result,
                replan_signal=replan_signal,
            )

    # ------------------------------------------------------------------
    # Replan acknowledgement
    # ------------------------------------------------------------------

    def acknowledge_replan(self) -> None:
        """Acknowledge that the higher layer has processed the replan signal.

        Resets the loop detector history (so the new plan is not
        contaminated by old loop evidence) and returns the controller
        to ``READY``. ``action_index`` is NOT reset — it continues
        from where it left off, preserving the Agent action timeline.

        Raises ``InvalidActionControllerStateError`` if the controller
        is not in ``REPLAN_REQUIRED``.
        """
        with safe_span(self._tracer, SPAN_AGENT_REPLAN_ACKNOWLEDGE) as ack_span:
            safe_set_attribute(
                ack_span, ATTR_AGENT_ACTION_NEXT_INDEX, self._next_action_index
            )
            if self._state is not ActionControllerState.REPLAN_REQUIRED:
                safe_record_exception(ack_span, InvalidActionControllerStateError(""))
                safe_set_status(ack_span, Status(StatusCode.ERROR))
                raise InvalidActionControllerStateError(
                    "acknowledge_replan() can only be called in "
                    "REPLAN_REQUIRED state; current state is "
                    f"{self._state.value}"
                )
            self._loop_detector.reset()
            self._state = ActionControllerState.READY
            self._pending_replan_signal = None
            safe_set_status(ack_span, Status(StatusCode.OK))
