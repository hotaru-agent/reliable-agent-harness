"""Tests for the AgentActionController & replan gate (Phase 3 Step 2).

Covers:

* Normal action flow (Section 39)
* Duplicate triggers replan (Section 40)
* Gate blocks fourth tool (Section 41)
* Gate does not consume index (Section 42)
* Acknowledge replan (Section 43)
* Index continues after replan (Section 44)
* New plan not contaminated by old history (Section 45)
* Loop can be detected again later (Section 46)
* Sequence replan (Section 47)
* No-progress replan (Section 48)
* Unknown progress does not gate (Section 49)
* Failed actions participate (Section 50)
* Retry still one logical action (Section 51)
* Cancellation (Section 52)
* Preflight failure counts as action (Section 53)
* ProgressProvider called once (Section 54)
* Replan signal evidence (Section 55)
* acknowledge_replan while READY (Section 56)
* Controller does not modify Task/Run (Section 57 context)

All offline, deterministic, no LLM, no network, no MCP.
"""

from __future__ import annotations

import asyncio

import pytest

from harness.action_controller import (
    ActionControllerState,
    AgentActionController,
    AgentActionResult,
    InvalidActionControllerStateError,
    ReplanRequiredError,
    ReplanSignal,
)
from harness.loop_detection import (
    ActionFingerprint,
    ActionHistory,
    ActionRecord,
    LoopDetectionReason,
    LoopDetector,
    LoopDetectorConfig,
)
from tests.fake_tools import (
    CooperativeCancelTool,
    FakeSleeper,
    PermanentFailureTool,
    ScriptedTool,
    SuccessTool,
)
from tools import (
    RetryPolicy,
    ToolCall,
    ToolErrorType,
    ToolExecutionContext,
    ToolRegistry,
    ToolRuntime,
    ToolSideEffect,
    ToolSpec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeProgressProvider:
    """Scripted progress provider that returns tokens from a list."""

    def __init__(self, tokens: list[str | None]) -> None:
        self.tokens = tokens
        self.call_count = 0

    def get_progress_token(self, call, result) -> str | None:
        idx = min(self.call_count, len(self.tokens) - 1)
        self.call_count += 1
        return self.tokens[idx]


class ConstantProgressProvider:
    """Always returns the same token."""

    def __init__(self, token: str | None) -> None:
        self.token = token
        self.call_count = 0

    def get_progress_token(self, call, result) -> str | None:
        self.call_count += 1
        return self.token


def _spec(
    name: str = "tool",
    *,
    side_effect: ToolSideEffect = ToolSideEffect.READ_ONLY,
    retry_policy: RetryPolicy | None = None,
    timeout_seconds: float = 1.0,
    schema: dict | None = None,
    required_permissions: frozenset[str] = frozenset(),
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="test tool",
        input_schema=schema or {"type": "object"},
        required_permissions=required_permissions,
        timeout_seconds=timeout_seconds,
        side_effect=side_effect,
        retry_policy=retry_policy or RetryPolicy(),
    )


def _make_controller(
    *,
    handler,
    spec: ToolSpec | None = None,
    detector_config: LoopDetectorConfig | None = None,
    progress_provider=None,
    history_size: int = 64,
) -> tuple[AgentActionController, ToolRegistry, object]:
    registry = ToolRegistry()
    spec = spec or _spec()
    registry.register(spec, handler)
    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())
    detector = LoopDetector(
        config=detector_config or LoopDetectorConfig(),
        history=ActionHistory(max_size=history_size),
    )
    pp = progress_provider or ConstantProgressProvider("s1")
    controller = AgentActionController(
        tool_runtime=runtime,
        loop_detector=detector,
        progress_provider=pp,
    )
    return controller, registry, handler


def _run(coro):
    return asyncio.run(coro)


# ===========================================================================
# Section 39 — Normal action flow
# ===========================================================================


class TestNormalAction:
    def test_normal_action_remains_ready(self):
        """A A A with changing progress -> READY, no replan."""
        controller, _, handler = _make_controller(
            handler=SuccessTool("ok"),
            progress_provider=FakeProgressProvider(["p1", "p2", "p3"]),
        )
        call = ToolCall(tool_name="tool", arguments={})
        ctx = ToolExecutionContext()

        for _ in range(3):
            result = _run(controller.execute(call, ctx))
            assert result.replan_signal is None
            assert controller.state is ActionControllerState.READY

        assert handler.call_count == 3
        assert controller.next_action_index == 3

    def test_result_shape(self):
        controller, _, _ = _make_controller(handler=SuccessTool("ok"))
        result = _run(
            controller.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )
        assert isinstance(result, AgentActionResult)
        assert result.tool_result.success is True
        assert isinstance(result.action_record, ActionRecord)
        assert result.loop_result.detected is False
        assert result.replan_signal is None


# ===========================================================================
# Section 40 — Duplicate triggers replan
# ===========================================================================


class TestDuplicateTriggersReplan:
    def test_duplicate_emits_replan_signal(self):
        controller, _, handler = _make_controller(
            handler=SuccessTool("ok"),
            detector_config=LoopDetectorConfig(duplicate_threshold=3),
            progress_provider=ConstantProgressProvider("s1"),
        )
        call = ToolCall(tool_name="tool", arguments={})
        ctx = ToolExecutionContext()

        r0 = _run(controller.execute(call, ctx))
        r1 = _run(controller.execute(call, ctx))
        r2 = _run(controller.execute(call, ctx))

        assert r0.replan_signal is None
        assert r1.replan_signal is None
        assert r2.replan_signal is not None
        assert r2.replan_signal.reason is LoopDetectionReason.DUPLICATE_CALL
        assert controller.state is ActionControllerState.REPLAN_REQUIRED
        assert handler.call_count == 3

    def test_pending_replan_signal_available(self):
        controller, _, _ = _make_controller(
            handler=SuccessTool("ok"),
            detector_config=LoopDetectorConfig(duplicate_threshold=3),
            progress_provider=ConstantProgressProvider("s1"),
        )
        call = ToolCall(tool_name="tool", arguments={})
        ctx = ToolExecutionContext()

        _run(controller.execute(call, ctx))
        _run(controller.execute(call, ctx))
        _run(controller.execute(call, ctx))

        signal = controller.pending_replan_signal
        assert signal is not None
        assert signal.reason is LoopDetectionReason.DUPLICATE_CALL
        assert len(signal.evidence_action_indices) == 3


# ===========================================================================
# Sections 41-42 — Gate blocks tool, does not consume index
# ===========================================================================


class TestReplanGate:
    def test_gate_blocks_tool_execution(self):
        controller, _, handler = _make_controller(
            handler=SuccessTool("ok"),
            detector_config=LoopDetectorConfig(duplicate_threshold=3),
            progress_provider=ConstantProgressProvider("s1"),
        )
        call = ToolCall(tool_name="tool", arguments={})
        ctx = ToolExecutionContext()

        # Trigger replan.
        for _ in range(3):
            _run(controller.execute(call, ctx))
        assert controller.state is ActionControllerState.REPLAN_REQUIRED
        assert handler.call_count == 3

        # Attempt another call -> blocked.
        with pytest.raises(ReplanRequiredError):
            _run(controller.execute(call, ctx))

        # Handler was NOT called again.
        assert handler.call_count == 3

    def test_gate_does_not_consume_action_index(self):
        controller, _, _ = _make_controller(
            handler=SuccessTool("ok"),
            detector_config=LoopDetectorConfig(duplicate_threshold=3),
            progress_provider=ConstantProgressProvider("s1"),
        )
        call = ToolCall(tool_name="tool", arguments={})
        ctx = ToolExecutionContext()

        for _ in range(3):
            _run(controller.execute(call, ctx))
        assert controller.next_action_index == 3

        with pytest.raises(ReplanRequiredError):
            _run(controller.execute(call, ctx))

        # Index unchanged.
        assert controller.next_action_index == 3

    def test_gate_does_not_enter_history(self):
        controller, _, _ = _make_controller(
            handler=SuccessTool("ok"),
            detector_config=LoopDetectorConfig(duplicate_threshold=3),
            progress_provider=ConstantProgressProvider("s1"),
        )
        call = ToolCall(tool_name="tool", arguments={})
        ctx = ToolExecutionContext()

        for _ in range(3):
            _run(controller.execute(call, ctx))
        history_len = len(controller.loop_detector.history)
        assert history_len == 3

        with pytest.raises(ReplanRequiredError):
            _run(controller.execute(call, ctx))

        # History unchanged.
        assert len(controller.loop_detector.history) == 3


# ===========================================================================
# Sections 43-44 — Acknowledge replan
# ===========================================================================


class TestAcknowledgeReplan:
    def test_acknowledge_returns_to_ready(self):
        controller, _, _ = _make_controller(
            handler=SuccessTool("ok"),
            detector_config=LoopDetectorConfig(duplicate_threshold=3),
            progress_provider=ConstantProgressProvider("s1"),
        )
        call = ToolCall(tool_name="tool", arguments={})
        ctx = ToolExecutionContext()

        for _ in range(3):
            _run(controller.execute(call, ctx))
        assert controller.state is ActionControllerState.REPLAN_REQUIRED

        controller.acknowledge_replan()
        assert controller.state is ActionControllerState.READY
        assert controller.pending_replan_signal is None

    def test_acknowledge_clears_history(self):
        controller, _, _ = _make_controller(
            handler=SuccessTool("ok"),
            detector_config=LoopDetectorConfig(duplicate_threshold=3),
            progress_provider=ConstantProgressProvider("s1"),
        )
        call = ToolCall(tool_name="tool", arguments={})
        ctx = ToolExecutionContext()

        for _ in range(3):
            _run(controller.execute(call, ctx))
        assert len(controller.loop_detector.history) == 3

        controller.acknowledge_replan()
        assert len(controller.loop_detector.history) == 0

    def test_index_continues_after_replan(self):
        controller, _, _ = _make_controller(
            handler=SuccessTool("ok"),
            detector_config=LoopDetectorConfig(duplicate_threshold=3),
            progress_provider=ConstantProgressProvider("s1"),
        )
        call = ToolCall(tool_name="tool", arguments={})
        ctx = ToolExecutionContext()

        for _ in range(3):
            _run(controller.execute(call, ctx))
        assert controller.next_action_index == 3

        controller.acknowledge_replan()

        # Next action should be index 3, not 0.
        result = _run(controller.execute(call, ctx))
        assert result.action_record.action_index == 3
        assert controller.next_action_index == 4


# ===========================================================================
# Section 45 — New plan not contaminated by old history
# ===========================================================================


def test_new_plan_not_contaminated():
    """After replan, a single A does not immediately re-trigger."""
    controller, _, _ = _make_controller(
        handler=SuccessTool("ok"),
        detector_config=LoopDetectorConfig(duplicate_threshold=3),
        progress_provider=ConstantProgressProvider("s1"),
    )
    call = ToolCall(tool_name="tool", arguments={})
    ctx = ToolExecutionContext()

    for _ in range(3):
        _run(controller.execute(call, ctx))
    controller.acknowledge_replan()

    # Single A after reset — should not trigger.
    result = _run(controller.execute(call, ctx))
    assert result.replan_signal is None
    assert controller.state is ActionControllerState.READY


# ===========================================================================
# Section 46 — Loop can be detected again later
# ===========================================================================


def test_loop_detected_again_after_replan():
    """After replan + reset, a new loop can still be detected."""
    controller, _, _ = _make_controller(
        handler=SuccessTool("ok"),
        detector_config=LoopDetectorConfig(duplicate_threshold=3),
        progress_provider=ConstantProgressProvider("s2"),
    )
    call = ToolCall(tool_name="tool", arguments={})
    ctx = ToolExecutionContext()

    # First loop.
    for _ in range(3):
        _run(controller.execute(call, ctx))
    assert controller.state is ActionControllerState.REPLAN_REQUIRED
    controller.acknowledge_replan()

    # Second loop with same tool but different progress token.
    r3 = _run(controller.execute(call, ctx))
    r4 = _run(controller.execute(call, ctx))
    r5 = _run(controller.execute(call, ctx))
    assert r3.replan_signal is None
    assert r4.replan_signal is None
    assert r5.replan_signal is not None
    assert r5.replan_signal.reason is LoopDetectionReason.DUPLICATE_CALL


# ===========================================================================
# Section 47 — Sequence replan
# ===========================================================================


def test_sequence_replan():
    """A B A B A B -> REPEATING_SEQUENCE -> ReplanSignal."""
    registry = ToolRegistry()
    registry.register(_spec("A"), SuccessTool("a"))
    registry.register(_spec("B"), SuccessTool("b"))
    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())
    detector = LoopDetector(
        config=LoopDetectorConfig(
            duplicate_threshold=10,
            max_cycle_length=4, cycle_repetitions=3,
            no_progress_window=10,  # disable no-progress for this test
        ),
    )
    controller = AgentActionController(
        tool_runtime=runtime,
        loop_detector=detector,
        progress_provider=ConstantProgressProvider("s1"),
    )
    ctx = ToolExecutionContext()

    call_a = ToolCall(tool_name="A", arguments={})
    call_b = ToolCall(tool_name="B", arguments={})

    result = None
    for rep in range(3):
        result = _run(controller.execute(call_a, ctx))
        result = _run(controller.execute(call_b, ctx))

    assert result.replan_signal is not None
    assert result.replan_signal.reason is LoopDetectionReason.REPEATING_SEQUENCE
    assert controller.state is ActionControllerState.REPLAN_REQUIRED


# ===========================================================================
# Section 48 — No-progress replan
# ===========================================================================


def test_no_progress_replan():
    """5 distinct actions, same token -> NO_PROGRESS -> ReplanSignal."""
    registry = ToolRegistry()
    for name in ["A", "B", "C", "D", "E"]:
        registry.register(_spec(name), SuccessTool(name))
    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())
    detector = LoopDetector(
        config=LoopDetectorConfig(
            no_progress_window=5,
            duplicate_threshold=10,
            max_cycle_length=2, cycle_repetitions=10,
        ),
    )
    controller = AgentActionController(
        tool_runtime=runtime,
        loop_detector=detector,
        progress_provider=ConstantProgressProvider("state-x"),
    )
    ctx = ToolExecutionContext()

    result = None
    for name in ["A", "B", "C", "D", "E"]:
        result = _run(
            controller.execute(ToolCall(tool_name=name, arguments={}), ctx)
        )

    assert result.replan_signal is not None
    assert result.replan_signal.reason is LoopDetectionReason.NO_PROGRESS
    assert controller.state is ActionControllerState.REPLAN_REQUIRED


# ===========================================================================
# Section 49 — Unknown progress does not gate
# ===========================================================================


def test_unknown_progress_does_not_gate():
    """All None progress -> no duplicate/no-progress replan."""
    controller, _, _ = _make_controller(
        handler=SuccessTool("ok"),
        detector_config=LoopDetectorConfig(duplicate_threshold=3),
        progress_provider=ConstantProgressProvider(None),
    )
    call = ToolCall(tool_name="tool", arguments={})
    ctx = ToolExecutionContext()

    for _ in range(5):
        result = _run(controller.execute(call, ctx))
        assert result.replan_signal is None

    assert controller.state is ActionControllerState.READY


# ===========================================================================
# Section 50 — Failed actions participate
# ===========================================================================


def test_failed_actions_participate():
    """PERMANENT x3, same progress -> DUPLICATE_CALL -> ReplanSignal."""
    controller, _, handler = _make_controller(
        handler=PermanentFailureTool(),
        detector_config=LoopDetectorConfig(duplicate_threshold=3),
        progress_provider=ConstantProgressProvider("s1"),
    )
    call = ToolCall(tool_name="tool", arguments={})
    ctx = ToolExecutionContext()

    r0 = _run(controller.execute(call, ctx))
    r1 = _run(controller.execute(call, ctx))
    r2 = _run(controller.execute(call, ctx))

    assert r0.replan_signal is None
    assert r1.replan_signal is None
    assert r2.replan_signal is not None
    assert r2.replan_signal.reason is LoopDetectionReason.DUPLICATE_CALL
    assert r2.action_record.success is False
    assert r2.action_record.error_type is ToolErrorType.PERMANENT
    assert handler.call_count == 3


# ===========================================================================
# Section 51 — Retry still one logical action
# ===========================================================================


def test_retry_still_one_logical_action():
    """3 internal retry attempts -> 1 ActionRecord, 1 action_index."""
    registry = ToolRegistry()
    handler = ScriptedTool([
        ("transient", "fail 1"),
        ("transient", "fail 2"),
        ("success", "done"),
    ])
    registry.register(
        _spec(
            retry_policy=RetryPolicy(
                max_attempts=3, initial_backoff_seconds=0,
            ),
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())
    detector = LoopDetector()
    pp = ConstantProgressProvider("v1")
    controller = AgentActionController(
        tool_runtime=runtime,
        loop_detector=detector,
        progress_provider=pp,
    )

    result = _run(
        controller.execute(
            ToolCall(tool_name="tool", arguments={}),
            ToolExecutionContext(),
        )
    )

    assert result.tool_result.attempt_count == 3
    assert result.action_record.action_index == 0
    assert controller.next_action_index == 1
    assert len(controller.loop_detector.history) == 1
    assert pp.call_count == 1  # ProgressProvider called once


# ===========================================================================
# Section 52 — Cancellation
# ===========================================================================


def test_cancellation_no_action_record():
    """CancelledError propagates; no record, no index, no progress call."""
    registry = ToolRegistry()
    handler = CooperativeCancelTool(delay=5.0)
    registry.register(
        _spec(timeout_seconds=30.0),
        handler,
    )
    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())
    detector = LoopDetector()
    pp = ConstantProgressProvider("v1")
    controller = AgentActionController(
        tool_runtime=runtime,
        loop_detector=detector,
        progress_provider=pp,
    )

    call = ToolCall(tool_name="tool", arguments={})
    ctx = ToolExecutionContext()

    async def _drive():
        task = asyncio.create_task(controller.execute(call, ctx))
        await asyncio.sleep(0.05)
        task.cancel()
        return await task

    with pytest.raises(asyncio.CancelledError):
        _run(_drive())

    assert controller.next_action_index == 0
    assert len(controller.loop_detector.history) == 0
    assert pp.call_count == 0
    assert controller.state is ActionControllerState.READY


# ===========================================================================
# Section 53 — Preflight failure counts as action
# ===========================================================================


def test_preflight_failure_counts_as_action():
    """NOT_FOUND is still a logical action -> ActionRecord in history."""
    registry = ToolRegistry()
    # Register nothing — the tool is missing.
    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())
    detector = LoopDetector()
    pp = ConstantProgressProvider("v1")
    controller = AgentActionController(
        tool_runtime=runtime,
        loop_detector=detector,
        progress_provider=pp,
    )

    result = _run(
        controller.execute(
            ToolCall(tool_name="unknown_tool", arguments={}),
            ToolExecutionContext(),
        )
    )

    assert result.tool_result.success is False
    assert result.tool_result.error.error_type is ToolErrorType.NOT_FOUND
    assert result.tool_result.attempt_count == 0
    assert result.action_record.action_index == 0
    assert result.action_record.success is False
    assert result.action_record.error_type is ToolErrorType.NOT_FOUND
    assert len(controller.loop_detector.history) == 1
    assert controller.next_action_index == 1


# ===========================================================================
# Section 54 — ProgressProvider called once
# ===========================================================================


def test_progress_provider_called_once():
    """Even with 3 internal retries, ProgressProvider is called once."""
    registry = ToolRegistry()
    handler = ScriptedTool([
        ("transient", "fail 1"),
        ("transient", "fail 2"),
        ("success", "done"),
    ])
    registry.register(
        _spec(
            retry_policy=RetryPolicy(
                max_attempts=3, initial_backoff_seconds=0,
            ),
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())
    detector = LoopDetector()
    pp = ConstantProgressProvider("v1")
    controller = AgentActionController(
        tool_runtime=runtime,
        loop_detector=detector,
        progress_provider=pp,
    )

    _run(
        controller.execute(
            ToolCall(tool_name="tool", arguments={}),
            ToolExecutionContext(),
        )
    )

    assert pp.call_count == 1


# ===========================================================================
# Section 55 — Replan signal evidence
# ===========================================================================


def test_replan_signal_evidence():
    """ReplanSignal preserves reason + evidence_action_indices."""
    controller, _, _ = _make_controller(
        handler=SuccessTool("ok"),
        detector_config=LoopDetectorConfig(duplicate_threshold=3),
        progress_provider=ConstantProgressProvider("s1"),
    )
    call = ToolCall(tool_name="tool", arguments={})
    ctx = ToolExecutionContext()

    # Use indices 5, 6, 7 by pre-advancing the controller.
    controller._next_action_index = 5  # type: ignore[attr-defined]
    r5 = _run(controller.execute(call, ctx))
    r6 = _run(controller.execute(call, ctx))
    r7 = _run(controller.execute(call, ctx))

    assert r7.replan_signal is not None
    assert r7.replan_signal.evidence_action_indices == (5, 6, 7)
    assert r7.replan_signal.reason is LoopDetectionReason.DUPLICATE_CALL
    assert isinstance(r7.replan_signal.message, str)
    assert len(r7.replan_signal.message) > 0


# ===========================================================================
# Section 56 — acknowledge_replan while READY
# ===========================================================================


def test_acknowledge_replan_while_ready_rejected():
    """Calling acknowledge_replan in READY state is an error."""
    controller, _, _ = _make_controller(handler=SuccessTool("ok"))
    assert controller.state is ActionControllerState.READY

    with pytest.raises(InvalidActionControllerStateError):
        controller.acknowledge_replan()


# ===========================================================================
# Controller does not modify Task/Run state
# ===========================================================================


def test_controller_does_not_modify_task_run():
    """The controller has no Task/Run reference and cannot modify them."""
    controller, _, _ = _make_controller(handler=SuccessTool("ok"))
    # The controller has no task / run / checkpoint attributes.
    assert not hasattr(controller, "_task")
    assert not hasattr(controller, "_run")
    assert not hasattr(controller, "_checkpoint")

    result = _run(
        controller.execute(
            ToolCall(tool_name="tool", arguments={}),
            ToolExecutionContext(),
        )
    )
    # State is still READY (no loop).
    assert controller.state is ActionControllerState.READY


# ===========================================================================
# ReplanSignal invariants
# ===========================================================================


class TestReplanSignalInvariants:
    def test_empty_message_rejected(self):
        with pytest.raises(ValueError):
            ReplanSignal(
                reason=LoopDetectionReason.DUPLICATE_CALL,
                evidence_action_indices=(0,),
                message="",
            )

    def test_non_str_message_rejected(self):
        with pytest.raises(ValueError):
            ReplanSignal(
                reason=LoopDetectionReason.DUPLICATE_CALL,
                evidence_action_indices=(0,),
                message=123,  # type: ignore[arg-type]
            )

    def test_non_tuple_evidence_rejected(self):
        with pytest.raises(ValueError):
            ReplanSignal(
                reason=LoopDetectionReason.DUPLICATE_CALL,
                evidence_action_indices=[0],  # type: ignore[arg-type]
                message="loop",
            )

    def test_non_reason_rejected(self):
        with pytest.raises(ValueError):
            ReplanSignal(
                reason="duplicate",  # type: ignore[arg-type]
                evidence_action_indices=(0,),
                message="loop",
            )
