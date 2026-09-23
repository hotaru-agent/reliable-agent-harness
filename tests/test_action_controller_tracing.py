"""Tests for AgentActionController OpenTelemetry tracing (Phase 6 Step 2).

All offline, deterministic, no network, no collector. Uses a recording
span exporter to assert on span names, attributes, events, statuses
and parent-child relationships.

Tests prove:
* agent.action span exists with action index, controller state, tool name
* agent.action → tool.execute → tool.attempt parenting via OTel context
* success: agent.action OK, success=True, loop_detected=False
* failed tool: agent.action ERROR, success=False
* loop detected event + replan required event
* blocked action: agent.replan.blocked event, ERROR status, no tool.execute
* acknowledge_replan span
* no fingerprint/args in telemetry
* tracer injection isolation
* telemetry failure isolation
"""

from __future__ import annotations

import asyncio

import pytest
from opentelemetry.trace.status import Status, StatusCode

from harness.action_controller import (
    ActionControllerState,
    AgentActionController,
    InvalidActionControllerStateError,
    ReplanRequiredError,
)
from harness.loop_detection import (
    ActionHistory,
    LoopDetector,
    LoopDetectorConfig,
)
from observability import (
    ATTR_AGENT_ACTION_INDEX,
    ATTR_AGENT_ACTION_LOOP_DETECTED,
    ATTR_AGENT_ACTION_NEXT_INDEX,
    ATTR_AGENT_ACTION_SUCCESS,
    ATTR_AGENT_CONTROLLER_STATE,
    ATTR_AGENT_LOOP_EVIDENCE_COUNT,
    ATTR_AGENT_LOOP_REASON,
    ATTR_TOOL_NAME,
    EVENT_AGENT_LOOP_DETECTED,
    EVENT_AGENT_REPLAN_BLOCKED,
    EVENT_AGENT_REPLAN_REQUIRED,
    SPAN_AGENT_ACTION,
    SPAN_AGENT_REPLAN_ACKNOWLEDGE,
    SPAN_TOOL_ATTEMPT,
    SPAN_TOOL_EXECUTE,
)
from tests.fake_tools import FakeSleeper, ScriptedTool, SuccessTool
from tests.fakes_telemetry import make_recording_tracer
from tools import (
    RetryPolicy,
    ToolCall,
    ToolExecutionContext,
    ToolRegistry,
    ToolRuntime,
    ToolSideEffect,
    ToolSpec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ConstantProgressProvider:
    def __init__(self, token: str | None) -> None:
        self.token = token
        self.call_count = 0

    def get_progress_token(self, call, result) -> str | None:
        self.call_count += 1
        return self.token


class FakeProgressProvider:
    def __init__(self, tokens: list[str | None]) -> None:
        self.tokens = tokens
        self.call_count = 0

    def get_progress_token(self, call, result) -> str | None:
        idx = min(self.call_count, len(self.tokens) - 1)
        self.call_count += 1
        return self.tokens[idx]


def _spec(
    name: str = "tool",
    *,
    side_effect: ToolSideEffect = ToolSideEffect.READ_ONLY,
    retry_policy: RetryPolicy | None = None,
    timeout_seconds: float = 1.0,
    schema: dict | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="test tool",
        input_schema=schema or {"type": "object"},
        required_permissions=frozenset(),
        timeout_seconds=timeout_seconds,
        side_effect=side_effect,
        retry_policy=retry_policy or RetryPolicy(),
    )


def _make_traced_controller(
    tracer,
    *,
    handler,
    spec: ToolSpec | None = None,
    detector_config: LoopDetectorConfig | None = None,
    progress_provider=None,
    sleep=None,
) -> tuple[AgentActionController, ToolRegistry, object]:
    registry = ToolRegistry()
    spec = spec or _spec()
    registry.register(spec, handler)
    runtime = ToolRuntime(
        registry=registry, tracer=tracer, sleep=sleep or FakeSleeper()
    )
    detector = LoopDetector(
        config=detector_config or LoopDetectorConfig(),
        history=ActionHistory(max_size=64),
    )
    pp = progress_provider or ConstantProgressProvider("s1")
    controller = AgentActionController(
        tool_runtime=runtime,
        loop_detector=detector,
        progress_provider=pp,
        tracer=tracer,
    )
    return controller, registry, handler


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Section 17 — Agent Action Normal Success
# ---------------------------------------------------------------------------


class TestAgentActionSuccess:
    def test_action_span_exists(self):
        tracer, exporter = make_recording_tracer()
        controller, _, _ = _make_traced_controller(
            tracer, handler=SuccessTool("ok")
        )

        _run(
            controller.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )

        action_spans = exporter.spans_named(SPAN_AGENT_ACTION)
        assert len(action_spans) == 1

    def test_action_span_has_tool_name(self):
        tracer, exporter = make_recording_tracer()
        controller, _, _ = _make_traced_controller(
            tracer, handler=SuccessTool("ok")
        )

        _run(
            controller.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )

        span = exporter.span_by_name(SPAN_AGENT_ACTION)
        assert span.attributes[ATTR_TOOL_NAME] == "tool"

    def test_action_span_has_controller_state(self):
        tracer, exporter = make_recording_tracer()
        controller, _, _ = _make_traced_controller(
            tracer, handler=SuccessTool("ok")
        )

        _run(
            controller.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )

        span = exporter.span_by_name(SPAN_AGENT_ACTION)
        assert span.attributes[ATTR_AGENT_CONTROLLER_STATE] == "ready"

    def test_action_span_has_index(self):
        tracer, exporter = make_recording_tracer()
        controller, _, _ = _make_traced_controller(
            tracer, handler=SuccessTool("ok")
        )

        _run(
            controller.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )

        span = exporter.span_by_name(SPAN_AGENT_ACTION)
        assert span.attributes[ATTR_AGENT_ACTION_INDEX] == 0

    def test_action_success_attributes(self):
        tracer, exporter = make_recording_tracer()
        controller, _, _ = _make_traced_controller(
            tracer, handler=SuccessTool("ok")
        )

        _run(
            controller.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )

        span = exporter.span_by_name(SPAN_AGENT_ACTION)
        assert span.attributes[ATTR_AGENT_ACTION_SUCCESS] is True
        assert span.attributes[ATTR_AGENT_ACTION_LOOP_DETECTED] is False
        assert span.status.status_code == StatusCode.OK


# ---------------------------------------------------------------------------
# Section 15 — Agent Action Parenting (end-to-end)
# ---------------------------------------------------------------------------


class TestAgentActionParenting:
    def test_action_is_parent_of_tool_execute(self):
        tracer, exporter = make_recording_tracer()
        controller, _, _ = _make_traced_controller(
            tracer, handler=SuccessTool("ok")
        )

        _run(
            controller.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )

        action_span = exporter.span_by_name(SPAN_AGENT_ACTION)
        exec_spans = exporter.spans_named(SPAN_TOOL_EXECUTE)
        assert len(exec_spans) == 1
        assert exec_spans[0].parent is not None
        assert exec_spans[0].parent.span_id == action_span.context.span_id

    def test_attempt_is_grandchild_of_action(self):
        tracer, exporter = make_recording_tracer()
        controller, _, _ = _make_traced_controller(
            tracer, handler=SuccessTool("ok")
        )

        _run(
            controller.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )

        action_span = exporter.span_by_name(SPAN_AGENT_ACTION)
        attempt_spans = exporter.spans_named(SPAN_TOOL_ATTEMPT)
        assert len(attempt_spans) == 1
        assert attempt_spans[0].parent is not None
        # attempt's parent should be tool.execute, whose parent is agent.action
        exec_spans = exporter.spans_named(SPAN_TOOL_EXECUTE)
        assert attempt_spans[0].parent.span_id == exec_spans[0].context.span_id
        assert exec_spans[0].parent.span_id == action_span.context.span_id

    def test_same_trace_id(self):
        tracer, exporter = make_recording_tracer()
        controller, _, _ = _make_traced_controller(
            tracer, handler=SuccessTool("ok")
        )

        _run(
            controller.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )

        action_span = exporter.span_by_name(SPAN_AGENT_ACTION)
        exec_span = exporter.spans_named(SPAN_TOOL_EXECUTE)[0]
        attempt_span = exporter.spans_named(SPAN_TOOL_ATTEMPT)[0]
        assert action_span.context.trace_id == exec_span.context.trace_id
        assert action_span.context.trace_id == attempt_span.context.trace_id

    def test_different_tracer_instances_same_provider(self):
        """Controller and ToolRuntime can use different tracer instances
        from the same provider; parenting still works via OTel context."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from tests.fakes_telemetry import RecordingSpanExporter

        provider = TracerProvider()
        exporter = RecordingSpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer_a = provider.get_tracer("controller_scope")
        tracer_b = provider.get_tracer("runtime_scope")

        registry = ToolRegistry()
        spec = _spec()
        registry.register(spec, SuccessTool("ok"))
        runtime = ToolRuntime(registry=registry, tracer=tracer_b, sleep=FakeSleeper())
        detector = LoopDetector(
            config=LoopDetectorConfig(), history=ActionHistory(max_size=64)
        )
        controller = AgentActionController(
            tool_runtime=runtime,
            loop_detector=detector,
            progress_provider=ConstantProgressProvider("s1"),
            tracer=tracer_a,
        )

        _run(
            controller.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )

        action_span = exporter.span_by_name(SPAN_AGENT_ACTION)
        exec_spans = exporter.spans_named(SPAN_TOOL_EXECUTE)
        assert len(exec_spans) == 1
        assert exec_spans[0].parent.span_id == action_span.context.span_id


# ---------------------------------------------------------------------------
# Section 18 — Failed Tool Action
# ---------------------------------------------------------------------------


class TestFailedToolAction:
    def test_failed_tool_action_error_status(self):
        tracer, exporter = make_recording_tracer()
        controller, _, _ = _make_traced_controller(
            tracer, handler=ScriptedTool([("permanent", "fail")])
        )

        _run(
            controller.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )

        span = exporter.span_by_name(SPAN_AGENT_ACTION)
        assert span.attributes[ATTR_AGENT_ACTION_SUCCESS] is False
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Section 19-20 — Loop Detection & Replan Required Events
# ---------------------------------------------------------------------------


class TestLoopDetectionEvents:
    def _trigger_loop(self, tracer, exporter):
        """Execute 3 identical actions to trigger loop detection."""
        controller, _, _ = _make_traced_controller(
            tracer,
            handler=SuccessTool("ok"),
            progress_provider=ConstantProgressProvider("same"),
        )
        call = ToolCall(tool_name="tool", arguments={})
        ctx = ToolExecutionContext()

        for _ in range(3):
            _run(controller.execute(call, ctx))
        return controller, exporter

    def test_loop_detected_event(self):
        tracer, exporter = make_recording_tracer()
        self._trigger_loop(tracer, exporter)

        action_spans = exporter.spans_named(SPAN_AGENT_ACTION)
        # The 3rd action should have loop detected event.
        third = action_spans[2]
        loop_events = [e for e in third.events if e.name == EVENT_AGENT_LOOP_DETECTED]
        assert len(loop_events) == 1
        assert ATTR_AGENT_LOOP_REASON in loop_events[0].attributes
        assert ATTR_AGENT_LOOP_EVIDENCE_COUNT in loop_events[0].attributes

    def test_replan_required_event(self):
        tracer, exporter = make_recording_tracer()
        self._trigger_loop(tracer, exporter)

        action_spans = exporter.spans_named(SPAN_AGENT_ACTION)
        third = action_spans[2]
        replan_events = [
            e for e in third.events if e.name == EVENT_AGENT_REPLAN_REQUIRED
        ]
        assert len(replan_events) == 1

    def test_loop_detected_attribute(self):
        tracer, exporter = make_recording_tracer()
        self._trigger_loop(tracer, exporter)

        action_spans = exporter.spans_named(SPAN_AGENT_ACTION)
        third = action_spans[2]
        assert third.attributes[ATTR_AGENT_ACTION_LOOP_DETECTED] is True

    def test_controller_enters_replan_required(self):
        tracer, exporter = make_recording_tracer()
        controller, _ = self._trigger_loop(tracer, exporter)
        assert controller.state is ActionControllerState.REPLAN_REQUIRED


# ---------------------------------------------------------------------------
# Section 21-22 — Replan Gate Block
# ---------------------------------------------------------------------------


class TestReplanGateBlock:
    def test_blocked_action_has_no_tool_execute(self):
        tracer, exporter = make_recording_tracer()
        controller, _, _ = _make_traced_controller(
            tracer,
            handler=SuccessTool("ok"),
            progress_provider=ConstantProgressProvider("same"),
        )
        call = ToolCall(tool_name="tool", arguments={})
        ctx = ToolExecutionContext()

        # Trigger loop detection (3 identical actions).
        for _ in range(3):
            _run(controller.execute(call, ctx))
        assert controller.state is ActionControllerState.REPLAN_REQUIRED

        # 4th action should be blocked.
        with pytest.raises(ReplanRequiredError):
            _run(controller.execute(call, ctx))

        # The 4th agent.action span should have NO tool.execute child.
        action_spans = exporter.spans_named(SPAN_AGENT_ACTION)
        blocked_span = action_spans[3]
        blocked_events = [
            e for e in blocked_span.events if e.name == EVENT_AGENT_REPLAN_BLOCKED
        ]
        assert len(blocked_events) == 1

        # No new tool.execute spans (still only 3 from before).
        exec_spans = exporter.spans_named(SPAN_TOOL_EXECUTE)
        assert len(exec_spans) == 3

    def test_blocked_action_error_status(self):
        tracer, exporter = make_recording_tracer()
        controller, _, _ = _make_traced_controller(
            tracer,
            handler=SuccessTool("ok"),
            progress_provider=ConstantProgressProvider("same"),
        )
        call = ToolCall(tool_name="tool", arguments={})
        ctx = ToolExecutionContext()

        for _ in range(3):
            _run(controller.execute(call, ctx))

        with pytest.raises(ReplanRequiredError):
            _run(controller.execute(call, ctx))

        action_spans = exporter.spans_named(SPAN_AGENT_ACTION)
        assert action_spans[3].status.status_code == StatusCode.ERROR

    def test_blocked_action_has_next_index_not_consumed(self):
        tracer, exporter = make_recording_tracer()
        controller, _, _ = _make_traced_controller(
            tracer,
            handler=SuccessTool("ok"),
            progress_provider=ConstantProgressProvider("same"),
        )
        call = ToolCall(tool_name="tool", arguments={})
        ctx = ToolExecutionContext()

        for _ in range(3):
            _run(controller.execute(call, ctx))

        with pytest.raises(ReplanRequiredError):
            _run(controller.execute(call, ctx))

        action_spans = exporter.spans_named(SPAN_AGENT_ACTION)
        blocked = action_spans[3]
        # next_index should be 3 (not consumed), not a fake action index.
        assert blocked.attributes[ATTR_AGENT_ACTION_NEXT_INDEX] == 3
        # No ATTR_AGENT_ACTION_INDEX (not consumed).
        from observability import ATTR_AGENT_ACTION_INDEX
        assert ATTR_AGENT_ACTION_INDEX not in blocked.attributes


# ---------------------------------------------------------------------------
# Section 23-24 — Replan Acknowledge
# ---------------------------------------------------------------------------


class TestReplanAcknowledge:
    def test_acknowledge_span_exists(self):
        tracer, exporter = make_recording_tracer()
        controller, _, _ = _make_traced_controller(
            tracer,
            handler=SuccessTool("ok"),
            progress_provider=ConstantProgressProvider("same"),
        )
        call = ToolCall(tool_name="tool", arguments={})
        ctx = ToolExecutionContext()

        for _ in range(3):
            _run(controller.execute(call, ctx))

        controller.acknowledge_replan()

        ack_spans = exporter.spans_named(SPAN_AGENT_REPLAN_ACKNOWLEDGE)
        assert len(ack_spans) == 1

    def test_acknowledge_ok_status(self):
        tracer, exporter = make_recording_tracer()
        controller, _, _ = _make_traced_controller(
            tracer,
            handler=SuccessTool("ok"),
            progress_provider=ConstantProgressProvider("same"),
        )
        call = ToolCall(tool_name="tool", arguments={})
        ctx = ToolExecutionContext()

        for _ in range(3):
            _run(controller.execute(call, ctx))

        controller.acknowledge_replan()

        ack_spans = exporter.spans_named(SPAN_AGENT_REPLAN_ACKNOWLEDGE)
        assert ack_spans[0].status.status_code == StatusCode.OK

    def test_acknowledge_has_next_index(self):
        tracer, exporter = make_recording_tracer()
        controller, _, _ = _make_traced_controller(
            tracer,
            handler=SuccessTool("ok"),
            progress_provider=ConstantProgressProvider("same"),
        )
        call = ToolCall(tool_name="tool", arguments={})
        ctx = ToolExecutionContext()

        for _ in range(3):
            _run(controller.execute(call, ctx))

        controller.acknowledge_replan()

        ack_spans = exporter.spans_named(SPAN_AGENT_REPLAN_ACKNOWLEDGE)
        assert ack_spans[0].attributes[ATTR_AGENT_ACTION_NEXT_INDEX] == 3

    def test_acknowledge_illegal_error_status(self):
        tracer, exporter = make_recording_tracer()
        controller, _, _ = _make_traced_controller(
            tracer, handler=SuccessTool("ok")
        )

        with pytest.raises(InvalidActionControllerStateError):
            controller.acknowledge_replan()

        ack_spans = exporter.spans_named(SPAN_AGENT_REPLAN_ACKNOWLEDGE)
        assert len(ack_spans) == 1
        assert ack_spans[0].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Section 13-14 — No Arguments / Fingerprint in Telemetry
# ---------------------------------------------------------------------------


class TestNoSensitiveData:
    def test_no_arguments_in_telemetry(self):
        tracer, exporter = make_recording_tracer()
        controller, _, _ = _make_traced_controller(
            tracer, handler=SuccessTool("ok")
        )

        _run(
            controller.execute(
                ToolCall(
                    tool_name="tool",
                    arguments={"password": "SECRET_PASSWORD_MARKER"},
                ),
                ToolExecutionContext(),
            )
        )

        for s in exporter.finished_spans:
            for key, val in (s.attributes or {}).items():
                assert "SECRET_PASSWORD_MARKER" not in str(val)
                assert "password" not in str(key)
            for ev in s.events:
                for key, val in (ev.attributes or {}).items():
                    assert "SECRET_PASSWORD_MARKER" not in str(val)

    def test_no_fingerprint_in_telemetry(self):
        tracer, exporter = make_recording_tracer()
        controller, _, _ = _make_traced_controller(
            tracer, handler=SuccessTool("ok")
        )

        _run(
            controller.execute(
                ToolCall(tool_name="tool", arguments={"x": 1}),
                ToolExecutionContext(),
            )
        )

        for s in exporter.finished_spans:
            for key in (s.attributes or {}).keys():
                assert "fingerprint" not in str(key).lower()
                assert "canonical" not in str(key).lower()
                assert "hash" not in str(key).lower()


# ---------------------------------------------------------------------------
# Section 67 — Deterministic Span Names
# ---------------------------------------------------------------------------


class TestDeterministicSpanNames:
    def test_span_names_are_fixed(self):
        tracer, exporter = make_recording_tracer()
        controller, _, _ = _make_traced_controller(
            tracer, handler=SuccessTool("ok")
        )

        _run(
            controller.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )

        all_names = {s.name for s in exporter.finished_spans}
        assert all_names <= {
            SPAN_AGENT_ACTION,
            SPAN_TOOL_EXECUTE,
            SPAN_TOOL_ATTEMPT,
        }


# ---------------------------------------------------------------------------
# Section 50 — No SDK Configured
# ---------------------------------------------------------------------------


class TestNoSDKConfigured:
    def test_business_behavior_without_tracer(self):
        registry = ToolRegistry()
        spec = _spec()
        registry.register(spec, SuccessTool("ok"))
        runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())
        detector = LoopDetector(
            config=LoopDetectorConfig(), history=ActionHistory(max_size=64)
        )
        controller = AgentActionController(
            tool_runtime=runtime,
            loop_detector=detector,
            progress_provider=ConstantProgressProvider("s1"),
        )

        result = _run(
            controller.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )
        assert result.tool_result.success is True
        assert controller.next_action_index == 1


# ---------------------------------------------------------------------------
# Section 55 — Telemetry Failure Isolation
# ---------------------------------------------------------------------------


class FailingSpan:
    def set_attribute(self, key, value):
        raise RuntimeError("telemetry failure")

    def add_event(self, name, attributes=None):
        raise RuntimeError("telemetry failure")

    def record_exception(self, exc):
        raise RuntimeError("telemetry failure")

    def set_status(self, status):
        raise RuntimeError("telemetry failure")

    @property
    def context(self):
        class _Ctx:
            span_id = 0
            trace_id = 0
        return _Ctx()

    @property
    def parent(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FailingTracer:
    def start_as_current_span(self, name, **kwargs):
        raise RuntimeError("span creation failure")


class TestTelemetryFailureIsolation:
    def test_controller_business_unaffected_by_failing_tracer(self):
        registry = ToolRegistry()
        spec = _spec()
        registry.register(spec, SuccessTool("ok"))
        runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())
        detector = LoopDetector(
            config=LoopDetectorConfig(), history=ActionHistory(max_size=64)
        )
        controller = AgentActionController(
            tool_runtime=runtime,
            loop_detector=detector,
            progress_provider=ConstantProgressProvider("s1"),
            tracer=FailingTracer(),
        )

        result = _run(
            controller.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )
        assert result.tool_result.success is True
        assert controller.next_action_index == 1

    def test_acknowledge_unaffected_by_failing_tracer(self):
        registry = ToolRegistry()
        spec = _spec()
        registry.register(spec, SuccessTool("ok"))
        runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())
        detector = LoopDetector(
            config=LoopDetectorConfig(), history=ActionHistory(max_size=64)
        )
        controller = AgentActionController(
            tool_runtime=runtime,
            loop_detector=detector,
            progress_provider=ConstantProgressProvider("same"),
            tracer=FailingTracer(),
        )
        call = ToolCall(tool_name="tool", arguments={})
        ctx = ToolExecutionContext()

        for _ in range(3):
            _run(controller.execute(call, ctx))
        assert controller.state is ActionControllerState.REPLAN_REQUIRED

        # acknowledge should still work.
        controller.acknowledge_replan()
        assert controller.state is ActionControllerState.READY


# ---------------------------------------------------------------------------
# Section 47 — Retry End-to-End Action Trace
# ---------------------------------------------------------------------------


class TestRetryEndToEnd:
    def test_retry_then_success_action_ok(self):
        tracer, exporter = make_recording_tracer()
        controller, _, handler = _make_traced_controller(
            tracer,
            handler=ScriptedTool([("transient", "fail"), ("success", "ok")]),
            spec=_spec(retry_policy=RetryPolicy(max_attempts=2, initial_backoff_seconds=0.1)),
        )

        result = _run(
            controller.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )

        assert result.tool_result.success is True
        assert result.tool_result.attempt_count == 2

        action_span = exporter.span_by_name(SPAN_AGENT_ACTION)
        assert action_span.status.status_code == StatusCode.OK
        assert action_span.attributes[ATTR_AGENT_ACTION_SUCCESS] is True

        # 2 attempt spans, 1 ERROR, 1 OK.
        attempt_spans = exporter.spans_named(SPAN_TOOL_ATTEMPT)
        assert len(attempt_spans) == 2
        assert attempt_spans[0].status.status_code == StatusCode.ERROR
        assert attempt_spans[1].status.status_code == StatusCode.OK

        # tool.execute OK.
        exec_spans = exporter.spans_named(SPAN_TOOL_EXECUTE)
        assert exec_spans[0].status.status_code == StatusCode.OK
