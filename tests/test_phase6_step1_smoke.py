"""Phase 6 Step 1 smoke test — OpenTelemetry tracing foundation.

Runs a Harness fresh Run (two successful steps → complete) and a
ToolRuntime call (transient → retry → success) under a recording
tracer, then asserts the combined trace structure:

    harness.run
    ├─ harness.step
    └─ harness.step

    tool.execute
    ├─ tool.attempt ERROR
    └─ tool.attempt OK

The two systems are NOT integrated into a single trace tree (that is
Phase 6 Step 2). This smoke test proves each subsystem produces
correct, observable telemetry independently.

All offline, deterministic, no network, no collector, no real LLM.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from opentelemetry.trace.status import Status, StatusCode

from harness import HarnessRuntime, Task
from observability import (
    ATTR_CHECKPOINT_ID,
    ATTR_RUN_ID,
    ATTR_RUN_RESUME,
    ATTR_RUN_STATUS,
    ATTR_STEP_INDEX,
    ATTR_TASK_ID,
    ATTR_TOOL_ATTEMPT_COUNT,
    ATTR_TOOL_NAME,
    ATTR_TOOL_RESULT_SUCCESS,
    EVENT_CHECKPOINT_SAVED,
    EVENT_RETRY_SCHEDULED,
    RUN_STATUS_COMPLETED,
    SPAN_HARNESS_RUN,
    SPAN_HARNESS_STEP,
    SPAN_TOOL_ATTEMPT,
    SPAN_TOOL_EXECUTE,
)
from storage import InMemoryCheckpointStore
from tests.fakes import FakeClock, FakeStepExecutor, make_id_factory
from tests.fakes_telemetry import make_recording_tracer
from tests.fake_tools import EchoTool, FakeSleeper, ScriptedTool
from tools import (
    ToolCall,
    ToolExecutionContext,
    ToolRegistry,
    ToolRuntime,
    ToolSideEffect,
    ToolSpec,
)
from tools.models import RetryPolicy

T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


class TestPhase6Step1Smoke:
    def test_harness_and_tool_tracing(self):
        """Both HarnessRuntime and ToolRuntime produce correct traces."""
        tracer, exporter = make_recording_tracer()

        # ------------------------------------------------------------------
        # Part 1: Harness fresh Run with 2 successful steps.
        # ------------------------------------------------------------------
        store = InMemoryCheckpointStore()
        executor = FakeStepExecutor(complete_at_step=1)  # steps 0, 1
        clock = FakeClock(T0)
        harness = HarnessRuntime(
            checkpoint_store=store,
            step_executor=executor,
            clock=clock,
            run_id_factory=make_id_factory("R"),
            checkpoint_id_factory=make_id_factory("C"),
            tracer=tracer,
        )
        task = Task(task_id="T-SMOKE", goal="smoke test", created_at=T0)

        run = _run(harness.start(task))

        # ------------------------------------------------------------------
        # Part 2: ToolRuntime with transient → retry → success.
        # ------------------------------------------------------------------
        registry = ToolRegistry()
        handler = ScriptedTool([
            ("transient", "fail"),
            ("success", "ok"),
        ])
        sleeper = FakeSleeper()
        spec = ToolSpec(
            name="smoke_tool",
            description="smoke test tool",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            required_permissions=frozenset(),
            timeout_seconds=1.0,
            side_effect=ToolSideEffect.READ_ONLY,
            retry_policy=RetryPolicy(
                max_attempts=2, initial_backoff_seconds=0.1,
            ),
        )
        registry.register(spec, handler)
        tool_runtime = ToolRuntime(
            registry=registry, tracer=tracer, sleep=sleeper,
        )

        tool_result = _run(
            tool_runtime.execute(
                ToolCall(tool_name="smoke_tool", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        # ------------------------------------------------------------------
        # Assert Harness trace.
        # ------------------------------------------------------------------
        run_spans = exporter.spans_named(SPAN_HARNESS_RUN)
        assert len(run_spans) == 1
        assert run_spans[0].attributes[ATTR_TASK_ID] == "T-SMOKE"
        assert run_spans[0].attributes[ATTR_RUN_ID] == run.run_id
        assert run_spans[0].attributes[ATTR_RUN_RESUME] is False
        assert run_spans[0].attributes[ATTR_RUN_STATUS] == RUN_STATUS_COMPLETED
        assert run_spans[0].status.status_code == StatusCode.OK

        step_spans = exporter.spans_named(SPAN_HARNESS_STEP)
        assert len(step_spans) == 2
        for s in step_spans:
            assert s.parent is not None
            assert s.parent.span_id == run_spans[0].context.span_id
            assert s.status.status_code == StatusCode.OK
            cp_events = [e for e in s.events if e.name == EVENT_CHECKPOINT_SAVED]
            assert len(cp_events) == 1
            assert ATTR_CHECKPOINT_ID in cp_events[0].attributes

        # ------------------------------------------------------------------
        # Assert Tool trace.
        # ------------------------------------------------------------------
        exec_spans = exporter.spans_named(SPAN_TOOL_EXECUTE)
        assert len(exec_spans) == 1
        assert exec_spans[0].attributes[ATTR_TOOL_NAME] == "smoke_tool"
        assert exec_spans[0].attributes[ATTR_TOOL_RESULT_SUCCESS] is True
        assert exec_spans[0].attributes[ATTR_TOOL_ATTEMPT_COUNT] == 2
        assert exec_spans[0].status.status_code == StatusCode.OK

        attempt_spans = exporter.spans_named(SPAN_TOOL_ATTEMPT)
        assert len(attempt_spans) == 2
        assert attempt_spans[0].parent.span_id == exec_spans[0].context.span_id
        assert attempt_spans[1].parent.span_id == exec_spans[0].context.span_id
        assert attempt_spans[0].status.status_code == StatusCode.ERROR
        assert attempt_spans[1].status.status_code == StatusCode.OK

        retry_events = [
            e for e in exec_spans[0].events if e.name == EVENT_RETRY_SCHEDULED
        ]
        assert len(retry_events) == 1

        # ------------------------------------------------------------------
        # Assert the two trace trees are NOT integrated.
        # ------------------------------------------------------------------
        # tool.execute should NOT be a child of harness.run.
        assert exec_spans[0].parent is None or (
            exec_spans[0].parent.span_id != run_spans[0].context.span_id
        )

    def test_business_results_correct(self):
        """Telemetry does not change business results."""
        tracer, exporter = make_recording_tracer()

        # Harness.
        store = InMemoryCheckpointStore()
        executor = FakeStepExecutor(complete_at_step=1)
        clock = FakeClock(T0)
        harness = HarnessRuntime(
            checkpoint_store=store,
            step_executor=executor,
            clock=clock,
            run_id_factory=make_id_factory("R"),
            checkpoint_id_factory=make_id_factory("C"),
            tracer=tracer,
        )
        task = Task(task_id="T-BIZ", goal="biz", created_at=T0)

        run = _run(harness.start(task))
        assert run.current_step == 2
        assert run.last_checkpoint_id == "C-002"

        # Tool.
        registry = ToolRegistry()
        handler = ScriptedTool([("transient", "fail"), ("success", "ok")])
        sleeper = FakeSleeper()
        spec = ToolSpec(
            name="t",
            description="t",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            required_permissions=frozenset(),
            timeout_seconds=1.0,
            side_effect=ToolSideEffect.READ_ONLY,
            retry_policy=RetryPolicy(max_attempts=2, initial_backoff_seconds=0.1),
        )
        registry.register(spec, handler)
        tool_runtime = ToolRuntime(registry=registry, tracer=tracer, sleep=sleeper)

        result = _run(
            tool_runtime.execute(
                ToolCall(tool_name="t", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )
        assert result.success is True
        assert result.output == "ok"
        assert result.attempt_count == 2
        assert handler.call_count == 2
        assert sleeper.delays == [0.1]

    def test_no_sdk_business_still_works(self):
        """Without any tracer, business behavior is unchanged."""
        store = InMemoryCheckpointStore()
        executor = FakeStepExecutor(complete_at_step=1)
        clock = FakeClock(T0)
        harness = HarnessRuntime(
            checkpoint_store=store,
            step_executor=executor,
            clock=clock,
            run_id_factory=make_id_factory("R"),
            checkpoint_id_factory=make_id_factory("C"),
        )
        task = Task(task_id="T-NOSDK", goal="nosdk", created_at=T0)

        run = _run(harness.start(task))
        assert run.current_step == 2
        assert run.status.value == "completed"

        registry = ToolRegistry()
        handler = EchoTool()
        spec = ToolSpec(
            name="echo",
            description="echo",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            required_permissions=frozenset(),
            timeout_seconds=1.0,
            side_effect=ToolSideEffect.READ_ONLY,
        )
        registry.register(spec, handler)
        tool_runtime = ToolRuntime(registry=registry)

        result = _run(
            tool_runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "hello"}),
                ToolExecutionContext(),
            )
        )
        assert result.success is True
        assert result.output == "hello"


# ---------------------------------------------------------------------------
# Section 43 — Tracer Failure Isolation
# ---------------------------------------------------------------------------


class FailingSpan:
    """A span object whose telemetry methods raise exceptions.

    Used to prove that telemetry failures never become business failures.
    The ``safe_*`` helpers in ``observability.tracing`` catch these.
    """

    def set_attribute(self, key, value):
        raise RuntimeError("telemetry failure: set_attribute")

    def add_event(self, name, attributes=None):
        raise RuntimeError("telemetry failure: add_event")

    def record_exception(self, exc):
        raise RuntimeError("telemetry failure: record_exception")

    def set_status(self, status):
        raise RuntimeError("telemetry failure: set_status")

    @property
    def context(self):
        class _Ctx:
            span_id = 0
        return _Ctx()

    @property
    def parent(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FailingTracer:
    """A tracer whose ``start_as_current_span`` returns a FailingSpan."""

    def start_as_current_span(self, name, **kwargs):
        return FailingSpan()


class TestTracerFailureIsolation:
    def test_harness_business_unaffected_by_telemetry_failure(self):
        """HarnessRuntime produces correct results even if tracer throws."""
        store = InMemoryCheckpointStore()
        executor = FakeStepExecutor(complete_at_step=1)
        clock = FakeClock(T0)
        harness = HarnessRuntime(
            checkpoint_store=store,
            step_executor=executor,
            clock=clock,
            run_id_factory=make_id_factory("R"),
            checkpoint_id_factory=make_id_factory("C"),
            tracer=FailingTracer(),
        )
        task = Task(task_id="T-FAIL", goal="fail tracer", created_at=T0)

        run = _run(harness.start(task))
        assert run.current_step == 2
        assert run.status.value == "completed"

    def test_tool_business_unaffected_by_telemetry_failure(self):
        """ToolRuntime produces correct results even if tracer throws."""
        registry = ToolRegistry()
        handler = EchoTool()
        spec = ToolSpec(
            name="echo",
            description="echo",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            required_permissions=frozenset(),
            timeout_seconds=1.0,
            side_effect=ToolSideEffect.READ_ONLY,
        )
        registry.register(spec, handler)
        tool_runtime = ToolRuntime(registry=registry, tracer=FailingTracer())

        result = _run(
            tool_runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "hello"}),
                ToolExecutionContext(),
            )
        )
        assert result.success is True
        assert result.output == "hello"
        assert result.attempt_count == 1

    def test_tool_retry_unaffected_by_telemetry_failure(self):
        """Retry logic still works correctly even if tracer throws."""
        from tools.models import RetryPolicy
        registry = ToolRegistry()
        handler = ScriptedTool([("transient", "fail"), ("success", "ok")])
        sleeper = FakeSleeper()
        spec = ToolSpec(
            name="t",
            description="t",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            required_permissions=frozenset(),
            timeout_seconds=1.0,
            side_effect=ToolSideEffect.READ_ONLY,
            retry_policy=RetryPolicy(max_attempts=2, initial_backoff_seconds=0.1),
        )
        registry.register(spec, handler)
        tool_runtime = ToolRuntime(
            registry=registry, tracer=FailingTracer(), sleep=sleeper,
        )

        result = _run(
            tool_runtime.execute(
                ToolCall(tool_name="t", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )
        assert result.success is True
        assert result.attempt_count == 2
        assert sleeper.delays == [0.1]
