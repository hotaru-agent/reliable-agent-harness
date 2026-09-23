"""Tests for ToolRuntime OpenTelemetry tracing (Phase 6 Step 1).

All offline, deterministic, no network, no collector. Uses a recording
span exporter to assert on span names, attributes, events, statuses
and parent-child relationships.

Tests prove:
* tool.execute span exists with tool identity attributes
* tool.attempt child spans exist for handler attempts
* success: parent OK, child OK
* preflight failure: execute span exists, 0 attempt children
* retry then success: attempt 1 ERROR, attempt 2 ERROR, attempt 3 OK, parent OK
* retry backoff attributes match FakeSleeper delays
* SIDE_EFFECTING retry block produces no retry event
* timeout: attempt ERROR, parent ERROR
* ordinary handler exception: exception recorded on attempt span
* CancelledError propagates, UNSET status
* MCP source attributes
* Local source attributes
* no sensitive output/arguments in telemetry
* tracer injection isolation
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from opentelemetry.trace.status import Status, StatusCode

from observability import (
    ATTR_TOOL_ATTEMPT_COUNT,
    ATTR_TOOL_ATTEMPT_MAX,
    ATTR_TOOL_ATTEMPT_NUMBER,
    ATTR_TOOL_ERROR_RETRYABLE,
    ATTR_TOOL_ERROR_TYPE,
    ATTR_TOOL_NAME,
    ATTR_TOOL_PROVIDER_ID,
    ATTR_TOOL_RESULT_SUCCESS,
    ATTR_TOOL_RETRY_BACKOFF_SECONDS,
    ATTR_TOOL_RETRY_NEXT_ATTEMPT,
    ATTR_TOOL_SIDE_EFFECT,
    ATTR_TOOL_SOURCE,
    EVENT_RETRY_SCHEDULED,
    EVENT_TOOL_CANCELLED,
    SPAN_TOOL_ATTEMPT,
    SPAN_TOOL_EXECUTE,
)
from tests.fake_tools import (
    CooperativeCancelTool,
    EchoTool,
    FakeSleeper,
    PermanentFailureTool,
    RaisingTool,
    ScriptedTool,
    SlowTool,
    TransientFailureTool,
)
from tests.fakes_telemetry import make_recording_tracer
from tools import (
    ToolCall,
    ToolErrorType,
    ToolExecutionContext,
    ToolRegistry,
    ToolRuntime,
    ToolSideEffect,
    ToolSource,
    ToolSpec,
)

# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

_ECHO_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


def _echo_spec(
    name: str = "echo",
    *,
    required_permissions: frozenset[str] = frozenset(),
    timeout_seconds: float = 1.0,
    side_effect: ToolSideEffect = ToolSideEffect.READ_ONLY,
    retry_policy=None,
    source: ToolSource = ToolSource.LOCAL,
    provider_id: str | None = None,
) -> ToolSpec:
    from tools.models import RetryPolicy
    return ToolSpec(
        name=name,
        description="echo the text argument",
        input_schema=_ECHO_SCHEMA,
        required_permissions=required_permissions,
        timeout_seconds=timeout_seconds,
        side_effect=side_effect,
        retry_policy=retry_policy or RetryPolicy(max_attempts=1),
        source=source,
        provider_id=provider_id,
    )


def _make_traced_runtime(
    tracer,
    *,
    spec=None,
    handler=None,
    name="echo",
    sleep=None,
) -> tuple[ToolRuntime, ToolRegistry, Any]:
    registry = ToolRegistry()
    handler = handler or EchoTool()
    spec = spec or _echo_spec(name)
    registry.register(spec, handler)
    return (
        ToolRuntime(registry=registry, tracer=tracer, sleep=sleep),
        registry,
        handler,
    )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Section 56 — Tool Success
# ---------------------------------------------------------------------------


class TestToolSuccessTracing:
    def test_execute_span_exists(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "hello"}),
                ToolExecutionContext(),
            )
        )

        exec_spans = exporter.spans_named(SPAN_TOOL_EXECUTE)
        assert len(exec_spans) == 1

    def test_execute_span_has_tool_name(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "hello"}),
                ToolExecutionContext(),
            )
        )

        exec_span = exporter.span_by_name(SPAN_TOOL_EXECUTE)
        assert exec_span.attributes[ATTR_TOOL_NAME] == "echo"

    def test_attempt_span_exists(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "hello"}),
                ToolExecutionContext(),
            )
        )

        attempt_spans = exporter.spans_named(SPAN_TOOL_ATTEMPT)
        assert len(attempt_spans) == 1

    def test_attempt_is_child_of_execute(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "hello"}),
                ToolExecutionContext(),
            )
        )

        exec_span = exporter.span_by_name(SPAN_TOOL_EXECUTE)
        attempt_spans = exporter.spans_named(SPAN_TOOL_ATTEMPT)
        assert attempt_spans[0].parent is not None
        assert attempt_spans[0].parent.span_id == exec_span.context.span_id

    def test_both_spans_ok(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "hello"}),
                ToolExecutionContext(),
            )
        )

        exec_span = exporter.span_by_name(SPAN_TOOL_EXECUTE)
        attempt_spans = exporter.spans_named(SPAN_TOOL_ATTEMPT)
        assert exec_span.status.status_code == StatusCode.OK
        assert attempt_spans[0].status.status_code == StatusCode.OK

    def test_attempt_count_attribute(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "hello"}),
                ToolExecutionContext(),
            )
        )

        exec_span = exporter.span_by_name(SPAN_TOOL_EXECUTE)
        assert exec_span.attributes[ATTR_TOOL_ATTEMPT_COUNT] == 1
        assert exec_span.attributes[ATTR_TOOL_RESULT_SUCCESS] is True

    def test_attempt_number_attribute(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "hello"}),
                ToolExecutionContext(),
            )
        )

        attempt_spans = exporter.spans_named(SPAN_TOOL_ATTEMPT)
        assert attempt_spans[0].attributes[ATTR_TOOL_ATTEMPT_NUMBER] == 1


# ---------------------------------------------------------------------------
# Section 57 — Tool Preflight Failure
# ---------------------------------------------------------------------------


class TestPreflightFailureTracing:
    def test_not_found_execute_span_exists(self):
        tracer, exporter = make_recording_tracer()
        registry = ToolRegistry()
        runtime = ToolRuntime(registry=registry, tracer=tracer)

        _run(
            runtime.execute(
                ToolCall(tool_name="unknown", arguments={}),
                ToolExecutionContext(),
            )
        )

        exec_spans = exporter.spans_named(SPAN_TOOL_EXECUTE)
        assert len(exec_spans) == 1

    def test_not_found_zero_attempt_children(self):
        tracer, exporter = make_recording_tracer()
        registry = ToolRegistry()
        runtime = ToolRuntime(registry=registry, tracer=tracer)

        _run(
            runtime.execute(
                ToolCall(tool_name="unknown", arguments={}),
                ToolExecutionContext(),
            )
        )

        attempt_spans = exporter.spans_named(SPAN_TOOL_ATTEMPT)
        assert len(attempt_spans) == 0

    def test_not_found_attributes(self):
        tracer, exporter = make_recording_tracer()
        registry = ToolRegistry()
        runtime = ToolRuntime(registry=registry, tracer=tracer)

        _run(
            runtime.execute(
                ToolCall(tool_name="unknown", arguments={}),
                ToolExecutionContext(),
            )
        )

        exec_span = exporter.span_by_name(SPAN_TOOL_EXECUTE)
        assert exec_span.attributes[ATTR_TOOL_RESULT_SUCCESS] is False
        assert exec_span.attributes[ATTR_TOOL_ATTEMPT_COUNT] == 0
        assert exec_span.attributes[ATTR_TOOL_ERROR_TYPE] == "not_found"

    def test_not_found_error_status(self):
        tracer, exporter = make_recording_tracer()
        registry = ToolRegistry()
        runtime = ToolRuntime(registry=registry, tracer=tracer)

        _run(
            runtime.execute(
                ToolCall(tool_name="unknown", arguments={}),
                ToolExecutionContext(),
            )
        )

        exec_span = exporter.span_by_name(SPAN_TOOL_EXECUTE)
        assert exec_span.status.status_code == StatusCode.ERROR

    def test_validation_failure(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"wrong": "x"}),
                ToolExecutionContext(),
            )
        )

        exec_span = exporter.span_by_name(SPAN_TOOL_EXECUTE)
        assert exec_span.attributes[ATTR_TOOL_ERROR_TYPE] == "validation"
        assert exec_span.attributes[ATTR_TOOL_ATTEMPT_COUNT] == 0
        assert exec_span.status.status_code == StatusCode.ERROR
        assert len(exporter.spans_named(SPAN_TOOL_ATTEMPT)) == 0

    def test_permission_failure(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(
            tracer,
            spec=_echo_spec(required_permissions=frozenset({"admin"})),
        )

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(granted_permissions=frozenset()),
            )
        )

        exec_span = exporter.span_by_name(SPAN_TOOL_EXECUTE)
        assert exec_span.attributes[ATTR_TOOL_ERROR_TYPE] == "permission"
        assert exec_span.attributes[ATTR_TOOL_ATTEMPT_COUNT] == 0
        assert exec_span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Section 58 — Tool Retry Then Success
# ---------------------------------------------------------------------------


class TestRetryThenSuccessTracing:
    def test_retry_then_success_spans(self):
        from tools.models import RetryPolicy
        tracer, exporter = make_recording_tracer()
        handler = ScriptedTool([
            ("transient", "fail 1"),
            ("transient", "fail 2"),
            ("success", "ok"),
        ])
        sleeper = FakeSleeper()
        spec = _echo_spec(
            retry_policy=RetryPolicy(
                max_attempts=3,
                initial_backoff_seconds=1.0,
                backoff_multiplier=2.0,
            ),
        )
        runtime, _, _ = _make_traced_runtime(
            tracer, spec=spec, handler=handler, sleep=sleeper
        )

        result = _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        assert result.success is True
        assert result.attempt_count == 3

        # 3 attempt spans.
        attempt_spans = exporter.spans_named(SPAN_TOOL_ATTEMPT)
        assert len(attempt_spans) == 3

        # attempt 1 ERROR, attempt 2 ERROR, attempt 3 OK.
        assert attempt_spans[0].status.status_code == StatusCode.ERROR
        assert attempt_spans[1].status.status_code == StatusCode.ERROR
        assert attempt_spans[2].status.status_code == StatusCode.OK

        # parent execute OK.
        exec_span = exporter.span_by_name(SPAN_TOOL_EXECUTE)
        assert exec_span.status.status_code == StatusCode.OK
        assert exec_span.attributes[ATTR_TOOL_RESULT_SUCCESS] is True
        assert exec_span.attributes[ATTR_TOOL_ATTEMPT_COUNT] == 3

    def test_retry_events_count(self):
        from tools.models import RetryPolicy
        tracer, exporter = make_recording_tracer()
        handler = ScriptedTool([
            ("transient", "fail 1"),
            ("transient", "fail 2"),
            ("success", "ok"),
        ])
        sleeper = FakeSleeper()
        spec = _echo_spec(
            retry_policy=RetryPolicy(
                max_attempts=3,
                initial_backoff_seconds=1.0,
                backoff_multiplier=2.0,
            ),
        )
        runtime, _, _ = _make_traced_runtime(
            tracer, spec=spec, handler=handler, sleep=sleeper
        )

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        exec_span = exporter.span_by_name(SPAN_TOOL_EXECUTE)
        retry_events = [e for e in exec_span.events if e.name == EVENT_RETRY_SCHEDULED]
        assert len(retry_events) == 2

    def test_retry_event_attributes(self):
        from tools.models import RetryPolicy
        tracer, exporter = make_recording_tracer()
        handler = ScriptedTool([
            ("transient", "fail 1"),
            ("success", "ok"),
        ])
        sleeper = FakeSleeper()
        spec = _echo_spec(
            retry_policy=RetryPolicy(
                max_attempts=2,
                initial_backoff_seconds=1.0,
                backoff_multiplier=2.0,
            ),
        )
        runtime, _, _ = _make_traced_runtime(
            tracer, spec=spec, handler=handler, sleep=sleeper
        )

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        exec_span = exporter.span_by_name(SPAN_TOOL_EXECUTE)
        retry_events = [e for e in exec_span.events if e.name == EVENT_RETRY_SCHEDULED]
        assert len(retry_events) == 1
        ev = retry_events[0]
        assert ev.attributes[ATTR_TOOL_ATTEMPT_NUMBER] == 1
        assert ev.attributes[ATTR_TOOL_RETRY_NEXT_ATTEMPT] == 2
        assert ev.attributes[ATTR_TOOL_ERROR_TYPE] == "transient"


# ---------------------------------------------------------------------------
# Section 59 — Retry Backoff Attributes
# ---------------------------------------------------------------------------


class TestRetryBackoffAttributes:
    def test_backoff_matches_sleeper(self):
        from tools.models import RetryPolicy
        tracer, exporter = make_recording_tracer()
        handler = ScriptedTool([
            ("transient", "fail 1"),
            ("transient", "fail 2"),
            ("success", "ok"),
        ])
        sleeper = FakeSleeper()
        spec = _echo_spec(
            retry_policy=RetryPolicy(
                max_attempts=3,
                initial_backoff_seconds=1.0,
                backoff_multiplier=2.0,
            ),
        )
        runtime, _, _ = _make_traced_runtime(
            tracer, spec=spec, handler=handler, sleep=sleeper
        )

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        exec_span = exporter.span_by_name(SPAN_TOOL_EXECUTE)
        retry_events = [e for e in exec_span.events if e.name == EVENT_RETRY_SCHEDULED]
        # backoff: retry 1 -> 1.0, retry 2 -> 2.0
        assert retry_events[0].attributes[ATTR_TOOL_RETRY_BACKOFF_SECONDS] == 1.0
        assert retry_events[1].attributes[ATTR_TOOL_RETRY_BACKOFF_SECONDS] == 2.0
        # Match the sleeper's recorded delays.
        assert sleeper.delays == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Section 60 — Side Effect Retry Block
# ---------------------------------------------------------------------------


class TestSideEffectRetryBlock:
    def test_side_effecting_no_retry_event(self):
        from tools.models import RetryPolicy
        tracer, exporter = make_recording_tracer()
        handler = ScriptedTool([("timeout",)])
        sleeper = FakeSleeper()
        spec = _echo_spec(
            side_effect=ToolSideEffect.SIDE_EFFECTING,
            retry_policy=RetryPolicy(max_attempts=3),
        )
        runtime, _, _ = _make_traced_runtime(
            tracer, spec=spec, handler=handler, sleep=sleeper
        )

        result = _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        # Only 1 attempt despite max_attempts=3.
        assert result.attempt_count == 1
        assert result.success is False
        assert result.error.error_type == ToolErrorType.TIMEOUT

        # No retry events.
        exec_span = exporter.span_by_name(SPAN_TOOL_EXECUTE)
        retry_events = [e for e in exec_span.events if e.name == EVENT_RETRY_SCHEDULED]
        assert len(retry_events) == 0

    def test_side_effecting_one_attempt_span(self):
        from tools.models import RetryPolicy
        tracer, exporter = make_recording_tracer()
        handler = ScriptedTool([("timeout",)])
        sleeper = FakeSleeper()
        spec = _echo_spec(
            side_effect=ToolSideEffect.SIDE_EFFECTING,
            retry_policy=RetryPolicy(max_attempts=3),
        )
        runtime, _, _ = _make_traced_runtime(
            tracer, spec=spec, handler=handler, sleep=sleeper
        )

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        attempt_spans = exporter.spans_named(SPAN_TOOL_ATTEMPT)
        assert len(attempt_spans) == 1


# ---------------------------------------------------------------------------
# Section 61 — Timeout
# ---------------------------------------------------------------------------


class TestTimeoutTracing:
    def test_timeout_attempt_error(self):
        tracer, exporter = make_recording_tracer()
        handler = SlowTool(delay=5.0)
        spec = _echo_spec(timeout_seconds=0.1)
        runtime, _, _ = _make_traced_runtime(tracer, spec=spec, handler=handler)

        result = _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        assert result.success is False
        assert result.error.error_type == ToolErrorType.TIMEOUT

        attempt_spans = exporter.spans_named(SPAN_TOOL_ATTEMPT)
        assert len(attempt_spans) == 1
        assert attempt_spans[0].status.status_code == StatusCode.ERROR
        assert attempt_spans[0].attributes[ATTR_TOOL_ERROR_TYPE] == "timeout"

    def test_timeout_parent_error(self):
        tracer, exporter = make_recording_tracer()
        handler = SlowTool(delay=5.0)
        spec = _echo_spec(timeout_seconds=0.1)
        runtime, _, _ = _make_traced_runtime(tracer, spec=spec, handler=handler)

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        exec_span = exporter.span_by_name(SPAN_TOOL_EXECUTE)
        assert exec_span.status.status_code == StatusCode.ERROR
        assert exec_span.attributes[ATTR_TOOL_ERROR_TYPE] == "timeout"


# ---------------------------------------------------------------------------
# Section 62 — Ordinary Handler Exception
# ---------------------------------------------------------------------------


class TestHandlerExceptionTracing:
    def test_exception_recorded_on_attempt(self):
        tracer, exporter = make_recording_tracer()
        handler = RaisingTool(RuntimeError("boom"))
        runtime, _, _ = _make_traced_runtime(tracer, handler=handler)

        result = _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        assert result.success is False
        assert result.error.error_type == ToolErrorType.EXECUTION

        attempt_spans = exporter.spans_named(SPAN_TOOL_ATTEMPT)
        assert len(attempt_spans) == 1
        # Default policy records exception type as attribute, not as event.
        from observability import ATTR_ERROR_EXCEPTION_TYPE
        assert attempt_spans[0].attributes[ATTR_ERROR_EXCEPTION_TYPE] == "RuntimeError"

    def test_exception_attempt_error_status(self):
        tracer, exporter = make_recording_tracer()
        handler = RaisingTool(RuntimeError("boom"))
        runtime, _, _ = _make_traced_runtime(tracer, handler=handler)

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        attempt_spans = exporter.spans_named(SPAN_TOOL_ATTEMPT)
        assert attempt_spans[0].status.status_code == StatusCode.ERROR

    def test_exception_parent_error_status(self):
        tracer, exporter = make_recording_tracer()
        handler = RaisingTool(RuntimeError("boom"))
        runtime, _, _ = _make_traced_runtime(tracer, handler=handler)

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        exec_span = exporter.span_by_name(SPAN_TOOL_EXECUTE)
        assert exec_span.status.status_code == StatusCode.ERROR

    def test_business_result_still_returned(self):
        tracer, exporter = make_recording_tracer()
        handler = RaisingTool(RuntimeError("boom"))
        runtime, _, _ = _make_traced_runtime(tracer, handler=handler)

        result = _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        # Runtime returns structured result, does not propagate exception.
        assert isinstance(result, type(result))
        assert result.success is False


# ---------------------------------------------------------------------------
# Section 40 — Tool CancelledError
# ---------------------------------------------------------------------------


class TestCancellationTracing:
    def test_cancelled_error_propagates(self):
        tracer, exporter = make_recording_tracer()
        handler = CooperativeCancelTool(delay=5.0)
        spec = _echo_spec(timeout_seconds=10.0)
        runtime, _, _ = _make_traced_runtime(tracer, spec=spec, handler=handler)

        async def _do():
            task = asyncio.create_task(
                runtime.execute(
                    ToolCall(tool_name="echo", arguments={"text": "x"}),
                    ToolExecutionContext(),
                )
            )
            await asyncio.sleep(0.1)
            task.cancel()
            await task

        with pytest.raises(asyncio.CancelledError):
            _run(_do())

    def test_cancelled_span_unset_status(self):
        tracer, exporter = make_recording_tracer()
        handler = CooperativeCancelTool(delay=5.0)
        spec = _echo_spec(timeout_seconds=10.0)
        runtime, _, _ = _make_traced_runtime(tracer, spec=spec, handler=handler)

        async def _do():
            task = asyncio.create_task(
                runtime.execute(
                    ToolCall(tool_name="echo", arguments={"text": "x"}),
                    ToolExecutionContext(),
                )
            )
            await asyncio.sleep(0.1)
            task.cancel()
            await task

        with pytest.raises(asyncio.CancelledError):
            _run(_do())

        # The execute span may or may not be finished depending on timing,
        # but if it exists, status should be UNSET.
        exec_spans = exporter.spans_named(SPAN_TOOL_EXECUTE)
        if exec_spans:
            assert exec_spans[0].status.status_code == StatusCode.UNSET


# ---------------------------------------------------------------------------
# Section 63 — MCP Source Attributes
# ---------------------------------------------------------------------------


class TestSourceAttributes:
    def test_mcp_source_attributes(self):
        tracer, exporter = make_recording_tracer()
        handler = EchoTool()
        spec = _echo_spec(
            source=ToolSource.MCP,
            provider_id="demo",
        )
        runtime, _, _ = _make_traced_runtime(tracer, spec=spec, handler=handler)

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        exec_span = exporter.span_by_name(SPAN_TOOL_EXECUTE)
        assert exec_span.attributes[ATTR_TOOL_SOURCE] == "mcp"
        assert exec_span.attributes[ATTR_TOOL_PROVIDER_ID] == "demo"

    def test_local_source_attributes(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        exec_span = exporter.span_by_name(SPAN_TOOL_EXECUTE)
        assert exec_span.attributes[ATTR_TOOL_SOURCE] == "local"
        # Local tools should not set provider_id.
        assert ATTR_TOOL_PROVIDER_ID not in exec_span.attributes

    def test_side_effect_attribute(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        exec_span = exporter.span_by_name(SPAN_TOOL_EXECUTE)
        assert exec_span.attributes[ATTR_TOOL_SIDE_EFFECT] == "read_only"


# ---------------------------------------------------------------------------
# Section 64 — No Sensitive Output
# ---------------------------------------------------------------------------


class TestNoSensitiveData:
    def test_tool_output_not_in_telemetry(self):
        tracer, exporter = make_recording_tracer()
        handler = EchoTool()
        runtime, _, _ = _make_traced_runtime(tracer, handler=handler)

        _run(
            runtime.execute(
                ToolCall(
                    tool_name="echo",
                    arguments={"text": "VERY_SECRET_OUTPUT_MARKER"},
                ),
                ToolExecutionContext(),
            )
        )

        for s in exporter.finished_spans:
            for key, val in (s.attributes or {}).items():
                assert "VERY_SECRET_OUTPUT_MARKER" not in str(val)
            for ev in s.events:
                for key, val in (ev.attributes or {}).items():
                    assert "VERY_SECRET_OUTPUT_MARKER" not in str(val)

    def test_tool_arguments_not_in_telemetry(self):
        tracer, exporter = make_recording_tracer()
        handler = EchoTool()
        runtime, _, _ = _make_traced_runtime(tracer, handler=handler)

        _run(
            runtime.execute(
                ToolCall(
                    tool_name="echo",
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


# ---------------------------------------------------------------------------
# Section 67 — Deterministic Span Names
# ---------------------------------------------------------------------------


class TestDeterministicSpanNames:
    def test_span_names_are_fixed(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        all_names = {s.name for s in exporter.finished_spans}
        assert all_names <= {SPAN_TOOL_EXECUTE, SPAN_TOOL_ATTEMPT}

    def test_no_tool_name_in_span_name(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer, name="unique_tool_xyz")

        _run(
            runtime.execute(
                ToolCall(tool_name="unique_tool_xyz", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        for s in exporter.finished_spans:
            assert "unique_tool_xyz" not in s.name


# ---------------------------------------------------------------------------
# Section 68 — Tracer Injection Isolation
# ---------------------------------------------------------------------------


class TestTracerInjectionIsolation:
    def test_two_runtimes_do_not_cross_contaminate(self):
        tracer_a, exporter_a = make_recording_tracer()
        tracer_b, exporter_b = make_recording_tracer()

        runtime_a, _, _ = _make_traced_runtime(tracer_a, name="tool_a")
        runtime_b, _, _ = _make_traced_runtime(tracer_b, name="tool_b")

        _run(
            runtime_a.execute(
                ToolCall(tool_name="tool_a", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )
        _run(
            runtime_b.execute(
                ToolCall(tool_name="tool_b", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        # Exporter A only has tool_a.
        exec_a = exporter_a.spans_named(SPAN_TOOL_EXECUTE)
        assert len(exec_a) == 1
        assert exec_a[0].attributes[ATTR_TOOL_NAME] == "tool_a"

        # Exporter B only has tool_b.
        exec_b = exporter_b.spans_named(SPAN_TOOL_EXECUTE)
        assert len(exec_b) == 1
        assert exec_b[0].attributes[ATTR_TOOL_NAME] == "tool_b"


# ---------------------------------------------------------------------------
# Section 50 — No SDK Configured
# ---------------------------------------------------------------------------


class TestNoSDKConfigured:
    def test_business_behavior_unchanged_without_tracer(self):
        registry = ToolRegistry()
        handler = EchoTool()
        registry.register(_echo_spec(), handler)
        runtime = ToolRuntime(registry=registry)

        result = _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "hello"}),
                ToolExecutionContext(),
            )
        )

        assert result.success is True
        assert result.output == "hello"
        assert result.attempt_count == 1
        assert handler.call_count == 1

    def test_retry_behavior_unchanged_without_tracer(self):
        from tools.models import RetryPolicy
        registry = ToolRegistry()
        handler = ScriptedTool([
            ("transient", "fail 1"),
            ("success", "ok"),
        ])
        sleeper = FakeSleeper()
        spec = _echo_spec(
            retry_policy=RetryPolicy(max_attempts=2, initial_backoff_seconds=1.0),
        )
        registry.register(spec, handler)
        runtime = ToolRuntime(registry=registry, sleep=sleeper)

        result = _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        assert result.success is True
        assert result.attempt_count == 2
        assert handler.call_count == 2
        assert sleeper.delays == [1.0]


# ---------------------------------------------------------------------------
# Section 38 — Transient attempt ERROR while parent succeeds
# ---------------------------------------------------------------------------


class TestAttemptErrorParentSuccess:
    def test_transient_attempt_error_parent_ok(self):
        from tools.models import RetryPolicy
        tracer, exporter = make_recording_tracer()
        handler = ScriptedTool([
            ("transient", "fail"),
            ("success", "ok"),
        ])
        sleeper = FakeSleeper()
        spec = _echo_spec(
            retry_policy=RetryPolicy(max_attempts=2, initial_backoff_seconds=0.1),
        )
        runtime, _, _ = _make_traced_runtime(
            tracer, spec=spec, handler=handler, sleep=sleeper
        )

        result = _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        assert result.success is True

        attempt_spans = exporter.spans_named(SPAN_TOOL_ATTEMPT)
        assert attempt_spans[0].status.status_code == StatusCode.ERROR
        assert attempt_spans[1].status.status_code == StatusCode.OK

        exec_span = exporter.span_by_name(SPAN_TOOL_EXECUTE)
        assert exec_span.status.status_code == StatusCode.OK

    def test_no_error_type_on_success(self):
        from tools.models import RetryPolicy
        tracer, exporter = make_recording_tracer()
        handler = ScriptedTool([
            ("transient", "fail"),
            ("success", "ok"),
        ])
        sleeper = FakeSleeper()
        spec = _echo_spec(
            retry_policy=RetryPolicy(max_attempts=2, initial_backoff_seconds=0.1),
        )
        runtime, _, _ = _make_traced_runtime(
            tracer, spec=spec, handler=handler, sleep=sleeper
        )

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        exec_span = exporter.span_by_name(SPAN_TOOL_EXECUTE)
        # Success should not leave error.type attribute.
        assert ATTR_TOOL_ERROR_TYPE not in exec_span.attributes
