"""Tests for the single-attempt tool runtime (Phase 2 Step 1).

Covers every failure path through the unified runtime boundary:

* success (Section 36)
* missing tool (Section 37)
* missing required arg (Section 38)
* wrong type (Section 39)
* unknown arg with additionalProperties=False (Section 40)
* permission allowed (Section 41)
* permission denied (Section 42)
* timeout (Section 43)
* external cancellation != timeout (Section 44)
* ordinary handler exception (Section 45)
* transient reported failure (Section 46)
* permanent reported failure (Section 47)
* side-effect metadata (Section 48)
* structured result invariants (Section 49)

All offline, deterministic, no network, no LLM, no MCP.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.fake_tools import (
    CooperativeCancelTool,
    EchoTool,
    ExecutionReportedTool,
    PermanentFailureTool,
    RaisingTool,
    SlowTool,
    SuccessTool,
    TransientFailureTool,
)
from tools import (
    ToolCall,
    ToolErrorType,
    ToolExecutionContext,
    ToolExecutionResult,
    ToolRegistry,
    ToolRuntime,
    ToolSideEffect,
    ToolSpec,
)

# ---------------------------------------------------------------------------
# Shared schema / spec builders
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
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="echo the text argument",
        input_schema=_ECHO_SCHEMA,
        required_permissions=required_permissions,
        timeout_seconds=timeout_seconds,
        side_effect=side_effect,
    )


def _make_runtime(*, spec=None, handler=None, name="echo") -> tuple[ToolRuntime, ToolRegistry, object]:
    registry = ToolRegistry()
    handler = handler or EchoTool()
    spec = spec or _echo_spec(name)
    registry.register(spec, handler)
    return ToolRuntime(registry=registry), registry, handler


# ---------------------------------------------------------------------------
# Section 36 — success
# ---------------------------------------------------------------------------


def test_success():
    runtime, _, handler = _make_runtime()
    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="echo", arguments={"text": "hello"}),
            ToolExecutionContext(),
        )
    )
    assert result.success is True
    assert result.output == "hello"
    assert result.error is None
    assert result.tool_name == "echo"
    assert handler.call_count == 1


# ---------------------------------------------------------------------------
# Section 37 — missing tool
# ---------------------------------------------------------------------------


def test_missing_tool():
    registry = ToolRegistry()
    runtime = ToolRuntime(registry=registry)
    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="unknown_tool", arguments={}),
            ToolExecutionContext(),
        )
    )
    assert result.success is False
    assert result.error.error_type is ToolErrorType.NOT_FOUND
    assert result.error.retryable is False


# ---------------------------------------------------------------------------
# Section 38 — missing required arg
# ---------------------------------------------------------------------------


def test_missing_required_arg():
    runtime, _, handler = _make_runtime()
    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="echo", arguments={}),
            ToolExecutionContext(),
        )
    )
    assert result.success is False
    assert result.error.error_type is ToolErrorType.VALIDATION
    assert result.error.retryable is False
    assert handler.call_count == 0


# ---------------------------------------------------------------------------
# Section 39 — wrong type
# ---------------------------------------------------------------------------


def test_wrong_type():
    runtime, _, handler = _make_runtime()
    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="echo", arguments={"text": 123}),
            ToolExecutionContext(),
        )
    )
    assert result.success is False
    assert result.error.error_type is ToolErrorType.VALIDATION
    assert handler.call_count == 0


# ---------------------------------------------------------------------------
# Section 40 — unknown arg (additionalProperties=False)
# ---------------------------------------------------------------------------


def test_unknown_arg_rejected_when_additional_properties_false():
    runtime, _, handler = _make_runtime()
    result = asyncio.run(
        runtime.execute(
            ToolCall(
                tool_name="echo",
                arguments={"text": "hello", "unexpected": True},
            ),
            ToolExecutionContext(),
        )
    )
    assert result.success is False
    assert result.error.error_type is ToolErrorType.VALIDATION
    assert handler.call_count == 0


def test_unknown_arg_allowed_when_additional_properties_true():
    registry = ToolRegistry()
    handler = EchoTool()
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": True,
    }
    registry.register(
        ToolSpec(
            name="echo",
            description="echo",
            input_schema=schema,
            required_permissions=frozenset(),
            timeout_seconds=1.0,
            side_effect=ToolSideEffect.READ_ONLY,
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry)
    result = asyncio.run(
        runtime.execute(
            ToolCall(
                tool_name="echo",
                arguments={"text": "hello", "extra": 1},
            ),
            ToolExecutionContext(),
        )
    )
    assert result.success is True
    assert handler.call_count == 1


# ---------------------------------------------------------------------------
# Section 41 — permission allowed
# ---------------------------------------------------------------------------


def test_permission_allowed():
    runtime, _, handler = _make_runtime(
        spec=_echo_spec(required_permissions=frozenset({"repo.read"}))
    )
    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="echo", arguments={"text": "hi"}),
            ToolExecutionContext(granted_permissions=frozenset({"repo.read"})),
        )
    )
    assert result.success is True
    assert handler.call_count == 1


# ---------------------------------------------------------------------------
# Section 42 — permission denied
# ---------------------------------------------------------------------------


def test_permission_denied():
    runtime, _, handler = _make_runtime(
        spec=_echo_spec(required_permissions=frozenset({"repo.write"}))
    )
    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="echo", arguments={"text": "hi"}),
            ToolExecutionContext(granted_permissions=frozenset({"repo.read"})),
        )
    )
    assert result.success is False
    assert result.error.error_type is ToolErrorType.PERMISSION
    assert result.error.retryable is False
    assert handler.call_count == 0


# ---------------------------------------------------------------------------
# Section 43 — timeout
# ---------------------------------------------------------------------------


def test_timeout():
    registry = ToolRegistry()
    handler = SlowTool(delay=1.0)
    registry.register(
        ToolSpec(
            name="slow",
            description="slow tool",
            input_schema={"type": "object"},
            required_permissions=frozenset(),
            timeout_seconds=0.05,
            side_effect=ToolSideEffect.READ_ONLY,
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry)
    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="slow", arguments={}),
            ToolExecutionContext(),
        )
    )
    assert result.success is False
    assert result.error.error_type is ToolErrorType.TIMEOUT
    assert result.error.retryable is True
    # Handler was attempted exactly once; no retry.
    assert handler.call_count == 1


# ---------------------------------------------------------------------------
# Section 44 — external cancellation != timeout
# ---------------------------------------------------------------------------


def test_external_cancellation_propagates():
    registry = ToolRegistry()
    handler = CooperativeCancelTool(delay=5.0)
    registry.register(
        ToolSpec(
            name="coop",
            description="cooperative cancel tool",
            input_schema={"type": "object"},
            required_permissions=frozenset(),
            timeout_seconds=30.0,  # large so the runtime deadline does not fire
            side_effect=ToolSideEffect.READ_ONLY,
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry)

    async def _drive():
        task = asyncio.create_task(
            runtime.execute(
                ToolCall(tool_name="coop", arguments={}),
                ToolExecutionContext(),
            )
        )
        # Let the handler enter its sleep.
        await asyncio.sleep(0.05)
        task.cancel()
        return await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_drive())
    # Handler was entered (recorded) but the result is not a TIMEOUT result;
    # the cancellation propagated as CancelledError.
    assert handler.call_count == 1


# ---------------------------------------------------------------------------
# Section 45 — ordinary handler exception
# ---------------------------------------------------------------------------


def test_ordinary_handler_exception():
    registry = ToolRegistry()
    handler = RaisingTool(RuntimeError("boom"))
    registry.register(
        ToolSpec(
            name="boom",
            description="raising tool",
            input_schema={"type": "object"},
            required_permissions=frozenset(),
            timeout_seconds=1.0,
            side_effect=ToolSideEffect.READ_ONLY,
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry)
    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="boom", arguments={}),
            ToolExecutionContext(),
        )
    )
    assert result.success is False
    assert result.error.error_type is ToolErrorType.EXECUTION
    assert result.error.retryable is False
    assert handler.call_count == 1


# ---------------------------------------------------------------------------
# Section 46 — transient reported failure
# ---------------------------------------------------------------------------


def test_transient_reported_failure():
    registry = ToolRegistry()
    handler = TransientFailureTool()
    registry.register(
        ToolSpec(
            name="transient",
            description="transient failure tool",
            input_schema={"type": "object"},
            required_permissions=frozenset(),
            timeout_seconds=1.0,
            side_effect=ToolSideEffect.IDEMPOTENT,
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry)
    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="transient", arguments={}),
            ToolExecutionContext(),
        )
    )
    assert result.success is False
    assert result.error.error_type is ToolErrorType.TRANSIENT
    assert result.error.retryable is True
    # Single attempt — no retry in this phase.
    assert handler.call_count == 1


# ---------------------------------------------------------------------------
# Section 47 — permanent reported failure
# ---------------------------------------------------------------------------


def test_permanent_reported_failure():
    registry = ToolRegistry()
    handler = PermanentFailureTool()
    registry.register(
        ToolSpec(
            name="permanent",
            description="permanent failure tool",
            input_schema={"type": "object"},
            required_permissions=frozenset(),
            timeout_seconds=1.0,
            side_effect=ToolSideEffect.SIDE_EFFECTING,
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry)
    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="permanent", arguments={}),
            ToolExecutionContext(),
        )
    )
    assert result.success is False
    assert result.error.error_type is ToolErrorType.PERMANENT
    assert result.error.retryable is False
    assert handler.call_count == 1


# ---------------------------------------------------------------------------
# Section 48 — side-effect metadata is modelled but does not trigger retry
# ---------------------------------------------------------------------------


def test_side_effect_metadata_distinguishable():
    registry = ToolRegistry()
    for se in (
        ToolSideEffect.READ_ONLY,
        ToolSideEffect.IDEMPOTENT,
        ToolSideEffect.SIDE_EFFECTING,
    ):
        registry.register(
            ToolSpec(
                name=f"tool-{se.value}",
                description=f"tool with {se.value} side effect",
                input_schema={"type": "object"},
                required_permissions=frozenset(),
                timeout_seconds=1.0,
                side_effect=se,
            ),
            EchoTool(),
        )
    assert registry.get("tool-read_only").spec.side_effect is ToolSideEffect.READ_ONLY
    assert registry.get("tool-idempotent").spec.side_effect is ToolSideEffect.IDEMPOTENT
    assert (
        registry.get("tool-side_effecting").spec.side_effect
        is ToolSideEffect.SIDE_EFFECTING
    )


def test_retryable_true_does_not_trigger_retry():
    """A TRANSIENT failure (retryable=True) must still execute only once."""
    registry = ToolRegistry()
    handler = TransientFailureTool()
    registry.register(
        ToolSpec(
            name="transient",
            description="transient failure tool",
            input_schema={"type": "object"},
            required_permissions=frozenset(),
            timeout_seconds=1.0,
            side_effect=ToolSideEffect.IDEMPOTENT,
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry)
    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="transient", arguments={}),
            ToolExecutionContext(),
        )
    )
    assert result.error.retryable is True
    assert handler.call_count == 1  # no retry despite retryable=True


# ---------------------------------------------------------------------------
# Section 49 — structured result invariants
# ---------------------------------------------------------------------------


def test_success_result_cannot_carry_error():
    # success=True with error=None is legal.
    legal = ToolExecutionResult(
        tool_name="echo", success=True, output="x", error=None
    )
    assert legal.success is True
    # success=True with an error is illegal.
    with pytest.raises(ValueError):
        ToolExecutionResult(
            tool_name="echo",
            success=True,
            output="x",
            error=__import__("tools").ToolError(
                error_type=ToolErrorType.EXECUTION,
                message="x",
                retryable=False,
            ),
        )


def test_failure_result_must_carry_error():
    with pytest.raises(ValueError):
        ToolExecutionResult(tool_name="echo", success=False, error=None)


def test_success_with_none_output_is_legal():
    r = ToolExecutionResult(tool_name="echo", success=True, output=None, error=None)
    assert r.success is True
    assert r.output is None


# ---------------------------------------------------------------------------
# Extra — validation happens before permission (both before handler)
# ---------------------------------------------------------------------------


def test_validation_before_permission():
    """Bad args + missing permission -> VALIDATION wins (handler not called)."""
    runtime, _, handler = _make_runtime(
        spec=_echo_spec(required_permissions=frozenset({"repo.write"}))
    )
    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="echo", arguments={"text": 123}),  # wrong type
            ToolExecutionContext(granted_permissions=frozenset()),  # no perms
        )
    )
    assert result.error.error_type is ToolErrorType.VALIDATION
    assert handler.call_count == 0


# ---------------------------------------------------------------------------
# Extra — execution-reported failure via ToolReportedFailure(EXECUTION)
# ---------------------------------------------------------------------------


def test_execution_reported_failure():
    registry = ToolRegistry()
    handler = ExecutionReportedTool()
    registry.register(
        ToolSpec(
            name="exec-reported",
            description="execution reported tool",
            input_schema={"type": "object"},
            required_permissions=frozenset(),
            timeout_seconds=1.0,
            side_effect=ToolSideEffect.SIDE_EFFECTING,
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry)
    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="exec-reported", arguments={}),
            ToolExecutionContext(),
        )
    )
    assert result.success is False
    assert result.error.error_type is ToolErrorType.EXECUTION
    assert result.error.retryable is False
    assert handler.call_count == 1


# ---------------------------------------------------------------------------
# Extra — nested object / array validation
# ---------------------------------------------------------------------------


def test_nested_object_and_array_validation():
    registry = ToolRegistry()
    handler = SuccessTool("ok")
    schema = {
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["config"],
        "additionalProperties": False,
    }
    registry.register(
        ToolSpec(
            name="nested",
            description="nested schema tool",
            input_schema=schema,
            required_permissions=frozenset(),
            timeout_seconds=1.0,
            side_effect=ToolSideEffect.READ_ONLY,
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry)

    # Valid.
    ok = asyncio.run(
        runtime.execute(
            ToolCall(
                tool_name="nested",
                arguments={"config": {"name": "a"}, "tags": ["x", "y"]},
            ),
            ToolExecutionContext(),
        )
    )
    assert ok.success is True

    # Nested missing required.
    bad = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="nested", arguments={"config": {}}),
            ToolExecutionContext(),
        )
    )
    assert bad.error.error_type is ToolErrorType.VALIDATION

    # Array wrong element type.
    bad2 = asyncio.run(
        runtime.execute(
            ToolCall(
                tool_name="nested",
                arguments={"config": {"name": "a"}, "tags": [1, 2]},
            ),
            ToolExecutionContext(),
        )
    )
    assert bad2.error.error_type is ToolErrorType.VALIDATION


# ---------------------------------------------------------------------------
# Extra — integer vs bool disambiguation
# ---------------------------------------------------------------------------


def test_bool_not_accepted_as_integer():
    registry = ToolRegistry()
    handler = EchoTool()
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "additionalProperties": False,
    }
    registry.register(
        ToolSpec(
            name="inttool",
            description="integer arg tool",
            input_schema=schema,
            required_permissions=frozenset(),
            timeout_seconds=1.0,
            side_effect=ToolSideEffect.READ_ONLY,
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry)
    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="inttool", arguments={"count": True}),
            ToolExecutionContext(),
        )
    )
    assert result.success is False
    assert result.error.error_type is ToolErrorType.VALIDATION
