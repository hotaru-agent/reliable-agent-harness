"""Exception privacy regression tests for Phase 6 Step 2.

Verifies that the default ExceptionTelemetryPolicy does NOT leak
exception messages or stacktraces into telemetry. Only the exception
type/class identity is recorded.

Tests cover:
* Harness step exception secret marker
* Tool handler exception secret marker
* Agent/controller exception secret marker
* Context error marker
* MCP client exception secret marker

All offline, deterministic, no network, no collector.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pytest

from harness.action_controller import (
    ActionControllerState,
    AgentActionController,
)
from harness.context import (
    ContextAssembler,
    ContextBudget,
    ContextBudgetExceededError,
    ContextItem,
    ContextKind,
)
from harness.loop_detection import (
    ActionHistory,
    LoopDetector,
    LoopDetectorConfig,
)
from mcp_adapter import MCPToolProvider
from observability import ATTR_ERROR_EXCEPTION_TYPE
from storage import InMemoryCheckpointStore
from tests.fakes import FakeClock, FakeStepExecutor, make_id_factory
from tests.fakes_telemetry import make_recording_tracer
from tests.fake_tools import FakeSleeper, RaisingTool, SuccessTool
from tools import (
    ToolCall,
    ToolExecutionContext,
    ToolRegistry,
    ToolRuntime,
    ToolSideEffect,
    ToolSpec,
)


T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_HARNESS_SECRET = "HARNESS_SECRET_MARKER"
_TOOL_SECRET = "TOOL_SECRET_MARKER"
_AGENT_SECRET = "AGENT_SECRET_MARKER"
_CONTEXT_SECRET = "CONTEXT_SECRET_MARKER"
_MCP_SECRET = "MCP_SECRET_MARKER"


def _run(coro):
    return asyncio.run(coro)


def _scan_for_marker(exporter, marker: str) -> bool:
    """Scan all finished spans for the marker in attributes, events,
    event attributes, and status descriptions."""
    for s in exporter.finished_spans:
        # Check span attributes.
        for key, val in (s.attributes or {}).items():
            if marker in str(val):
                return True
        # Check span status description.
        if s.status and s.status.description:
            if marker in s.status.description:
                return True
        # Check events.
        for ev in s.events:
            if marker in ev.name:
                return True
            for key, val in (ev.attributes or {}).items():
                if marker in str(val):
                    return True
    return False


# ---------------------------------------------------------------------------
# Section 10 — Harness Exception Test
# ---------------------------------------------------------------------------


class TestHarnessExceptionPrivacy:
    def test_harness_exception_message_not_in_telemetry(self):
        tracer, exporter = make_recording_tracer()
        store = InMemoryCheckpointStore()
        executor = FakeStepExecutor(
            complete_at_step=2,
            fail_at_step=1,
            fail_exc=RuntimeError(f"token={_HARNESS_SECRET}"),
        )
        clock = FakeClock(T0)
        from harness.runtime import HarnessRuntime
        from harness.state import Task
        runtime = HarnessRuntime(
            checkpoint_store=store,
            step_executor=executor,
            clock=clock,
            run_id_factory=make_id_factory("R"),
            checkpoint_id_factory=make_id_factory("C"),
            tracer=tracer,
        )
        task = Task(task_id="T-PRIV", goal="privacy test", created_at=T0)

        with pytest.raises(RuntimeError):
            _run(runtime.start(task))

        assert not _scan_for_marker(exporter, _HARNESS_SECRET)

    def test_harness_exception_type_still_recorded(self):
        tracer, exporter = make_recording_tracer()
        store = InMemoryCheckpointStore()
        executor = FakeStepExecutor(
            complete_at_step=2,
            fail_at_step=1,
            fail_exc=RuntimeError(f"token={_HARNESS_SECRET}"),
        )
        clock = FakeClock(T0)
        from harness.runtime import HarnessRuntime
        from harness.state import Task
        runtime = HarnessRuntime(
            checkpoint_store=store,
            step_executor=executor,
            clock=clock,
            run_id_factory=make_id_factory("R"),
            checkpoint_id_factory=make_id_factory("C"),
            tracer=tracer,
        )
        task = Task(task_id="T-PRIV", goal="privacy test", created_at=T0)

        with pytest.raises(RuntimeError):
            _run(runtime.start(task))

        # Exception type should still be recorded as a safe attribute.
        run_span = exporter.span_by_name("harness.run")
        assert run_span.attributes[ATTR_ERROR_EXCEPTION_TYPE] == "RuntimeError"


# ---------------------------------------------------------------------------
# Section 9 — Tool Handler Exception Test
# ---------------------------------------------------------------------------


class TestToolExceptionPrivacy:
    def test_tool_exception_message_not_in_telemetry(self):
        tracer, exporter = make_recording_tracer()
        registry = ToolRegistry()
        handler = RaisingTool(RuntimeError(f"password={_TOOL_SECRET}"))
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
        runtime = ToolRuntime(registry=registry, tracer=tracer)

        result = _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        assert result.success is False
        assert not _scan_for_marker(exporter, _TOOL_SECRET)

    def test_tool_exception_type_still_recorded(self):
        tracer, exporter = make_recording_tracer()
        registry = ToolRegistry()
        handler = RaisingTool(RuntimeError(f"password={_TOOL_SECRET}"))
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
        runtime = ToolRuntime(registry=registry, tracer=tracer)

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        attempt_spans = exporter.spans_named("tool.attempt")
        assert len(attempt_spans) == 1
        assert attempt_spans[0].attributes[ATTR_ERROR_EXCEPTION_TYPE] == "RuntimeError"


# ---------------------------------------------------------------------------
# Section 60 — Agent/Controller Exception Privacy
# ---------------------------------------------------------------------------


class _RaisingProgressProvider:
    """Progress provider that raises an exception with a secret marker."""

    def __init__(self):
        self.call_count = 0

    def get_progress_token(self, call, result) -> str | None:
        self.call_count += 1
        raise RuntimeError(f"agent_error={_AGENT_SECRET}")


class TestAgentExceptionPrivacy:
    def test_agent_exception_message_not_in_telemetry(self):
        tracer, exporter = make_recording_tracer()
        registry = ToolRegistry()
        spec = ToolSpec(
            name="tool",
            description="test",
            input_schema={"type": "object"},
            required_permissions=frozenset(),
            timeout_seconds=1.0,
            side_effect=ToolSideEffect.READ_ONLY,
        )
        registry.register(spec, SuccessTool("ok"))
        runtime = ToolRuntime(registry=registry, tracer=tracer, sleep=FakeSleeper())
        detector = LoopDetector(
            config=LoopDetectorConfig(), history=ActionHistory(max_size=64)
        )
        controller = AgentActionController(
            tool_runtime=runtime,
            loop_detector=detector,
            progress_provider=_RaisingProgressProvider(),
            tracer=tracer,
        )

        with pytest.raises(RuntimeError):
            _run(
                controller.execute(
                    ToolCall(tool_name="tool", arguments={}),
                    ToolExecutionContext(),
                )
            )

        assert not _scan_for_marker(exporter, _AGENT_SECRET)


# ---------------------------------------------------------------------------
# Section 60 — Context Error Privacy
# ---------------------------------------------------------------------------


class TestContextExceptionPrivacy:
    def test_context_overflow_message_not_in_telemetry(self):
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        items = [
            ContextItem(
                item_id="critical",
                kind=ContextKind.CRITICAL_STATE,
                content=f"secret={_CONTEXT_SECRET}",
                estimated_tokens=100,
                sequence_index=0,
                must_keep=True,
            ),
        ]

        with pytest.raises(ContextBudgetExceededError):
            assembler.assemble(items, ContextBudget(max_tokens=10))

        assert not _scan_for_marker(exporter, _CONTEXT_SECRET)

    def test_context_exception_type_still_recorded(self):
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        items = [
            ContextItem(
                item_id="critical",
                kind=ContextKind.CRITICAL_STATE,
                content="x",
                estimated_tokens=100,
                sequence_index=0,
                must_keep=True,
            ),
        ]

        with pytest.raises(ContextBudgetExceededError):
            assembler.assemble(items, ContextBudget(max_tokens=10))

        span = exporter.span_by_name("context.assemble")
        assert span.attributes[ATTR_ERROR_EXCEPTION_TYPE] == "ContextBudgetExceededError"


# ---------------------------------------------------------------------------
# Section 60 — MCP Client Exception Privacy
# ---------------------------------------------------------------------------


@dataclass
class _FakeTool:
    name: str = "bad"
    description: Optional[str] = None
    input_schema: dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    })
    annotations: Any = None


@dataclass
class _FakeListToolsResult:
    tools: list = field(default_factory=list)
    next_cursor: Optional[str] = None


class _FailingMCPClient:
    """MCP client that raises an exception with a secret marker."""

    async def list_tools(self, *, cursor=None):
        raise RuntimeError(f"mcp_error={_MCP_SECRET}")

    async def call_tool(self, name, arguments=None):
        raise NotImplementedError


class TestMCPExceptionPrivacy:
    def test_mcp_exception_message_not_in_telemetry(self):
        tracer, exporter = make_recording_tracer()
        provider = MCPToolProvider(
            provider_id="demo",
            client=_FailingMCPClient(),
            tracer=tracer,
        )
        registry = ToolRegistry()

        with pytest.raises(RuntimeError):
            _run(provider.discover_into(registry))

        assert not _scan_for_marker(exporter, _MCP_SECRET)

    def test_mcp_exception_type_still_recorded(self):
        tracer, exporter = make_recording_tracer()
        provider = MCPToolProvider(
            provider_id="demo",
            client=_FailingMCPClient(),
            tracer=tracer,
        )
        registry = ToolRegistry()

        with pytest.raises(RuntimeError):
            _run(provider.discover_into(registry))

        span = exporter.span_by_name("mcp.discover")
        assert span.attributes[ATTR_ERROR_EXCEPTION_TYPE] == "RuntimeError"


# ---------------------------------------------------------------------------
# Section 61 — No Secret in Status Description
# ---------------------------------------------------------------------------


class TestStatusDescriptionPrivacy:
    def test_no_secret_in_status_description(self):
        tracer, exporter = make_recording_tracer()
        registry = ToolRegistry()
        handler = RaisingTool(RuntimeError(f"password={_TOOL_SECRET}"))
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
        runtime = ToolRuntime(registry=registry, tracer=tracer)

        _run(
            runtime.execute(
                ToolCall(tool_name="echo", arguments={"text": "x"}),
                ToolExecutionContext(),
            )
        )

        for s in exporter.finished_spans:
            if s.status and s.status.description:
                assert _TOOL_SECRET not in s.status.description
