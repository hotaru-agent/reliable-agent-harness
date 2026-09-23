"""Phase 6 Step 2 smoke test — Full Harness Component Tracing.

Runs end-to-end action traces through AgentActionController → ToolRuntime,
loop/replan scenarios, context assembly, artifact externalization, and
MCP discovery, all under a recording tracer, then asserts the combined
trace structure.

All offline, deterministic, no network, no collector, no real LLM.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest
from opentelemetry.trace.status import Status, StatusCode

from harness.action_controller import (
    ActionControllerState,
    AgentActionController,
    ReplanRequiredError,
)
from harness.context import (
    ContextAssembler,
    ContextBudget,
    ContextItem,
    ContextKind,
    ContextPriority,
)
from harness.loop_detection import (
    ActionHistory,
    LoopDetector,
    LoopDetectorConfig,
)
from harness.output_externalization import (
    DeterministicToolOutputRenderer,
    OutputExternalizationPolicy,
    ToolOutputProcessor,
)
from mcp_adapter import MCPToolProvider
from observability import (
    ATTR_AGENT_ACTION_INDEX,
    ATTR_AGENT_ACTION_LOOP_DETECTED,
    ATTR_AGENT_ACTION_SUCCESS,
    ATTR_ARTIFACT_EXTERNALIZED,
    ATTR_ARTIFACT_ID,
    ATTR_CONTEXT_INCLUDED_COUNT,
    ATTR_CONTEXT_TOKEN_MAX,
    ATTR_CONTEXT_TOKEN_USED,
    ATTR_MCP_DISCOVERY_TOOL_COUNT,
    ATTR_MCP_PROVIDER_ID,
    ATTR_TOOL_ATTEMPT_COUNT,
    ATTR_TOOL_NAME,
    ATTR_TOOL_RESULT_SUCCESS,
    EVENT_AGENT_LOOP_DETECTED,
    EVENT_AGENT_REPLAN_BLOCKED,
    EVENT_AGENT_REPLAN_REQUIRED,
    EVENT_RETRY_SCHEDULED,
    SPAN_AGENT_ACTION,
    SPAN_AGENT_REPLAN_ACKNOWLEDGE,
    SPAN_ARTIFACT_PROCESS_OUTPUT,
    SPAN_CONTEXT_ASSEMBLE,
    SPAN_MCP_DISCOVER,
    SPAN_TOOL_ATTEMPT,
    SPAN_TOOL_EXECUTE,
)
from tests.fakes_telemetry import make_recording_tracer
from tests.fake_tools import FakeSleeper, ScriptedTool, SuccessTool
from tools import (
    RetryPolicy,
    ToolCall,
    ToolExecutionContext,
    ToolRegistry,
    ToolRuntime,
    ToolSideEffect,
    ToolSpec,
)
from storage import InMemoryArtifactStore


def _run(coro):
    return asyncio.run(coro)


class _ConstantProgressProvider:
    def __init__(self, token: str | None) -> None:
        self.token = token
        self.call_count = 0

    def get_progress_token(self, call, result) -> str | None:
        self.call_count += 1
        return self.token


def _spec(
    name: str = "tool",
    *,
    retry_policy: RetryPolicy | None = None,
    side_effect: ToolSideEffect = ToolSideEffect.READ_ONLY,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="test tool",
        input_schema={"type": "object"},
        required_permissions=frozenset(),
        timeout_seconds=1.0,
        side_effect=side_effect,
        retry_policy=retry_policy or RetryPolicy(),
    )


# ---------------------------------------------------------------------------
# Fake MCP types for discovery smoke test
# ---------------------------------------------------------------------------


@dataclass
class _FakeTool:
    name: str
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


class _FakeMCPClient:
    def __init__(self, tools):
        self._tools = tools

    async def list_tools(self, *, cursor=None):
        return _FakeListToolsResult(tools=self._tools)

    async def call_tool(self, name, arguments=None):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Section 46 — End-to-End Action Trace
# ---------------------------------------------------------------------------


class TestEndToEndActionTrace:
    def test_action_to_tool_to_attempt_parenting(self):
        """agent.action → tool.execute → tool.attempt in one trace."""
        tracer, exporter = make_recording_tracer()
        registry = ToolRegistry()
        spec = _spec()
        registry.register(spec, SuccessTool("ok"))
        runtime = ToolRuntime(registry=registry, tracer=tracer, sleep=FakeSleeper())
        detector = LoopDetector(
            config=LoopDetectorConfig(), history=ActionHistory(max_size=64)
        )
        controller = AgentActionController(
            tool_runtime=runtime,
            loop_detector=detector,
            progress_provider=_ConstantProgressProvider("s1"),
            tracer=tracer,
        )

        result = _run(
            controller.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )

        assert result.tool_result.success is True

        action_span = exporter.span_by_name(SPAN_AGENT_ACTION)
        exec_spans = exporter.spans_named(SPAN_TOOL_EXECUTE)
        attempt_spans = exporter.spans_named(SPAN_TOOL_ATTEMPT)

        assert len(exec_spans) == 1
        assert len(attempt_spans) == 1

        # Parenting: action → execute → attempt
        assert exec_spans[0].parent.span_id == action_span.context.span_id
        assert attempt_spans[0].parent.span_id == exec_spans[0].context.span_id

        # Same trace_id
        assert action_span.context.trace_id == exec_spans[0].context.trace_id
        assert action_span.context.trace_id == attempt_spans[0].context.trace_id

        # All OK
        assert action_span.status.status_code == StatusCode.OK
        assert exec_spans[0].status.status_code == StatusCode.OK
        assert attempt_spans[0].status.status_code == StatusCode.OK


# ---------------------------------------------------------------------------
# Section 47 — Retry End-to-End Action Trace
# ---------------------------------------------------------------------------


class TestRetryEndToEndActionTrace:
    def test_retry_then_success_trace(self):
        """agent.action OK → tool.execute OK → attempt 1 ERROR, attempt 2 OK."""
        tracer, exporter = make_recording_tracer()
        registry = ToolRegistry()
        handler = ScriptedTool([("transient", "fail"), ("success", "ok")])
        spec = _spec(retry_policy=RetryPolicy(max_attempts=2, initial_backoff_seconds=0.1))
        registry.register(spec, handler)
        runtime = ToolRuntime(registry=registry, tracer=tracer, sleep=FakeSleeper())
        detector = LoopDetector(
            config=LoopDetectorConfig(), history=ActionHistory(max_size=64)
        )
        controller = AgentActionController(
            tool_runtime=runtime,
            loop_detector=detector,
            progress_provider=_ConstantProgressProvider("s1"),
            tracer=tracer,
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
        exec_spans = exporter.spans_named(SPAN_TOOL_EXECUTE)
        attempt_spans = exporter.spans_named(SPAN_TOOL_ATTEMPT)

        assert len(attempt_spans) == 2
        assert attempt_spans[0].status.status_code == StatusCode.ERROR
        assert attempt_spans[1].status.status_code == StatusCode.OK

        assert action_span.status.status_code == StatusCode.OK
        assert action_span.attributes[ATTR_AGENT_ACTION_SUCCESS] is True

        # Retry event on execute span.
        retry_events = [
            e for e in exec_spans[0].events if e.name == EVENT_RETRY_SCHEDULED
        ]
        assert len(retry_events) == 1


# ---------------------------------------------------------------------------
# Section 48 — Loop End-to-End Action Trace
# ---------------------------------------------------------------------------


class TestLoopEndToEndActionTrace:
    def test_loop_detection_and_replan_trace(self):
        """3 identical actions → loop detected → replan required."""
        tracer, exporter = make_recording_tracer()
        registry = ToolRegistry()
        spec = _spec()
        registry.register(spec, SuccessTool("ok"))
        runtime = ToolRuntime(registry=registry, tracer=tracer, sleep=FakeSleeper())
        detector = LoopDetector(
            config=LoopDetectorConfig(), history=ActionHistory(max_size=64)
        )
        controller = AgentActionController(
            tool_runtime=runtime,
            loop_detector=detector,
            progress_provider=_ConstantProgressProvider("same"),  # identical progress
            tracer=tracer,
        )
        call = ToolCall(tool_name="tool", arguments={})
        ctx = ToolExecutionContext()

        for _ in range(3):
            _run(controller.execute(call, ctx))

        assert controller.state is ActionControllerState.REPLAN_REQUIRED

        action_spans = exporter.spans_named(SPAN_AGENT_ACTION)
        third = action_spans[2]

        # Loop detected event.
        loop_events = [e for e in third.events if e.name == EVENT_AGENT_LOOP_DETECTED]
        assert len(loop_events) == 1

        # Replan required event.
        replan_events = [
            e for e in third.events if e.name == EVENT_AGENT_REPLAN_REQUIRED
        ]
        assert len(replan_events) == 1

        # Loop detected attribute.
        assert third.attributes[ATTR_AGENT_ACTION_LOOP_DETECTED] is True


# ---------------------------------------------------------------------------
# Section 49 — Blocked Action Trace
# ---------------------------------------------------------------------------


class TestBlockedActionTrace:
    def test_blocked_action_no_tool_execute(self):
        """4th action blocked: agent.replan.blocked, no tool.execute."""
        tracer, exporter = make_recording_tracer()
        registry = ToolRegistry()
        spec = _spec()
        registry.register(spec, SuccessTool("ok"))
        runtime = ToolRuntime(registry=registry, tracer=tracer, sleep=FakeSleeper())
        detector = LoopDetector(
            config=LoopDetectorConfig(), history=ActionHistory(max_size=64)
        )
        controller = AgentActionController(
            tool_runtime=runtime,
            loop_detector=detector,
            progress_provider=_ConstantProgressProvider("same"),
            tracer=tracer,
        )
        call = ToolCall(tool_name="tool", arguments={})
        ctx = ToolExecutionContext()

        for _ in range(3):
            _run(controller.execute(call, ctx))

        with pytest.raises(ReplanRequiredError):
            _run(controller.execute(call, ctx))

        action_spans = exporter.spans_named(SPAN_AGENT_ACTION)
        blocked = action_spans[3]

        # Blocked event.
        blocked_events = [
            e for e in blocked.events if e.name == EVENT_AGENT_REPLAN_BLOCKED
        ]
        assert len(blocked_events) == 1

        # No new tool.execute (still only 3 from before).
        assert len(exporter.spans_named(SPAN_TOOL_EXECUTE)) == 3
        # No new tool.attempt.
        assert len(exporter.spans_named(SPAN_TOOL_ATTEMPT)) == 3


# ---------------------------------------------------------------------------
# Section 50 — Replan Acknowledge Trace
# ---------------------------------------------------------------------------


class TestReplanAcknowledgeTrace:
    def test_acknowledge_then_new_action(self):
        """acknowledge_replan → new action works normally."""
        tracer, exporter = make_recording_tracer()
        registry = ToolRegistry()
        spec = _spec()
        registry.register(spec, SuccessTool("ok"))
        runtime = ToolRuntime(registry=registry, tracer=tracer, sleep=FakeSleeper())
        detector = LoopDetector(
            config=LoopDetectorConfig(), history=ActionHistory(max_size=64)
        )
        controller = AgentActionController(
            tool_runtime=runtime,
            loop_detector=detector,
            progress_provider=_ConstantProgressProvider("same"),
            tracer=tracer,
        )
        call = ToolCall(tool_name="tool", arguments={})
        ctx = ToolExecutionContext()

        for _ in range(3):
            _run(controller.execute(call, ctx))

        controller.acknowledge_replan()
        assert controller.state is ActionControllerState.READY

        # New action should work.
        result = _run(controller.execute(call, ctx))
        assert result.tool_result.success is True

        # Acknowledge span exists.
        ack_spans = exporter.spans_named(SPAN_AGENT_REPLAN_ACKNOWLEDGE)
        assert len(ack_spans) == 1
        assert ack_spans[0].status.status_code == StatusCode.OK


# ---------------------------------------------------------------------------
# Section 51 — Context Trace Test
# ---------------------------------------------------------------------------


class TestContextTraceSmoke:
    def test_context_assemble_trace(self):
        """context.assemble span with correct counts/tokens."""
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        items = [
            ContextItem(
                item_id="critical",
                kind=ContextKind.CRITICAL_STATE,
                content="critical state",
                estimated_tokens=10,
                sequence_index=0,
                must_keep=True,
            ),
            ContextItem(
                item_id="recent",
                kind=ContextKind.RECENT_INTERACTION,
                content="recent interaction",
                estimated_tokens=20,
                sequence_index=1,
                priority=ContextPriority.HIGH,
            ),
            ContextItem(
                item_id="historical",
                kind=ContextKind.HISTORICAL_EVIDENCE,
                content="historical evidence",
                estimated_tokens=30,
                sequence_index=2,
                priority=ContextPriority.LOW,
            ),
        ]

        result = assembler.assemble(items, ContextBudget(max_tokens=100))

        span = exporter.span_by_name(SPAN_CONTEXT_ASSEMBLE)
        assert span.status.status_code == StatusCode.OK
        assert span.attributes[ATTR_CONTEXT_TOKEN_MAX] == 100
        assert span.attributes[ATTR_CONTEXT_INCLUDED_COUNT] == 3
        assert span.attributes[ATTR_CONTEXT_TOKEN_USED] == 60


# ---------------------------------------------------------------------------
# Section 52 — Artifact Trace Test
# ---------------------------------------------------------------------------


class TestArtifactTraceSmoke:
    def test_artifact_externalization_trace(self):
        """artifact.process_output span for externalized output."""
        tracer, exporter = make_recording_tracer()
        store = InMemoryArtifactStore()

        class _FakeEstimator:
            def estimate(self, text: str) -> int:
                return max(1, len(text) // 4)

        class _IdFactory:
            def __init__(self, prefix):
                self.prefix = prefix
                self.counter = 1

            def __call__(self):
                id_str = f"{self.prefix}-{self.counter:03d}"
                self.counter += 1
                return id_str

        from datetime import datetime, timezone
        proc = ToolOutputProcessor(
            artifact_store=store,
            token_estimator=_FakeEstimator(),
            output_renderer=DeterministicToolOutputRenderer(),
            externalization_policy=OutputExternalizationPolicy(
                inline_token_limit=2, preview_chars=10,
            ),
            artifact_id_factory=_IdFactory("A"),
            context_item_id_factory=_IdFactory("CTX"),
            clock=lambda: datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            tracer=tracer,
        )

        result = _run(
            proc.process_success_output(
                tool_name="big_tool",
                output="x" * 100,
                sequence_index=0,
            )
        )

        assert result.externalized is True

        span = exporter.span_by_name(SPAN_ARTIFACT_PROCESS_OUTPUT)
        assert span.status.status_code == StatusCode.OK
        assert span.attributes[ATTR_ARTIFACT_EXTERNALIZED] is True
        assert span.attributes[ATTR_ARTIFACT_ID] == "A-001"


# ---------------------------------------------------------------------------
# Section 53-54 — MCP Discovery Trace Test
# ---------------------------------------------------------------------------


class TestMCPDiscoveryTraceSmoke:
    def test_mcp_discovery_trace(self):
        """mcp.discover span with provider id and tool count."""
        tracer, exporter = make_recording_tracer()
        client = _FakeMCPClient(tools=[
            _FakeTool(name="add", description="add two numbers"),
            _FakeTool(name="mul", description="multiply two numbers"),
        ])
        provider = MCPToolProvider(
            provider_id="demo",
            client=client,
            tracer=tracer,
        )
        registry = ToolRegistry()

        result = _run(provider.discover_into(registry))

        span = exporter.span_by_name(SPAN_MCP_DISCOVER)
        assert span.status.status_code == StatusCode.OK
        assert span.attributes[ATTR_MCP_PROVIDER_ID] == "demo"
        assert span.attributes[ATTR_MCP_DISCOVERY_TOOL_COUNT] == 2


# ---------------------------------------------------------------------------
# Section 55 — Telemetry Failure Isolation (all components)
# ---------------------------------------------------------------------------


class FailingTracer:
    def start_as_current_span(self, name, **kwargs):
        raise RuntimeError("span creation failure")


class TestTelemetryFailureIsolationAll:
    def test_controller_unaffected(self):
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
            progress_provider=_ConstantProgressProvider("s1"),
            tracer=FailingTracer(),
        )

        result = _run(
            controller.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )
        assert result.tool_result.success is True

    def test_context_unaffected(self):
        assembler = ContextAssembler(tracer=FailingTracer())
        result = assembler.assemble(
            [
                ContextItem(
                    item_id="x",
                    kind=ContextKind.RECENT_INTERACTION,
                    content="x",
                    estimated_tokens=1,
                    sequence_index=0,
                ),
            ],
            ContextBudget(max_tokens=100),
        )
        assert len(result.included_items) == 1

    def test_artifact_unaffected(self):
        store = InMemoryArtifactStore()

        class _FakeEstimator:
            def estimate(self, text: str) -> int:
                return max(1, len(text) // 4)

        class _IdFactory:
            def __init__(self, prefix):
                self.prefix = prefix
                self.counter = 1

            def __call__(self):
                id_str = f"{self.prefix}-{self.counter:03d}"
                self.counter += 1
                return id_str

        from datetime import datetime, timezone
        proc = ToolOutputProcessor(
            artifact_store=store,
            token_estimator=_FakeEstimator(),
            output_renderer=DeterministicToolOutputRenderer(),
            externalization_policy=OutputExternalizationPolicy(
                inline_token_limit=2, preview_chars=10,
            ),
            artifact_id_factory=_IdFactory("A"),
            context_item_id_factory=_IdFactory("CTX"),
            clock=lambda: datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            tracer=FailingTracer(),
        )

        result = _run(
            proc.process_success_output(
                tool_name="big_tool",
                output="x" * 100,
                sequence_index=0,
            )
        )
        assert result.externalized is True

    def test_mcp_discovery_unaffected(self):
        client = _FakeMCPClient(tools=[_FakeTool(name="add")])
        provider = MCPToolProvider(
            provider_id="demo",
            client=client,
            tracer=FailingTracer(),
        )
        registry = ToolRegistry()

        result = _run(provider.discover_into(registry))
        assert len(result.registered_tool_names) == 1


# ---------------------------------------------------------------------------
# Section 66 — No model.call span
# ---------------------------------------------------------------------------


class TestNoModelSpan:
    def test_no_model_call_span(self):
        tracer, exporter = make_recording_tracer()
        registry = ToolRegistry()
        spec = _spec()
        registry.register(spec, SuccessTool("ok"))
        runtime = ToolRuntime(registry=registry, tracer=tracer, sleep=FakeSleeper())
        detector = LoopDetector(
            config=LoopDetectorConfig(), history=ActionHistory(max_size=64)
        )
        controller = AgentActionController(
            tool_runtime=runtime,
            loop_detector=detector,
            progress_provider=_ConstantProgressProvider("s1"),
            tracer=tracer,
        )

        _run(
            controller.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )

        all_names = {s.name for s in exporter.finished_spans}
        assert "model.call" not in all_names
