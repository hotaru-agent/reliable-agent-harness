"""Tests for MCPToolProvider OpenTelemetry tracing (Phase 6 Step 2).

All offline, deterministic, no network, no collector. Uses a recording
span exporter to assert on span names, attributes, statuses.

Tests prove:
* mcp.discover span exists with provider id, annotation trusted
* success: tool count, page count, OK status
* failure: ERROR status, exception type recorded
* no schema/description in telemetry
* tracer injection isolation
* telemetry failure isolation
* real MCP SDK discovery tracing
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest
from opentelemetry.trace.status import Status, StatusCode

from mcp_adapter import MCPToolProvider, MCPToolCompatibilityError
from observability import (
    ATTR_ERROR_EXCEPTION_TYPE,
    ATTR_MCP_ANNOTATION_TRUSTED,
    ATTR_MCP_DISCOVERY_PAGE_COUNT,
    ATTR_MCP_DISCOVERY_TOOL_COUNT,
    ATTR_MCP_PROVIDER_ID,
    SPAN_MCP_DISCOVER,
)
from tests.fakes_telemetry import make_recording_tracer
from tools import ToolRegistry


# ---------------------------------------------------------------------------
# Fake MCP client / types
# ---------------------------------------------------------------------------

_SECRET_SCHEMA_MARKER = "VERY_SECRET_SCHEMA_MARKER"


@dataclass
class FakeTextContent:
    type: str = "text"
    text: str = ""
    annotations: Any = None
    meta: Any = None


@dataclass
class FakeToolAnnotations:
    read_only_hint: Optional[bool] = None
    destructive_hint: Optional[bool] = None
    idempotent_hint: Optional[bool] = None
    open_world_hint: Optional[bool] = None


@dataclass
class FakeTool:
    name: str
    description: Optional[str] = None
    input_schema: dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    })
    annotations: Optional[FakeToolAnnotations] = None


@dataclass
class FakeListToolsResult:
    tools: list[FakeTool]
    next_cursor: Optional[str] = None


@dataclass
class FakeCallToolResult:
    content: list[Any]
    structured_content: Any = None
    is_error: bool = False


class FakeMCPClient:
    def __init__(
        self,
        *,
        list_tools_pages: list[FakeListToolsResult] | None = None,
    ) -> None:
        self._pages = list_tools_pages or []
        self.list_tools_calls: list[Optional[str]] = []

    async def list_tools(self, *, cursor: Optional[str] = None) -> Any:
        self.list_tools_calls.append(cursor)
        idx = len(self.list_tools_calls) - 1
        if idx < len(self._pages):
            return self._pages[idx]
        return FakeListToolsResult(tools=[])

    async def call_tool(
        self,
        name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> Any:
        raise NotImplementedError("not used in discovery tests")


def _run(coro):
    return asyncio.run(coro)


def _make_provider(
    tracer,
    *,
    client: FakeMCPClient,
    trust_tool_annotations: bool = False,
    **kwargs,
) -> MCPToolProvider:
    return MCPToolProvider(
        provider_id="demo",
        client=client,
        trust_tool_annotations=trust_tool_annotations,
        tracer=tracer,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Section 42 — Successful Discovery
# ---------------------------------------------------------------------------


class TestSuccessfulDiscoveryTracing:
    def test_span_exists(self):
        tracer, exporter = make_recording_tracer()
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(tools=[
                    FakeTool(name="add", description="add two numbers"),
                    FakeTool(name="mul", description="multiply two numbers"),
                ]),
            ]
        )
        provider = _make_provider(tracer, client=client)
        registry = ToolRegistry()

        _run(provider.discover_into(registry))

        spans = exporter.spans_named(SPAN_MCP_DISCOVER)
        assert len(spans) == 1

    def test_provider_id_attribute(self):
        tracer, exporter = make_recording_tracer()
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(tools=[FakeTool(name="add")]),
            ]
        )
        provider = _make_provider(tracer, client=client)
        registry = ToolRegistry()

        _run(provider.discover_into(registry))

        span = exporter.span_by_name(SPAN_MCP_DISCOVER)
        assert span.attributes[ATTR_MCP_PROVIDER_ID] == "demo"

    def test_annotation_trusted_attribute(self):
        tracer, exporter = make_recording_tracer()
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(tools=[FakeTool(name="add")]),
            ]
        )
        provider = _make_provider(
            tracer, client=client, trust_tool_annotations=True
        )
        registry = ToolRegistry()

        _run(provider.discover_into(registry))

        span = exporter.span_by_name(SPAN_MCP_DISCOVER)
        assert span.attributes[ATTR_MCP_ANNOTATION_TRUSTED] is True

    def test_tool_count_attribute(self):
        tracer, exporter = make_recording_tracer()
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(tools=[
                    FakeTool(name="add"),
                    FakeTool(name="mul"),
                ]),
            ]
        )
        provider = _make_provider(tracer, client=client)
        registry = ToolRegistry()

        _run(provider.discover_into(registry))

        span = exporter.span_by_name(SPAN_MCP_DISCOVER)
        assert span.attributes[ATTR_MCP_DISCOVERY_TOOL_COUNT] == 2

    def test_page_count_attribute(self):
        tracer, exporter = make_recording_tracer()
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(tools=[FakeTool(name="add")], next_cursor="c1"),
                FakeListToolsResult(tools=[FakeTool(name="mul")]),
            ]
        )
        provider = _make_provider(tracer, client=client)
        registry = ToolRegistry()

        _run(provider.discover_into(registry))

        span = exporter.span_by_name(SPAN_MCP_DISCOVER)
        assert span.attributes[ATTR_MCP_DISCOVERY_PAGE_COUNT] == 2

    def test_ok_status(self):
        tracer, exporter = make_recording_tracer()
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(tools=[FakeTool(name="add")]),
            ]
        )
        provider = _make_provider(tracer, client=client)
        registry = ToolRegistry()

        _run(provider.discover_into(registry))

        span = exporter.span_by_name(SPAN_MCP_DISCOVER)
        assert span.status.status_code == StatusCode.OK


# ---------------------------------------------------------------------------
# Section 43 — Failed Discovery
# ---------------------------------------------------------------------------


class TestFailedDiscoveryTracing:
    def test_error_status_on_compatibility_error(self):
        tracer, exporter = make_recording_tracer()
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(tools=[
                    FakeTool(
                        name="bad",
                        input_schema={"oneOf": []},  # unsupported schema
                    ),
                ]),
            ]
        )
        provider = _make_provider(tracer, client=client)
        registry = ToolRegistry()

        with pytest.raises(MCPToolCompatibilityError):
            _run(provider.discover_into(registry))

        span = exporter.span_by_name(SPAN_MCP_DISCOVER)
        assert span.status.status_code == StatusCode.ERROR

    def test_exception_type_recorded(self):
        tracer, exporter = make_recording_tracer()
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(tools=[
                    FakeTool(
                        name="bad",
                        input_schema={"oneOf": []},
                    ),
                ]),
            ]
        )
        provider = _make_provider(tracer, client=client)
        registry = ToolRegistry()

        with pytest.raises(MCPToolCompatibilityError):
            _run(provider.discover_into(registry))

        span = exporter.span_by_name(SPAN_MCP_DISCOVER)
        assert (
            span.attributes[ATTR_ERROR_EXCEPTION_TYPE]
            == "MCPToolCompatibilityError"
        )

    def test_no_tool_count_on_failure(self):
        tracer, exporter = make_recording_tracer()
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(tools=[
                    FakeTool(
                        name="bad",
                        input_schema={"oneOf": []},
                    ),
                ]),
            ]
        )
        provider = _make_provider(tracer, client=client)
        registry = ToolRegistry()

        with pytest.raises(MCPToolCompatibilityError):
            _run(provider.discover_into(registry))

        span = exporter.span_by_name(SPAN_MCP_DISCOVER)
        # tool_count should NOT be set on failure (no tools registered).
        assert ATTR_MCP_DISCOVERY_TOOL_COUNT not in span.attributes


# ---------------------------------------------------------------------------
# Section 41 — No Schema / Description in Telemetry
# ---------------------------------------------------------------------------


class TestNoSensitiveData:
    def test_no_schema_in_telemetry(self):
        tracer, exporter = make_recording_tracer()
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(tools=[
                    FakeTool(
                        name="add",
                        description="add two numbers",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                            },
                            "required": ["x", "y"],
                            "additionalProperties": False,
                        },
                    ),
                ]),
            ]
        )
        provider = _make_provider(tracer, client=client)
        registry = ToolRegistry()

        _run(provider.discover_into(registry))

        for s in exporter.finished_spans:
            for key, val in (s.attributes or {}).items():
                assert "properties" not in str(val)
                assert "additionalProperties" not in str(val)
                assert "required" not in str(val)
            for ev in s.events:
                for key, val in (ev.attributes or {}).items():
                    assert "properties" not in str(val)

    def test_no_description_in_telemetry(self):
        tracer, exporter = make_recording_tracer()
        secret_desc = "SECRET_DESCRIPTION_MARKER"
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(tools=[
                    FakeTool(name="add", description=secret_desc),
                ]),
            ]
        )
        provider = _make_provider(tracer, client=client)
        registry = ToolRegistry()

        _run(provider.discover_into(registry))

        for s in exporter.finished_spans:
            for key, val in (s.attributes or {}).items():
                assert secret_desc not in str(val)
            for ev in s.events:
                for key, val in (ev.attributes or {}).items():
                    assert secret_desc not in str(val)


# ---------------------------------------------------------------------------
# Section 55 — Telemetry Failure Isolation
# ---------------------------------------------------------------------------


class FailingTracer:
    def start_as_current_span(self, name, **kwargs):
        raise RuntimeError("span creation failure")


class TestTelemetryFailureIsolation:
    def test_discovery_unaffected_by_failing_tracer(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(tools=[FakeTool(name="add")]),
            ]
        )
        provider = _make_provider(FailingTracer(), client=client)
        registry = ToolRegistry()

        result = _run(provider.discover_into(registry))
        assert len(result.registered_tool_names) == 1


# ---------------------------------------------------------------------------
# Section 67 — Deterministic Span Names
# ---------------------------------------------------------------------------


class TestDeterministicSpanNames:
    def test_span_name_is_fixed(self):
        tracer, exporter = make_recording_tracer()
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(tools=[FakeTool(name="add")]),
            ]
        )
        provider = _make_provider(tracer, client=client)
        registry = ToolRegistry()

        _run(provider.discover_into(registry))

        all_names = {s.name for s in exporter.finished_spans}
        assert all_names == {SPAN_MCP_DISCOVER}


# ---------------------------------------------------------------------------
# No SDK Configured
# ---------------------------------------------------------------------------


class TestNoSDKConfigured:
    def test_discovery_without_tracer(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(tools=[FakeTool(name="add")]),
            ]
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()

        result = _run(provider.discover_into(registry))
        assert len(result.registered_tool_names) == 1


# ---------------------------------------------------------------------------
# Section 54 — Real MCP SDK Discovery Tracing
# ---------------------------------------------------------------------------


class TestRealMCPSDKDiscoveryTracing:
    """Integration test using the real MCP Python SDK v2 in-process."""

    def test_real_mcp_discovery_traced(self):
        """Discover tools from a real in-process MCP server and verify
        the mcp.discover span is produced correctly."""
        try:
            from mcp import Client
            from mcp.server import MCPServer
        except ImportError:
            pytest.skip("MCP SDK not available")

        tracer, exporter = make_recording_tracer()

        # Build a minimal in-process MCP server.
        server = MCPServer(name="test-server", version="1.0")

        @server.tool()
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(
                    provider_id="real-demo",
                    client=client,
                    tracer=tracer,
                )
                registry = ToolRegistry()
                result = await provider.discover_into(registry)
                return result

        result = _run(_do())

        span = exporter.span_by_name(SPAN_MCP_DISCOVER)
        assert span is not None
        assert span.attributes[ATTR_MCP_PROVIDER_ID] == "real-demo"
        assert span.attributes[ATTR_MCP_DISCOVERY_TOOL_COUNT] >= 1
        assert span.status.status_code == StatusCode.OK
