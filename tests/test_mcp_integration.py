"""Integration tests with the real official MCP Python SDK v2 (Phase 5 Step 1).

These tests use the real ``mcp.Client`` + ``mcp.server.MCPServer``
in-process (no TCP port, no HTTP server, no subprocess, no network)
to prove the full round-trip:

    official MCPServer
        |
        v
    official MCP Client
        |
        v
    MCPToolProvider
        |
        v
    ToolRegistry
        |
        v
    ToolRuntime
        |
        v
    MCP call
        |
        v
    ToolExecutionResult

All offline, deterministic, no network.
"""

from __future__ import annotations

import asyncio
from typing import Any, TypedDict

import pytest
from mcp import Client
from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from mcp_adapter import MCPToolProvider
from tools import (
    ToolCall,
    ToolErrorType,
    ToolExecutionContext,
    ToolRegistry,
    ToolRuntime,
    ToolSideEffect,
    ToolSource,
)


def _run(coro: Any) -> Any:  # type: ignore[name-defined]
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helper: build an in-process MCP server with tools
# ---------------------------------------------------------------------------


def _make_echo_server(
    *,
    annotations: ToolAnnotations | None = None,
    structured_output: bool = False,
) -> MCPServer:
    """Build an MCPServer with a single ``echo`` tool returning text."""
    server = MCPServer(name="test-server", version="1.0")

    @server.tool(annotations=annotations, structured_output=structured_output)
    def echo(text: str) -> str:
        """Echo the text argument."""
        return text

    return server


class AddResult(TypedDict):
    sum: int


def _make_add_server() -> MCPServer:
    """Build an MCPServer with a structured-output ``add`` tool."""
    server = MCPServer(name="test-server", version="1.0")

    @server.tool()
    def add(a: int, b: int) -> AddResult:
        """Add two numbers."""
        return {"sum": a + b}

    return server


def _make_failing_server() -> MCPServer:
    """Build an MCPServer with a tool that raises an exception."""
    server = MCPServer(name="test-server", version="1.0")

    @server.tool()
    def fail(msg: str) -> str:
        """Always fails."""
        raise ValueError("tool failure: " + msg)

    return server


def _make_slow_server() -> MCPServer:
    """Build an MCPServer with a tool that sleeps longer than its timeout."""
    server = MCPServer(name="test-server", version="1.0")

    @server.tool()
    def slow(text: str) -> str:
        """Sleeps then returns (will hit timeout)."""
        import time
        time.sleep(5)
        return "should-not-reach"

    return server


# ---------------------------------------------------------------------------
# Discovery integration tests
# ---------------------------------------------------------------------------


class TestRealDiscovery:
    def test_discover_echo_tool(self):
        server = _make_echo_server()
        registry = ToolRegistry()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(provider_id="demo", client=client)
                result = await provider.discover_into(registry)
                return result

        result = _run(_do())
        assert result.provider_id == "demo"
        assert "mcp.demo.echo" in result.registered_tool_names
        assert registry.contains("mcp.demo.echo")

    def test_discovered_spec_has_mcp_source(self):
        server = _make_echo_server()
        registry = ToolRegistry()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(provider_id="demo", client=client)
                await provider.discover_into(registry)

        _run(_do())
        spec = registry.get("mcp.demo.echo").spec
        assert spec.source == ToolSource.MCP
        assert spec.provider_id == "demo"

    def test_discovered_spec_has_non_empty_description(self):
        server = _make_echo_server()
        registry = ToolRegistry()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(provider_id="demo", client=client)
                await provider.discover_into(registry)

        _run(_do())
        spec = registry.get("mcp.demo.echo").spec
        assert spec.description and spec.description.strip()

    def test_discovered_spec_has_input_schema(self):
        server = _make_echo_server()
        registry = ToolRegistry()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(provider_id="demo", client=client)
                await provider.discover_into(registry)

        _run(_do())
        spec = registry.get("mcp.demo.echo").spec
        assert isinstance(spec.input_schema, dict)
        assert spec.input_schema.get("type") == "object"

    def test_discover_multiple_tools(self):
        server = MCPServer(name="test-server", version="1.0")

        @server.tool()
        def echo(text: str) -> str:
            """Echo."""
            return text

        @server.tool()
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        registry = ToolRegistry()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(provider_id="demo", client=client)
                return await provider.discover_into(registry)

        result = _run(_do())
        assert len(result.registered_tool_names) == 2
        assert "mcp.demo.echo" in result.registered_tool_names
        assert "mcp.demo.add" in result.registered_tool_names


# ---------------------------------------------------------------------------
# Successful call integration tests
# ---------------------------------------------------------------------------


class TestRealSuccessfulCall:
    def test_echo_through_full_chain(self):
        """Full chain: MCPServer -> Client -> Provider -> Registry -> ToolRuntime."""
        server = _make_echo_server(structured_output=False)
        registry = ToolRegistry()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(provider_id="demo", client=client)
                await provider.discover_into(registry)
                runtime = ToolRuntime(registry=registry)
                result = await runtime.execute(
                    ToolCall(tool_name="mcp.demo.echo", arguments={"text": "hello"}),
                    ToolExecutionContext(),
                )
                return result

        result = _run(_do())
        assert result.success is True
        assert result.error is None
        assert result.tool_name == "mcp.demo.echo"
        # With structured_output=False, the adapter returns text.
        assert result.output == "hello"

    def test_structured_content_preferred(self):
        """Structured content is preferred over text content."""
        server = _make_add_server()
        registry = ToolRegistry()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(provider_id="demo", client=client)
                await provider.discover_into(registry)
                runtime = ToolRuntime(registry=registry)
                result = await runtime.execute(
                    ToolCall(tool_name="mcp.demo.add", arguments={"a": 3, "b": 4}),
                    ToolExecutionContext(),
                )
                return result

        result = _run(_do())
        assert result.success is True
        # structured_content is the dict, not the JSON text.
        assert result.output == {"sum": 7}

    def test_text_fallback_when_no_structured(self):
        """When structured_content is None, text content is returned."""
        server = _make_echo_server(structured_output=False)
        registry = ToolRegistry()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(provider_id="demo", client=client)
                await provider.discover_into(registry)
                runtime = ToolRuntime(registry=registry)
                result = await runtime.execute(
                    ToolCall(tool_name="mcp.demo.echo", arguments={"text": "world"}),
                    ToolExecutionContext(),
                )
                return result

        result = _run(_do())
        assert result.success is True
        assert result.output == "world"


# ---------------------------------------------------------------------------
# MCP tool error integration tests
# ---------------------------------------------------------------------------


class TestRealMCPError:
    def test_mcp_tool_error_becomes_execution_failure(self):
        """MCP is_error=True becomes ToolReportedFailure(EXECUTION)."""
        server = _make_failing_server()
        registry = ToolRegistry()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(provider_id="demo", client=client)
                await provider.discover_into(registry)
                runtime = ToolRuntime(registry=registry)
                result = await runtime.execute(
                    ToolCall(tool_name="mcp.demo.fail", arguments={"msg": "oops"}),
                    ToolExecutionContext(),
                )
                return result

        result = _run(_do())
        assert result.success is False
        assert result.error is not None
        assert result.error.error_type == ToolErrorType.EXECUTION


# ---------------------------------------------------------------------------
# Timeout integration tests
# ---------------------------------------------------------------------------


class TestRealTimeout:
    def test_timeout_uses_existing_runtime(self):
        """ToolRuntime timeout applies to MCP tools (adapter has no own timeout)."""
        server = _make_slow_server()
        registry = ToolRegistry()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(
                    provider_id="demo",
                    client=client,
                    default_timeout_seconds=0.5,
                )
                await provider.discover_into(registry)
                runtime = ToolRuntime(registry=registry)
                result = await runtime.execute(
                    ToolCall(tool_name="mcp.demo.slow", arguments={"text": "x"}),
                    ToolExecutionContext(),
                )
                return result

        result = _run(_do())
        assert result.success is False
        assert result.error is not None
        assert result.error.error_type == ToolErrorType.TIMEOUT


# ---------------------------------------------------------------------------
# Annotation trust integration tests
# ---------------------------------------------------------------------------


class TestRealAnnotationTrust:
    def test_untrusted_readonly_annotation_still_side_effecting(self):
        """Even with readOnlyHint=True, untrusted -> SIDE_EFFECTING."""
        server = _make_echo_server(
            annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True),
            structured_output=False,
        )
        registry = ToolRegistry()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(
                    provider_id="demo",
                    client=client,
                    trust_tool_annotations=False,
                )
                await provider.discover_into(registry)

        _run(_do())
        spec = registry.get("mcp.demo.echo").spec
        assert spec.side_effect == ToolSideEffect.SIDE_EFFECTING

    def test_trusted_readonly_annotation_maps_read_only(self):
        """With trust=True, readOnlyHint=True -> READ_ONLY."""
        server = _make_echo_server(
            annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True),
            structured_output=False,
        )
        registry = ToolRegistry()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(
                    provider_id="demo",
                    client=client,
                    trust_tool_annotations=True,
                )
                await provider.discover_into(registry)

        _run(_do())
        spec = registry.get("mcp.demo.echo").spec
        assert spec.side_effect == ToolSideEffect.READ_ONLY


# ---------------------------------------------------------------------------
# External cancellation integration test
# ---------------------------------------------------------------------------


class TestRealCancellation:
    def test_external_cancellation_propagates(self):
        """CancelledError propagates from MCP call through adapter."""
        server = _make_slow_server()
        registry = ToolRegistry()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(
                    provider_id="demo",
                    client=client,
                    default_timeout_seconds=30.0,
                )
                await provider.discover_into(registry)
                runtime = ToolRuntime(registry=registry)
                task = asyncio.create_task(
                    runtime.execute(
                        ToolCall(tool_name="mcp.demo.slow", arguments={"text": "x"}),
                        ToolExecutionContext(),
                    )
                )
                await asyncio.sleep(0.1)
                task.cancel()
                await task

        with pytest.raises(asyncio.CancelledError):
            _run(_do())


# ---------------------------------------------------------------------------
# Validation integration test
# ---------------------------------------------------------------------------


class TestRealValidation:
    def test_validation_failure_through_mcp(self):
        """ToolRuntime validation applies to MCP tools too."""
        server = _make_echo_server(structured_output=False)
        registry = ToolRegistry()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(provider_id="demo", client=client)
                await provider.discover_into(registry)
                runtime = ToolRuntime(registry=registry)
                result = await runtime.execute(
                    ToolCall(
                        tool_name="mcp.demo.echo",
                        arguments={"wrong_arg": "hello"},
                    ),
                    ToolExecutionContext(),
                )
                return result

        result = _run(_do())
        assert result.success is False
        assert result.error is not None
        assert result.error.error_type == ToolErrorType.VALIDATION
        assert result.attempt_count == 0


# ---------------------------------------------------------------------------
# Permission integration test
# ---------------------------------------------------------------------------


class TestRealPermission:
    def test_permission_check_applies_to_mcp(self):
        """ToolRuntime permission check applies to MCP tools."""
        server = _make_echo_server(structured_output=False)
        registry = ToolRegistry()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(
                    provider_id="demo",
                    client=client,
                    permission_resolver=lambda tool: frozenset({"mcp:demo"}),
                )
                await provider.discover_into(registry)
                runtime = ToolRuntime(registry=registry)
                # No permissions granted -> should fail.
                result = await runtime.execute(
                    ToolCall(tool_name="mcp.demo.echo", arguments={"text": "x"}),
                    ToolExecutionContext(granted_permissions=frozenset()),
                )
                return result

        result = _run(_do())
        assert result.success is False
        assert result.error is not None
        assert result.error.error_type == ToolErrorType.PERMISSION

    def test_permission_granted_allows_mcp(self):
        """With correct permissions, MCP tool executes."""
        server = _make_echo_server(structured_output=False)
        registry = ToolRegistry()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(
                    provider_id="demo",
                    client=client,
                    permission_resolver=lambda tool: frozenset({"mcp:demo"}),
                )
                await provider.discover_into(registry)
                runtime = ToolRuntime(registry=registry)
                result = await runtime.execute(
                    ToolCall(tool_name="mcp.demo.echo", arguments={"text": "x"}),
                    ToolExecutionContext(granted_permissions=frozenset({"mcp:demo"})),
                )
                return result

        result = _run(_do())
        assert result.success is True
