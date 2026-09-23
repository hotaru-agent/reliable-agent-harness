"""Phase 5 Step 1 smoke test — unified Local + MCP Tool Runtime.

Proves that a local Python tool and an MCP tool (discovered via the
official MCP Python SDK v2) coexist in the same ``ToolRegistry`` and
execute through the same ``ToolRuntime`` with identical:

* validation boundary
* permission boundary
* timeout mechanism
* structured result contract

All offline, deterministic, no network, no real LLM.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from mcp import Client
from mcp.server import MCPServer

from mcp_adapter import MCPToolProvider
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


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Local tool
# ---------------------------------------------------------------------------


_LOCAL_ECHO_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


async def _local_echo_handler(arguments: dict[str, Any]) -> Any:
    return arguments["text"]


def _make_local_echo_spec() -> ToolSpec:
    return ToolSpec(
        name="local_echo",
        description="local echo tool",
        input_schema=_LOCAL_ECHO_SCHEMA,
        required_permissions=frozenset(),
        timeout_seconds=5.0,
        side_effect=ToolSideEffect.READ_ONLY,
    )


# ---------------------------------------------------------------------------
# MCP server with an echo tool
# ---------------------------------------------------------------------------


def _make_mcp_echo_server() -> MCPServer:
    server = MCPServer(name="demo-server", version="1.0")

    @server.tool(structured_output=False)
    def echo(text: str) -> str:
        """Echo the text argument."""
        return text

    return server


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


class TestPhase5Step1Smoke:
    def test_local_and_mcp_coexist_same_registry(self):
        """Both tools register into the same ToolRegistry."""
        registry = ToolRegistry()

        # Register local tool.
        registry.register(_make_local_echo_spec(), _local_echo_handler)

        # Discover MCP tool into the same registry.
        server = _make_mcp_echo_server()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(provider_id="demo", client=client)
                await provider.discover_into(registry)

        _run(_do())

        assert registry.contains("local_echo")
        assert registry.contains("mcp.demo.echo")

    def test_same_runtime_executes_both(self):
        """The same ToolRuntime executes both local and MCP tools."""
        registry = ToolRegistry()
        registry.register(_make_local_echo_spec(), _local_echo_handler)

        server = _make_mcp_echo_server()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(provider_id="demo", client=client)
                await provider.discover_into(registry)

                runtime = ToolRuntime(registry=registry)

                # Execute local tool.
                local_result = await runtime.execute(
                    ToolCall(tool_name="local_echo", arguments={"text": "hello"}),
                    ToolExecutionContext(),
                )

                # Execute MCP tool.
                mcp_result = await runtime.execute(
                    ToolCall(tool_name="mcp.demo.echo", arguments={"text": "hello"}),
                    ToolExecutionContext(),
                )

                return local_result, mcp_result

        local_result, mcp_result = _run(_do())

        # Both succeed.
        assert local_result.success is True
        assert mcp_result.success is True

        # Both return the same output.
        assert local_result.output == "hello"
        assert mcp_result.output == "hello"

        # Both are ToolExecutionResult with no error.
        assert local_result.error is None
        assert mcp_result.error is None

    def test_same_validation_boundary(self):
        """Both tools share the same validation boundary."""
        registry = ToolRegistry()
        registry.register(_make_local_echo_spec(), _local_echo_handler)

        server = _make_mcp_echo_server()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(provider_id="demo", client=client)
                await provider.discover_into(registry)

                runtime = ToolRuntime(registry=registry)

                # Invalid args for local tool.
                local_invalid = await runtime.execute(
                    ToolCall(
                        tool_name="local_echo",
                        arguments={"wrong": "x"},
                    ),
                    ToolExecutionContext(),
                )

                # Invalid args for MCP tool.
                mcp_invalid = await runtime.execute(
                    ToolCall(
                        tool_name="mcp.demo.echo",
                        arguments={"wrong": "x"},
                    ),
                    ToolExecutionContext(),
                )

                return local_invalid, mcp_invalid

        local_invalid, mcp_invalid = _run(_do())

        # Both fail with VALIDATION error, attempt_count=0.
        assert local_invalid.success is False
        assert local_invalid.error.error_type == ToolErrorType.VALIDATION
        assert local_invalid.attempt_count == 0

        assert mcp_invalid.success is False
        assert mcp_invalid.error.error_type == ToolErrorType.VALIDATION
        assert mcp_invalid.attempt_count == 0

    def test_same_permission_boundary(self):
        """Both tools share the same permission boundary."""
        registry = ToolRegistry()

        # Local tool requires a permission.
        local_spec = ToolSpec(
            name="local_echo",
            description="local echo tool",
            input_schema=_LOCAL_ECHO_SCHEMA,
            required_permissions=frozenset({"local:echo"}),
            timeout_seconds=5.0,
            side_effect=ToolSideEffect.READ_ONLY,
        )
        registry.register(local_spec, _local_echo_handler)

        server = _make_mcp_echo_server()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(
                    provider_id="demo",
                    client=client,
                    permission_resolver=lambda tool: frozenset({"mcp:demo"}),
                )
                await provider.discover_into(registry)

                runtime = ToolRuntime(registry=registry)

                # No permissions granted -> both should fail.
                local_denied = await runtime.execute(
                    ToolCall(tool_name="local_echo", arguments={"text": "x"}),
                    ToolExecutionContext(granted_permissions=frozenset()),
                )
                mcp_denied = await runtime.execute(
                    ToolCall(tool_name="mcp.demo.echo", arguments={"text": "x"}),
                    ToolExecutionContext(granted_permissions=frozenset()),
                )

                # With permissions -> both should succeed.
                local_allowed = await runtime.execute(
                    ToolCall(tool_name="local_echo", arguments={"text": "x"}),
                    ToolExecutionContext(granted_permissions=frozenset({"local:echo"})),
                )
                mcp_allowed = await runtime.execute(
                    ToolCall(tool_name="mcp.demo.echo", arguments={"text": "x"}),
                    ToolExecutionContext(granted_permissions=frozenset({"mcp:demo"})),
                )

                return local_denied, mcp_denied, local_allowed, mcp_allowed

        local_denied, mcp_denied, local_allowed, mcp_allowed = _run(_do())

        # Both denied without permissions.
        assert local_denied.success is False
        assert local_denied.error.error_type == ToolErrorType.PERMISSION
        assert mcp_denied.success is False
        assert mcp_denied.error.error_type == ToolErrorType.PERMISSION

        # Both succeed with permissions.
        assert local_allowed.success is True
        assert mcp_allowed.success is True

    def test_same_timeout_mechanism(self):
        """Both tools share the same ToolRuntime timeout mechanism."""
        registry = ToolRegistry()

        # Local tool with short timeout.
        local_spec = ToolSpec(
            name="local_slow",
            description="local slow tool",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            },
            required_permissions=frozenset(),
            timeout_seconds=0.3,
            side_effect=ToolSideEffect.READ_ONLY,
        )

        async def _local_slow_handler(arguments: dict[str, Any]) -> Any:
            await asyncio.sleep(5)
            return "should-not-reach"

        registry.register(local_spec, _local_slow_handler)

        # MCP tool with short timeout.
        server = MCPServer(name="demo-server", version="1.0")

        @server.tool()
        def slow(text: str) -> str:
            """Sleeps then returns."""
            import time
            time.sleep(5)
            return "should-not-reach"

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(
                    provider_id="demo",
                    client=client,
                    default_timeout_seconds=0.3,
                )
                await provider.discover_into(registry)

                runtime = ToolRuntime(registry=registry)

                local_timeout = await runtime.execute(
                    ToolCall(tool_name="local_slow", arguments={}),
                    ToolExecutionContext(),
                )
                mcp_timeout = await runtime.execute(
                    ToolCall(tool_name="mcp.demo.slow", arguments={"text": "x"}),
                    ToolExecutionContext(),
                )

                return local_timeout, mcp_timeout

        local_timeout, mcp_timeout = _run(_do())

        # Both timeout with TIMEOUT error.
        assert local_timeout.success is False
        assert local_timeout.error.error_type == ToolErrorType.TIMEOUT
        assert mcp_timeout.success is False
        assert mcp_timeout.error.error_type == ToolErrorType.TIMEOUT

    def test_source_metadata_distinguishes_local_and_mcp(self):
        """Local tool has source=LOCAL; MCP tool has source=MCP."""
        registry = ToolRegistry()
        registry.register(_make_local_echo_spec(), _local_echo_handler)

        server = _make_mcp_echo_server()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(provider_id="demo", client=client)
                await provider.discover_into(registry)

        _run(_do())

        local_spec = registry.get("local_echo").spec
        mcp_spec = registry.get("mcp.demo.echo").spec

        assert local_spec.source == ToolSource.LOCAL
        assert local_spec.provider_id is None

        assert mcp_spec.source == ToolSource.MCP
        assert mcp_spec.provider_id == "demo"

    def test_namespaced_name_does_not_collide_with_local(self):
        """Local 'echo' and MCP 'echo' coexist with namespaced naming."""
        registry = ToolRegistry()

        # Register a local tool named "echo" (not namespaced).
        local_spec = ToolSpec(
            name="echo",
            description="local echo",
            input_schema=_LOCAL_ECHO_SCHEMA,
            required_permissions=frozenset(),
            timeout_seconds=5.0,
            side_effect=ToolSideEffect.READ_ONLY,
        )
        registry.register(local_spec, _local_echo_handler)

        server = _make_mcp_echo_server()

        async def _do():
            async with Client(server) as client:
                provider = MCPToolProvider(provider_id="demo", client=client)
                await provider.discover_into(registry)

                runtime = ToolRuntime(registry=registry)

                local_result = await runtime.execute(
                    ToolCall(tool_name="echo", arguments={"text": "local"}),
                    ToolExecutionContext(),
                )
                mcp_result = await runtime.execute(
                    ToolCall(tool_name="mcp.demo.echo", arguments={"text": "mcp"}),
                    ToolExecutionContext(),
                )

                return local_result, mcp_result

        local_result, mcp_result = _run(_do())

        assert local_result.success is True
        assert local_result.output == "local"
        assert mcp_result.success is True
        assert mcp_result.output == "mcp"
