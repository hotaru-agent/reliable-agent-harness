"""Unit / component tests for the MCP Tool Provider Adapter (Phase 5 Step 1).

Uses a deterministic fake MCP client (no network, no real MCP SDK) to
test the adapter logic in isolation:

* namespace naming
* description fallback
* annotation mapping (trusted / untrusted / missing)
* side-effect mapping
* schema compatibility gate
* pagination
* discovery atomicity / collision preflight
* result normalization (structured_content, text fallback, is_error)
* unsupported content rejection
* source metadata

All offline, deterministic, no network.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from mcp_adapter import (
    MCPDiscoveryError,
    MCPToolCompatibilityError,
    MCPToolProvider,
    fallback_description,
    make_namespaced_name,
    map_side_effect,
    normalize_call_result,
)
from tools import (
    RetryPolicy,
    ToolErrorType,
    ToolRegistry,
    ToolSideEffect,
    ToolSource,
    ToolSpec,
)
from tools.errors import ToolReportedFailure


# ---------------------------------------------------------------------------
# Fake MCP client / types
# ---------------------------------------------------------------------------


@dataclass
class FakeTextContent:
    type: str = "text"
    text: str = ""
    annotations: Any = None
    meta: Any = None


@dataclass
class FakeImageContent:
    type: str = "image"
    data: str = ""
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
    """Deterministic fake MCP client for unit tests.

    ``list_tools_pages`` is a list of ``FakeListToolsResult`` pages.
    ``call_tool_results`` maps remote tool name -> ``FakeCallToolResult``.
    ``call_tool_exception`` maps remote tool name -> exception to raise.
    """

    def __init__(
        self,
        *,
        list_tools_pages: list[FakeListToolsResult] | None = None,
        call_tool_results: dict[str, FakeCallToolResult] | None = None,
        call_tool_exceptions: dict[str, BaseException] | None = None,
    ) -> None:
        self._pages = list_tools_pages or []
        self._results = call_tool_results or {}
        self._exceptions = call_tool_exceptions or {}
        self.list_tools_calls: list[Optional[str]] = []
        self.call_tool_calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self, *, cursor: Optional[str] = None) -> FakeListToolsResult:
        self.list_tools_calls.append(cursor)
        if not self._pages:
            return FakeListToolsResult(tools=[], next_cursor=None)
        idx = len(self.list_tools_calls) - 1
        if idx < len(self._pages):
            return self._pages[idx]
        return FakeListToolsResult(tools=[], next_cursor=None)

    async def call_tool(
        self,
        name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> FakeCallToolResult:
        self.call_tool_calls.append((name, dict(arguments or {})))
        if name in self._exceptions:
            raise self._exceptions[name]
        if name in self._results:
            return self._results[name]
        return FakeCallToolResult(content=[], structured_content=None, is_error=False)


def _run(coro: Any) -> Any:
    """Run an async coroutine synchronously (no pytest-asyncio needed)."""
    return asyncio.run(coro)


def _simple_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }


# ---------------------------------------------------------------------------
# Naming tests (Section 18-20)
# ---------------------------------------------------------------------------


class TestMakeNamespacedName:
    def test_basic(self):
        assert make_namespaced_name("demo", "echo") == "mcp.demo.echo"

    def test_multi_segment_provider(self):
        assert make_namespaced_name("github", "search_issues") == "mcp.github.search_issues"

    def test_empty_provider_rejected(self):
        with pytest.raises(ValueError, match="provider_id"):
            make_namespaced_name("", "echo")

    def test_empty_remote_name_rejected(self):
        with pytest.raises(ValueError, match="remote_name"):
            make_namespaced_name("demo", "")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValueError, match="provider_id"):
            make_namespaced_name("  ", "echo")


# ---------------------------------------------------------------------------
# Description fallback tests (Section 21)
# ---------------------------------------------------------------------------


class TestFallbackDescription:
    def test_deterministic(self):
        d1 = fallback_description("demo", "echo")
        d2 = fallback_description("demo", "echo")
        assert d1 == d2

    def test_format(self):
        assert fallback_description("demo", "echo") == "MCP tool 'echo' from provider 'demo'"

    def test_different_tools_different(self):
        assert fallback_description("demo", "echo") != fallback_description("demo", "add")


# ---------------------------------------------------------------------------
# Annotation mapping tests (Section 25-29)
# ---------------------------------------------------------------------------


class TestMapSideEffect:
    def test_untrusted_always_side_effecting_readonly(self):
        ann = FakeToolAnnotations(read_only_hint=True, idempotent_hint=True)
        assert map_side_effect(ann, trust_tool_annotations=False) == ToolSideEffect.SIDE_EFFECTING

    def test_untrusted_always_side_effecting_idempotent(self):
        ann = FakeToolAnnotations(idempotent_hint=True)
        assert map_side_effect(ann, trust_tool_annotations=False) == ToolSideEffect.SIDE_EFFECTING

    def test_untrusted_missing_annotations(self):
        assert map_side_effect(None, trust_tool_annotations=False) == ToolSideEffect.SIDE_EFFECTING

    def test_trusted_readonly_maps_read_only(self):
        ann = FakeToolAnnotations(read_only_hint=True, idempotent_hint=True)
        assert map_side_effect(ann, trust_tool_annotations=True) == ToolSideEffect.READ_ONLY

    def test_trusted_readonly_priority_over_idempotent(self):
        ann = FakeToolAnnotations(read_only_hint=True, idempotent_hint=True)
        result = map_side_effect(ann, trust_tool_annotations=True)
        assert result == ToolSideEffect.READ_ONLY

    def test_trusted_idempotent_without_readonly(self):
        ann = FakeToolAnnotations(read_only_hint=False, idempotent_hint=True)
        assert map_side_effect(ann, trust_tool_annotations=True) == ToolSideEffect.IDEMPOTENT

    def test_trusted_missing_annotations_side_effecting(self):
        assert map_side_effect(None, trust_tool_annotations=True) == ToolSideEffect.SIDE_EFFECTING

    def test_trusted_no_hints_side_effecting(self):
        ann = FakeToolAnnotations()
        assert map_side_effect(ann, trust_tool_annotations=True) == ToolSideEffect.SIDE_EFFECTING

    def test_trusted_readonly_false_idempotent_false(self):
        ann = FakeToolAnnotations(read_only_hint=False, idempotent_hint=False)
        assert map_side_effect(ann, trust_tool_annotations=True) == ToolSideEffect.SIDE_EFFECTING

    def test_trusted_readonly_none_idempotent_none(self):
        ann = FakeToolAnnotations(read_only_hint=None, idempotent_hint=None)
        assert map_side_effect(ann, trust_tool_annotations=True) == ToolSideEffect.SIDE_EFFECTING


# ---------------------------------------------------------------------------
# Result normalization tests (Section 34-37, 62-63)
# ---------------------------------------------------------------------------


class TestNormalizeCallResult:
    def test_structured_content_preferred(self):
        result = FakeCallToolResult(
            content=[FakeTextContent(text="ignored")],
            structured_content={"sum": 7},
            is_error=False,
        )
        assert normalize_call_result(result, "add") == {"sum": 7}

    def test_text_fallback_single(self):
        result = FakeCallToolResult(
            content=[FakeTextContent(text="hello")],
            structured_content=None,
            is_error=False,
        )
        assert normalize_call_result(result, "echo") == "hello"

    def test_text_fallback_multiple_joined(self):
        result = FakeCallToolResult(
            content=[
                FakeTextContent(text="line1"),
                FakeTextContent(text="line2"),
            ],
            structured_content=None,
            is_error=False,
        )
        assert normalize_call_result(result, "echo") == "line1\nline2"

    def test_text_fallback_empty_content(self):
        result = FakeCallToolResult(
            content=[],
            structured_content=None,
            is_error=False,
        )
        assert normalize_call_result(result, "echo") == ""

    def test_is_error_with_text_message(self):
        result = FakeCallToolResult(
            content=[FakeTextContent(text="something went wrong")],
            structured_content=None,
            is_error=True,
        )
        with pytest.raises(ToolReportedFailure) as exc_info:
            normalize_call_result(result, "failing_tool")
        assert exc_info.value.error_type == ToolErrorType.EXECUTION
        assert "something went wrong" in exc_info.value.message

    def test_is_error_without_text_fallback(self):
        result = FakeCallToolResult(
            content=[FakeImageContent(data="abc")],
            structured_content=None,
            is_error=True,
        )
        with pytest.raises(ToolReportedFailure) as exc_info:
            normalize_call_result(result, "failing_tool")
        assert exc_info.value.error_type == ToolErrorType.EXECUTION
        assert "failing_tool" in exc_info.value.message

    def test_unsupported_non_text_content(self):
        result = FakeCallToolResult(
            content=[FakeImageContent(data="abc")],
            structured_content=None,
            is_error=False,
        )
        with pytest.raises(ToolReportedFailure) as exc_info:
            normalize_call_result(result, "image_tool")
        assert exc_info.value.error_type == ToolErrorType.EXECUTION
        assert "unsupported" in exc_info.value.message.lower()

    def test_mixed_text_and_image_unsupported(self):
        result = FakeCallToolResult(
            content=[
                FakeTextContent(text="hello"),
                FakeImageContent(data="abc"),
            ],
            structured_content=None,
            is_error=False,
        )
        with pytest.raises(ToolReportedFailure):
            normalize_call_result(result, "mixed_tool")


# ---------------------------------------------------------------------------
# Provider construction tests
# ---------------------------------------------------------------------------


class TestMCPToolProviderConstruction:
    def test_valid_construction(self):
        client = FakeMCPClient()
        provider = MCPToolProvider(
            provider_id="demo",
            client=client,
            default_timeout_seconds=10.0,
        )
        assert provider is not None

    def test_empty_provider_id_rejected(self):
        client = FakeMCPClient()
        with pytest.raises(ValueError, match="provider_id"):
            MCPToolProvider(provider_id="", client=client)

    def test_zero_timeout_rejected(self):
        client = FakeMCPClient()
        with pytest.raises(ValueError, match="default_timeout_seconds"):
            MCPToolProvider(
                provider_id="demo",
                client=client,
                default_timeout_seconds=0,
            )

    def test_negative_timeout_rejected(self):
        client = FakeMCPClient()
        with pytest.raises(ValueError, match="default_timeout_seconds"):
            MCPToolProvider(
                provider_id="demo",
                client=client,
                default_timeout_seconds=-1,
            )


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_basic_discovery(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[
                        FakeTool(
                            name="echo",
                            description="Echo text",
                            input_schema={
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"],
                                "additionalProperties": False,
                            },
                        ),
                    ],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()
        result = _run(provider.discover_into(registry))

        assert result.provider_id == "demo"
        assert result.registered_tool_names == ("mcp.demo.echo",)
        assert result.tool_name_map == (("echo", "mcp.demo.echo"),)
        assert registry.contains("mcp.demo.echo")

    def test_discovery_with_empty_description_fallback(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[
                        FakeTool(name="echo", description=None),
                    ],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()
        _run(provider.discover_into(registry))

        spec = registry.get("mcp.demo.echo").spec
        assert spec.description == "MCP tool 'echo' from provider 'demo'"

    def test_discovery_with_empty_string_description_fallback(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[
                        FakeTool(name="echo", description="  "),
                    ],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()
        _run(provider.discover_into(registry))

        spec = registry.get("mcp.demo.echo").spec
        assert spec.description == "MCP tool 'echo' from provider 'demo'"

    def test_pagination_multi_page(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[FakeTool(name="tool_a")],
                    next_cursor="cursor-2",
                ),
                FakeListToolsResult(
                    tools=[FakeTool(name="tool_b")],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()
        result = _run(provider.discover_into(registry))

        assert set(result.registered_tool_names) == {"mcp.demo.tool_a", "mcp.demo.tool_b"}
        assert len(client.list_tools_calls) == 2
        assert client.list_tools_calls[0] is None
        assert client.list_tools_calls[1] == "cursor-2"

    def test_pagination_stops_on_none_cursor(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[FakeTool(name="only")],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()
        _run(provider.discover_into(registry))

        assert len(client.list_tools_calls) == 1

    def test_source_metadata_mcp(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[FakeTool(name="echo")],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()
        _run(provider.discover_into(registry))

        spec = registry.get("mcp.demo.echo").spec
        assert spec.source == ToolSource.MCP
        assert spec.provider_id == "demo"

    def test_untrusted_annotations_default_side_effecting(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[
                        FakeTool(
                            name="echo",
                            annotations=FakeToolAnnotations(read_only_hint=True),
                        ),
                    ],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(
            provider_id="demo",
            client=client,
            trust_tool_annotations=False,
        )
        registry = ToolRegistry()
        _run(provider.discover_into(registry))

        spec = registry.get("mcp.demo.echo").spec
        assert spec.side_effect == ToolSideEffect.SIDE_EFFECTING

    def test_trusted_readonly_annotation(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[
                        FakeTool(
                            name="echo",
                            annotations=FakeToolAnnotations(read_only_hint=True),
                        ),
                    ],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(
            provider_id="demo",
            client=client,
            trust_tool_annotations=True,
        )
        registry = ToolRegistry()
        _run(provider.discover_into(registry))

        spec = registry.get("mcp.demo.echo").spec
        assert spec.side_effect == ToolSideEffect.READ_ONLY

    def test_trusted_idempotent_annotation(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[
                        FakeTool(
                            name="echo",
                            annotations=FakeToolAnnotations(
                                read_only_hint=False,
                                idempotent_hint=True,
                            ),
                        ),
                    ],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(
            provider_id="demo",
            client=client,
            trust_tool_annotations=True,
        )
        registry = ToolRegistry()
        _run(provider.discover_into(registry))

        spec = registry.get("mcp.demo.echo").spec
        assert spec.side_effect == ToolSideEffect.IDEMPOTENT

    def test_missing_annotations_side_effecting(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[FakeTool(name="echo", annotations=None)],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(
            provider_id="demo",
            client=client,
            trust_tool_annotations=True,
        )
        registry = ToolRegistry()
        _run(provider.discover_into(registry))

        spec = registry.get("mcp.demo.echo").spec
        assert spec.side_effect == ToolSideEffect.SIDE_EFFECTING

    def test_default_retry_policy_max_attempts_1(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[FakeTool(name="echo")],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()
        _run(provider.discover_into(registry))

        spec = registry.get("mcp.demo.echo").spec
        assert spec.retry_policy.max_attempts == 1

    def test_custom_retry_policy_resolver(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[FakeTool(name="echo")],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(
            provider_id="demo",
            client=client,
            retry_policy_resolver=lambda tool: RetryPolicy(max_attempts=3),
        )
        registry = ToolRegistry()
        _run(provider.discover_into(registry))

        spec = registry.get("mcp.demo.echo").spec
        assert spec.retry_policy.max_attempts == 3

    def test_custom_timeout_resolver(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[FakeTool(name="echo")],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(
            provider_id="demo",
            client=client,
            default_timeout_seconds=60.0,
            timeout_resolver=lambda tool: 5.0,
        )
        registry = ToolRegistry()
        _run(provider.discover_into(registry))

        spec = registry.get("mcp.demo.echo").spec
        assert spec.timeout_seconds == 5.0

    def test_permission_resolver(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[FakeTool(name="echo")],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(
            provider_id="demo",
            client=client,
            permission_resolver=lambda tool: frozenset({"mcp:demo"}),
        )
        registry = ToolRegistry()
        _run(provider.discover_into(registry))

        spec = registry.get("mcp.demo.echo").spec
        assert spec.required_permissions == frozenset({"mcp:demo"})


# ---------------------------------------------------------------------------
# Schema compatibility tests (Section 23-24)
# ---------------------------------------------------------------------------


class TestSchemaCompatibility:
    def test_supported_schema_accepted(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[
                        FakeTool(
                            name="echo",
                            input_schema={
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "count": {"type": "integer"},
                                },
                                "required": ["text"],
                                "additionalProperties": False,
                            },
                        ),
                    ],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()
        result = _run(provider.discover_into(registry))
        assert "mcp.demo.echo" in result.registered_tool_names

    def test_one_of_rejected(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[
                        FakeTool(
                            name="complex",
                            input_schema={
                                "type": "object",
                                "properties": {},
                                "oneOf": [{"type": "string"}, {"type": "integer"}],
                            },
                        ),
                    ],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()
        with pytest.raises(MCPToolCompatibilityError):
            _run(provider.discover_into(registry))

    def test_any_of_rejected(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[
                        FakeTool(
                            name="complex",
                            input_schema={
                                "type": "object",
                                "properties": {},
                                "anyOf": [{"type": "string"}],
                            },
                        ),
                    ],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()
        with pytest.raises(MCPToolCompatibilityError):
            _run(provider.discover_into(registry))

    def test_ref_rejected(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[
                        FakeTool(
                            name="complex",
                            input_schema={
                                "type": "object",
                                "properties": {
                                    "x": {"$ref": "#/$defs/foo"},
                                },
                            },
                        ),
                    ],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()
        with pytest.raises(MCPToolCompatibilityError):
            _run(provider.discover_into(registry))

    def test_defs_rejected(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[
                        FakeTool(
                            name="complex",
                            input_schema={
                                "type": "object",
                                "properties": {},
                                "$defs": {"foo": {"type": "string"}},
                            },
                        ),
                    ],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()
        with pytest.raises(MCPToolCompatibilityError):
            _run(provider.discover_into(registry))

    def test_nested_array_items_supported(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[
                        FakeTool(
                            name="list_tool",
                            input_schema={
                                "type": "object",
                                "properties": {
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "additionalProperties": True,
                            },
                        ),
                    ],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()
        result = _run(provider.discover_into(registry))
        assert "mcp.demo.list_tool" in result.registered_tool_names

    def test_incompatible_schema_no_partial_registration(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[
                        FakeTool(name="good"),
                        FakeTool(
                            name="bad",
                            input_schema={
                                "type": "object",
                                "properties": {},
                                "oneOf": [{"type": "string"}],
                            },
                        ),
                    ],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()
        with pytest.raises(MCPToolCompatibilityError):
            _run(provider.discover_into(registry))
        assert not registry.contains("mcp.demo.good")
        assert not registry.contains("mcp.demo.bad")


# ---------------------------------------------------------------------------
# Discovery atomicity / collision tests (Section 46-48)
# ---------------------------------------------------------------------------


class TestDiscoveryAtomicity:
    def test_duplicate_remote_names_rejected(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[
                        FakeTool(name="echo"),
                        FakeTool(name="echo"),
                    ],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()
        with pytest.raises(MCPDiscoveryError, match="duplicate"):
            _run(provider.discover_into(registry))

    def test_registry_collision_rejected(self):
        existing_spec = ToolSpec(
            name="mcp.demo.echo",
            description="pre-existing",
            input_schema=_simple_schema(),
            required_permissions=frozenset(),
            timeout_seconds=10.0,
            side_effect=ToolSideEffect.SIDE_EFFECTING,
            source=ToolSource.MCP,
            provider_id="demo",
        )
        registry = ToolRegistry()
        registry.register(
            existing_spec,
            handler=lambda args: None,  # type: ignore[arg-type]
        )

        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[FakeTool(name="echo")],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        with pytest.raises(MCPDiscoveryError, match="already contains"):
            _run(provider.discover_into(registry))

    def test_collision_no_partial_registration(self):
        existing_spec = ToolSpec(
            name="mcp.demo.echo",
            description="pre-existing",
            input_schema=_simple_schema(),
            required_permissions=frozenset(),
            timeout_seconds=10.0,
            side_effect=ToolSideEffect.SIDE_EFFECTING,
            source=ToolSource.MCP,
            provider_id="demo",
        )
        registry = ToolRegistry()
        registry.register(
            existing_spec,
            handler=lambda args: None,  # type: ignore[arg-type]
        )

        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[
                        FakeTool(name="other"),
                        FakeTool(name="echo"),
                    ],
                    next_cursor=None,
                ),
            ],
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        with pytest.raises(MCPDiscoveryError):
            _run(provider.discover_into(registry))
        assert not registry.contains("mcp.demo.other")


# ---------------------------------------------------------------------------
# Handler adapter tests (Section 33-37, 39-40)
# ---------------------------------------------------------------------------


class TestHandlerAdapter:
    def test_handler_calls_remote_name_not_namespaced(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[FakeTool(name="echo")],
                    next_cursor=None,
                ),
            ],
            call_tool_results={
                "echo": FakeCallToolResult(
                    content=[FakeTextContent(text="hello")],
                    structured_content=None,
                ),
            },
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()
        _run(provider.discover_into(registry))

        registered = registry.get("mcp.demo.echo")
        output = _run(registered.handler({"text": "hello"}))

        assert output == "hello"
        assert client.call_tool_calls[0][0] == "echo"

    def test_handler_structured_content_preferred(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[FakeTool(name="add")],
                    next_cursor=None,
                ),
            ],
            call_tool_results={
                "add": FakeCallToolResult(
                    content=[FakeTextContent(text='{"sum": 7}')],
                    structured_content={"sum": 7},
                ),
            },
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()
        _run(provider.discover_into(registry))

        registered = registry.get("mcp.demo.add")
        output = _run(registered.handler({"a": 3, "b": 4}))
        assert output == {"sum": 7}

    def test_handler_is_error_raises_execution(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[FakeTool(name="failing")],
                    next_cursor=None,
                ),
            ],
            call_tool_results={
                "failing": FakeCallToolResult(
                    content=[FakeTextContent(text="tool failed")],
                    structured_content=None,
                    is_error=True,
                ),
            },
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()
        _run(provider.discover_into(registry))

        registered = registry.get("mcp.demo.failing")
        with pytest.raises(ToolReportedFailure) as exc_info:
            _run(registered.handler({}))
        assert exc_info.value.error_type == ToolErrorType.EXECUTION
        assert "tool failed" in exc_info.value.message

    def test_handler_unsupported_content_raises_execution(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[FakeTool(name="image_tool")],
                    next_cursor=None,
                ),
            ],
            call_tool_results={
                "image_tool": FakeCallToolResult(
                    content=[FakeImageContent(data="abc")],
                    structured_content=None,
                    is_error=False,
                ),
            },
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()
        _run(provider.discover_into(registry))

        registered = registry.get("mcp.demo.image_tool")
        with pytest.raises(ToolReportedFailure) as exc_info:
            _run(registered.handler({}))
        assert exc_info.value.error_type == ToolErrorType.EXECUTION

    def test_handler_client_exception_propagates(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[FakeTool(name="echo")],
                    next_cursor=None,
                ),
            ],
            call_tool_exceptions={
                "echo": RuntimeError("connection lost"),
            },
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()
        _run(provider.discover_into(registry))

        registered = registry.get("mcp.demo.echo")
        with pytest.raises(RuntimeError, match="connection lost"):
            _run(registered.handler({"text": "hello"}))

    def test_handler_cancelled_error_propagates(self):
        client = FakeMCPClient(
            list_tools_pages=[
                FakeListToolsResult(
                    tools=[FakeTool(name="echo")],
                    next_cursor=None,
                ),
            ],
            call_tool_exceptions={
                "echo": asyncio.CancelledError(),
            },
        )
        provider = MCPToolProvider(provider_id="demo", client=client)
        registry = ToolRegistry()
        _run(provider.discover_into(registry))

        registered = registry.get("mcp.demo.echo")
        with pytest.raises(asyncio.CancelledError):
            _run(registered.handler({"text": "hello"}))


# ---------------------------------------------------------------------------
# ToolSource metadata tests (Section 60)
# ---------------------------------------------------------------------------


class TestToolSourceMetadata:
    def test_local_default_source(self):
        spec = ToolSpec(
            name="local_echo",
            description="echo",
            input_schema=_simple_schema(),
            required_permissions=frozenset(),
            timeout_seconds=10.0,
            side_effect=ToolSideEffect.READ_ONLY,
        )
        assert spec.source == ToolSource.LOCAL
        assert spec.provider_id is None

    def test_mcp_source_requires_provider_id(self):
        with pytest.raises(ValueError, match="provider_id"):
            ToolSpec(
                name="mcp.demo.echo",
                description="echo",
                input_schema=_simple_schema(),
                required_permissions=frozenset(),
                timeout_seconds=10.0,
                side_effect=ToolSideEffect.SIDE_EFFECTING,
                source=ToolSource.MCP,
                provider_id=None,
            )

    def test_local_source_rejects_provider_id(self):
        with pytest.raises(ValueError, match="provider_id"):
            ToolSpec(
                name="local_echo",
                description="echo",
                input_schema=_simple_schema(),
                required_permissions=frozenset(),
                timeout_seconds=10.0,
                side_effect=ToolSideEffect.READ_ONLY,
                source=ToolSource.LOCAL,
                provider_id="demo",
            )

    def test_mcp_source_with_provider_id(self):
        spec = ToolSpec(
            name="mcp.demo.echo",
            description="echo",
            input_schema=_simple_schema(),
            required_permissions=frozenset(),
            timeout_seconds=10.0,
            side_effect=ToolSideEffect.SIDE_EFFECTING,
            source=ToolSource.MCP,
            provider_id="demo",
        )
        assert spec.source == ToolSource.MCP
        assert spec.provider_id == "demo"


# ---------------------------------------------------------------------------
# Artifact zero-token invariant (Section 68)
# ---------------------------------------------------------------------------


class TestArtifactZeroTokenInvariant:
    def test_empty_content_zero_tokens_allowed(self):
        from datetime import datetime, timezone
        from storage.artifact_store import Artifact, ArtifactKind

        a = Artifact.create(
            artifact_id="A-001",
            kind=ArtifactKind.TOOL_OUTPUT,
            content="",
            estimated_tokens=0,
            created_at=datetime.now(timezone.utc),
            source_tool_name="t",
        )
        assert a.estimated_tokens == 0

    def test_non_empty_content_zero_tokens_rejected(self):
        from datetime import datetime, timezone
        from storage.artifact_store import Artifact, ArtifactKind

        with pytest.raises(ValueError, match="estimated_tokens must be >= 1"):
            Artifact.create(
                artifact_id="A-001",
                kind=ArtifactKind.TOOL_OUTPUT,
                content="non-empty",
                estimated_tokens=0,
                created_at=datetime.now(timezone.utc),
                source_tool_name="t",
            )

    def test_non_empty_content_positive_tokens_allowed(self):
        from datetime import datetime, timezone
        from storage.artifact_store import Artifact, ArtifactKind

        a = Artifact.create(
            artifact_id="A-001",
            kind=ArtifactKind.TOOL_OUTPUT,
            content="non-empty",
            estimated_tokens=5,
            created_at=datetime.now(timezone.utc),
            source_tool_name="t",
        )
        assert a.estimated_tokens == 5


# ---------------------------------------------------------------------------
# validate_schema tests (Section 23)
# ---------------------------------------------------------------------------


class TestValidateSchema:
    def test_valid_simple_schema_accepted(self):
        from tools.validator import SimpleToolArgumentValidator

        validator = SimpleToolArgumentValidator()
        validator.validate_schema({
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        })

    def test_nested_object_schema_accepted(self):
        from tools.validator import SimpleToolArgumentValidator

        validator = SimpleToolArgumentValidator()
        validator.validate_schema({
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "additionalProperties": True,
                },
            },
            "additionalProperties": True,
        })

    def test_array_items_schema_accepted(self):
        from tools.validator import SimpleToolArgumentValidator

        validator = SimpleToolArgumentValidator()
        validator.validate_schema({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "additionalProperties": True,
        })

    def test_one_of_rejected(self):
        from tools.validator import SchemaError, SimpleToolArgumentValidator

        validator = SimpleToolArgumentValidator()
        with pytest.raises(SchemaError):
            validator.validate_schema({
                "type": "object",
                "properties": {},
                "oneOf": [{"type": "string"}],
            })

    def test_any_of_rejected(self):
        from tools.validator import SchemaError, SimpleToolArgumentValidator

        validator = SimpleToolArgumentValidator()
        with pytest.raises(SchemaError):
            validator.validate_schema({
                "type": "object",
                "properties": {},
                "anyOf": [{"type": "string"}],
            })

    def test_ref_in_property_rejected(self):
        from tools.validator import SchemaError, SimpleToolArgumentValidator

        validator = SimpleToolArgumentValidator()
        with pytest.raises(SchemaError):
            validator.validate_schema({
                "type": "object",
                "properties": {
                    "x": {"$ref": "#/$defs/foo"},
                },
            })

    def test_non_dict_schema_rejected(self):
        from tools.validator import SchemaError, SimpleToolArgumentValidator

        validator = SimpleToolArgumentValidator()
        with pytest.raises(SchemaError):
            validator.validate_schema("not a dict")  # type: ignore[arg-type]

    def test_non_object_top_level_rejected(self):
        from tools.validator import SchemaError, SimpleToolArgumentValidator

        validator = SimpleToolArgumentValidator()
        with pytest.raises(SchemaError):
            validator.validate_schema({"type": "string"})
