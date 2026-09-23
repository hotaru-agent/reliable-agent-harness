"""MCP Tool Provider Adapter (Phase 5 Step 1).

Adapts tools discovered from an MCP server (via the official MCP Python
SDK v2) into the harness ``ToolRegistry`` so that ``ToolRuntime`` can
execute them with the same validation, permission, timeout, retry and
side-effect governance as local Python tools.

Architecture:

    MCP Server
        |
        v
    MCP Client (owned by caller)
        |
        v
    MCPToolProvider.discover_into(registry)
        |
        v
    ToolRegistry  (local + MCP tools coexist)
        |
        v
    ToolRuntime.execute()  (unified boundary)

The provider does NOT own the MCP client lifecycle. The caller is
responsible for connecting and disconnecting the client (e.g. via
``async with Client(...) as client:``).

The provider does NOT implement its own timeout, retry, or
side-effect engine — those concerns remain with ``ToolRuntime``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Protocol, Sequence

from opentelemetry.trace import Tracer
from opentelemetry.trace.status import Status, StatusCode

from mcp_adapter.errors import MCPDiscoveryError, MCPToolCompatibilityError
from observability.tracing import (
    ATTR_MCP_ANNOTATION_TRUSTED,
    ATTR_MCP_DISCOVERY_PAGE_COUNT,
    ATTR_MCP_DISCOVERY_REGISTERED_COUNT,
    ATTR_MCP_DISCOVERY_TOOL_COUNT,
    ATTR_MCP_PROVIDER_ID,
    SPAN_MCP_DISCOVER,
    resolve_tracer,
    safe_record_exception,
    safe_set_attribute,
    safe_set_status,
    safe_span,
)
from tools.errors import ToolReportedFailure
from tools.models import (
    RetryPolicy,
    ToolErrorType,
    ToolHandler,
    ToolSideEffect,
    ToolSource,
    ToolSpec,
)
from tools.registry import ToolRegistry
from tools.validator import SimpleToolArgumentValidator


# ---------------------------------------------------------------------------
# MCP client protocol boundary
# ---------------------------------------------------------------------------


class MCPClientProtocol(Protocol):
    """Minimal client surface the provider depends on.

    The official ``mcp.Client`` satisfies this conceptually. Unit tests
    may use a fake client that implements these two async methods.
    """

    async def list_tools(self, *, cursor: Optional[str] = None) -> Any:
        """Return a ``ListToolsResult``-like object with ``tools`` and
        ``next_cursor`` attributes."""
        ...

    async def call_tool(
        self,
        name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Return a ``CallToolResult``-like object with ``content``,
        ``structured_content`` and ``is_error`` attributes."""
        ...


# ---------------------------------------------------------------------------
# Discovery result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCPDiscoveryResult:
    """Result of a successful MCP tool discovery.

    ``registered_tool_names`` is the tuple of namespaced names that were
    registered into the target ``ToolRegistry``, in discovery order.

    ``tool_name_map`` maps remote MCP tool names to the namespaced
    registry names, so callers can trace which remote tool a registry
    entry came from.
    """

    provider_id: str
    registered_tool_names: tuple[str, ...]
    tool_name_map: tuple[tuple[str, str], ...]


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def make_namespaced_name(provider_id: str, remote_name: str) -> str:
    """Build the deterministic registry name for an MCP tool.

    Format: ``mcp.<provider_id>.<remote_name>``

    Both ``provider_id`` and ``remote_name`` must be non-empty.
    """
    if not provider_id or not str(provider_id).strip():
        raise ValueError("provider_id must be a non-empty str")
    if not remote_name or not str(remote_name).strip():
        raise ValueError("remote_name must be a non-empty str")
    return f"mcp.{provider_id}.{remote_name}"


# ---------------------------------------------------------------------------
# Annotation mapping
# ---------------------------------------------------------------------------


def map_side_effect(
    annotations: Any,
    *,
    trust_tool_annotations: bool,
) -> ToolSideEffect:
    """Map MCP tool annotations to a harness ``ToolSideEffect``.

    Policy (Section 26-28):

    * ``trust_tool_annotations=False`` (default) -> always
      ``SIDE_EFFECTING``. The runtime cannot prove a remote server's
      annotation is truthful.
    * ``trust_tool_annotations=True``:
        * ``readOnlyHint == True`` -> ``READ_ONLY`` (highest priority)
        * else ``idempotentHint == True`` -> ``IDEMPOTENT``
        * else -> ``SIDE_EFFECTING``

    ``destructiveHint`` and ``openWorldHint`` are not currently modeled
    in ``ToolSideEffect`` and are ignored for side-effect mapping.
    """
    if not trust_tool_annotations:
        return ToolSideEffect.SIDE_EFFECTING

    if annotations is None:
        return ToolSideEffect.SIDE_EFFECTING

    read_only = getattr(annotations, "read_only_hint", None)
    if read_only is True:
        return ToolSideEffect.READ_ONLY

    idempotent = getattr(annotations, "idempotent_hint", None)
    if idempotent is True:
        return ToolSideEffect.IDEMPOTENT

    return ToolSideEffect.SIDE_EFFECTING


# ---------------------------------------------------------------------------
# Description fallback
# ---------------------------------------------------------------------------


def fallback_description(provider_id: str, remote_name: str) -> str:
    """Deterministic non-empty description for an MCP tool with no
    description (Section 21)."""
    return f"MCP tool '{remote_name}' from provider '{provider_id}'"


# ---------------------------------------------------------------------------
# Result normalization
# ---------------------------------------------------------------------------


def _extract_text_from_content(content_blocks: Sequence[Any]) -> Optional[str]:
    """Extract text from MCP content blocks.

    Returns the joined text if ALL blocks are TextContent, otherwise
    ``None`` (unsupported content type).
    """
    if not content_blocks:
        return ""

    texts: list[str] = []
    for block in content_blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            texts.append(getattr(block, "text", ""))
        else:
            return None
    return "\n".join(texts)


def normalize_call_result(result: Any, remote_name: str) -> Any:
    """Normalize an MCP ``CallToolResult`` into a handler output value.

    Priority (Section 34-37):

    1. ``is_error == True`` -> raise ``ToolReportedFailure(EXECUTION)``
       with a deterministic message extracted from TextContent, or a
       stable fallback.
    2. ``structured_content is not None`` -> return it directly.
    3. All-TextContent ``content`` -> join with ``"\\n"`` and return.
    4. Unsupported content -> raise ``ToolReportedFailure(EXECUTION)``.

    ``asyncio.CancelledError`` is never raised here; it propagates from
    the client call itself.
    """
    is_error = getattr(result, "is_error", False)
    if is_error:
        content = getattr(result, "content", [])
        text = _extract_text_from_content(content)
        if text:
            message = text
        else:
            message = f"MCP tool '{remote_name}' returned an error."
        raise ToolReportedFailure(
            ToolErrorType.EXECUTION,
            message,
        )

    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured

    content = getattr(result, "content", [])
    text = _extract_text_from_content(content)
    if text is None:
        raise ToolReportedFailure(
            ToolErrorType.EXECUTION,
            f"MCP tool '{remote_name}' returned unsupported non-text content.",
        )
    return text


# ---------------------------------------------------------------------------
# MCP Tool Provider
# ---------------------------------------------------------------------------


class MCPToolProvider:
    """Discovers MCP tools and adapts them into a ``ToolRegistry``.

    The provider does NOT own the MCP client lifecycle. The caller
    connects the client (e.g. ``async with Client(server) as client:``)
    and passes it to the provider. The provider only uses the client
    for ``list_tools`` and ``call_tool`` during discovery and handler
    execution.

    Discovery is atomic (Section 46): all validation and collision
    checks happen BEFORE any registration, so a failure never leaves the
    registry in a partially-modified state.
    """

    def __init__(
        self,
        *,
        provider_id: str,
        client: MCPClientProtocol,
        validator: SimpleToolArgumentValidator | None = None,
        trust_tool_annotations: bool = False,
        default_timeout_seconds: float = 30.0,
        default_retry_policy: RetryPolicy | None = None,
        permission_resolver: Callable[[Any], frozenset[str]] | None = None,
        timeout_resolver: Callable[[Any], float] | None = None,
        retry_policy_resolver: Callable[[Any], RetryPolicy] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        if not provider_id or not str(provider_id).strip():
            raise ValueError("provider_id must be a non-empty str")
        if not isinstance(default_timeout_seconds, (int, float)):
            raise ValueError("default_timeout_seconds must be a number")
        if default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be > 0")

        self._provider_id = provider_id
        self._client = client
        self._validator = validator or SimpleToolArgumentValidator()
        self._trust_tool_annotations = trust_tool_annotations
        self._default_timeout_seconds = default_timeout_seconds
        self._default_retry_policy = default_retry_policy or RetryPolicy(
            max_attempts=1
        )
        self._permission_resolver = permission_resolver
        self._timeout_resolver = timeout_resolver
        self._retry_policy_resolver = retry_policy_resolver
        self._tracer = resolve_tracer(tracer)

    async def discover_into(self, registry: ToolRegistry) -> MCPDiscoveryResult:
        """Discover all MCP tools and register them into ``registry``.

        Atomic (Section 46): validation before mutation. If any tool is
        incompatible or any namespaced name collides with an existing
        registry entry, discovery raises and NO tools are registered.

        Raises:
            MCPToolCompatibilityError: a tool's schema is unsupported.
            MCPDiscoveryError: duplicate remote names or registry
                collision.
        """
        with safe_span(self._tracer, SPAN_MCP_DISCOVER) as span:
            safe_set_attribute(span, ATTR_MCP_PROVIDER_ID, self._provider_id)
            safe_set_attribute(
                span, ATTR_MCP_ANNOTATION_TRUSTED, self._trust_tool_annotations
            )
            try:
                result = await self._discover_into_impl(registry, span)
            except Exception as exc:
                safe_record_exception(span, exc)
                safe_set_status(span, Status(StatusCode.ERROR))
                raise
            safe_set_attribute(
                span, ATTR_MCP_DISCOVERY_TOOL_COUNT, len(result.registered_tool_names)
            )
            safe_set_status(span, Status(StatusCode.OK))
            return result

    async def _discover_into_impl(
        self,
        registry: ToolRegistry,
        span: object,
    ) -> MCPDiscoveryResult:
        """Internal: discovery implementation with an active span."""
        # Phase A: list all tools (paginated).
        remote_tools = await self._list_all_tools(span)

        # Phase B: validate / adapt ALL tools (build specs + handlers).
        adapted: list[tuple[ToolSpec, ToolHandler]] = []
        seen_remote_names: set[str] = set()
        name_map: list[tuple[str, str]] = []
        namespaced_names: list[str] = []

        for tool in remote_tools:
            remote_name = tool.name
            if not remote_name or not str(remote_name).strip():
                raise MCPDiscoveryError(
                    "discovered MCP tool with empty name"
                )
            if remote_name in seen_remote_names:
                raise MCPDiscoveryError(
                    f"duplicate remote tool name {remote_name!r} in discovery"
                )
            seen_remote_names.add(remote_name)

            # Schema compatibility check (Section 23).
            input_schema = tool.input_schema
            if not isinstance(input_schema, dict):
                raise MCPToolCompatibilityError(
                    f"MCP tool {remote_name!r}: input_schema must be a dict"
                )
            try:
                self._validator.validate_schema(input_schema)
            except Exception as exc:
                raise MCPToolCompatibilityError(
                    f"MCP tool {remote_name!r}: incompatible input_schema: {exc}"
                ) from exc

            namespaced = make_namespaced_name(self._provider_id, remote_name)
            spec = self._build_spec(tool, remote_name, namespaced, input_schema)
            handler = self._build_handler(remote_name)
            adapted.append((spec, handler))
            name_map.append((remote_name, namespaced))
            namespaced_names.append(namespaced)

        # Phase C: check registry collisions (preflight).
        for ns_name in namespaced_names:
            if registry.contains(ns_name):
                raise MCPDiscoveryError(
                    f"registry already contains tool {ns_name!r}"
                )

        # Phase D: register all (no failures expected after preflight).
        for spec, handler in adapted:
            registry.register(spec, handler)

        return MCPDiscoveryResult(
            provider_id=self._provider_id,
            registered_tool_names=tuple(namespaced_names),
            tool_name_map=tuple(name_map),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _list_all_tools(self, span: object | None = None) -> list[Any]:
        """Paginated ``list_tools`` (Section 16)."""
        all_tools: list[Any] = []
        cursor: Optional[str] = None
        page_count = 0
        while True:
            page = await self._client.list_tools(cursor=cursor)
            page_tools = getattr(page, "tools", [])
            all_tools.extend(page_tools)
            cursor = getattr(page, "next_cursor", None)
            page_count += 1
            if cursor is None:
                break
        if span is not None:
            safe_set_attribute(span, ATTR_MCP_DISCOVERY_PAGE_COUNT, page_count)
        return all_tools

    def _build_spec(
        self,
        tool: Any,
        remote_name: str,
        namespaced_name: str,
        input_schema: dict[str, Any],
    ) -> ToolSpec:
        """Build a ``ToolSpec`` from an MCP tool definition."""
        description = tool.description
        if not description or not str(description).strip():
            description = fallback_description(self._provider_id, remote_name)

        side_effect = map_side_effect(
            getattr(tool, "annotations", None),
            trust_tool_annotations=self._trust_tool_annotations,
        )

        if self._timeout_resolver is not None:
            timeout_seconds = self._timeout_resolver(tool)
        else:
            timeout_seconds = self._default_timeout_seconds

        if self._retry_policy_resolver is not None:
            retry_policy = self._retry_policy_resolver(tool)
        else:
            retry_policy = self._default_retry_policy

        if self._permission_resolver is not None:
            required_permissions = self._permission_resolver(tool)
        else:
            required_permissions = frozenset()

        return ToolSpec(
            name=namespaced_name,
            description=description,
            input_schema=input_schema,
            required_permissions=required_permissions,
            timeout_seconds=timeout_seconds,
            side_effect=side_effect,
            retry_policy=retry_policy,
            source=ToolSource.MCP,
            provider_id=self._provider_id,
        )

    def _build_handler(self, remote_name: str) -> ToolHandler:
        """Build an async handler that calls the MCP tool via the client."""
        client = self._client

        async def handler(arguments: dict[str, Any]) -> Any:
            result = await client.call_tool(remote_name, arguments)
            return normalize_call_result(result, remote_name)

        return handler
