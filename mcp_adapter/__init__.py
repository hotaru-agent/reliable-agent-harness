"""MCP Tool Provider Adapter (Phase 5 Step 1).

Adapts MCP tools (discovered via the official MCP Python SDK v2) into
the harness ``ToolRegistry`` so ``ToolRuntime`` can execute them with
the same validation, permission, timeout, retry and side-effect
governance as local Python tools.

This module is an *adapter*, not a core runtime dependency. MCP SDK
imports are confined to this package and tests; the harness core
(``harness/*``, ``tools/*``, ``storage/*``) does not import MCP.
"""

from mcp_adapter.errors import MCPDiscoveryError, MCPToolCompatibilityError
from mcp_adapter.provider import (
    MCPClientProtocol,
    MCPDiscoveryResult,
    MCPToolProvider,
    fallback_description,
    make_namespaced_name,
    map_side_effect,
    normalize_call_result,
)

__all__ = [
    "MCPClientProtocol",
    "MCPDiscoveryError",
    "MCPDiscoveryResult",
    "MCPToolCompatibilityError",
    "MCPToolProvider",
    "fallback_description",
    "make_namespaced_name",
    "map_side_effect",
    "normalize_call_result",
]
