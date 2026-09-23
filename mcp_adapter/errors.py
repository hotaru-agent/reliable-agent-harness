"""Errors for the MCP adapter layer (Phase 5 Step 1).

These are configuration / discovery errors, distinct from runtime tool
failures (which flow through ``ToolReportedFailure`` and the
``ToolRuntime`` structured error contract).
"""

from __future__ import annotations


class MCPToolCompatibilityError(Exception):
    """Raised when an MCP tool cannot be adapted to the harness.

    Reasons include:

    * the MCP tool's ``input_schema`` uses JSON Schema keywords the
      harness ``SimpleToolArgumentValidator`` does not support
      (``oneOf`` / ``anyOf`` / ``allOf`` / ``$ref`` / ``$defs`` /
      conditional schemas, etc.);
    * the MCP tool returns content blocks the adapter does not support
      (non-text, non-structured content).

    This is raised at discovery time (schema) or at call time (content)
    so the harness never silently registers or invokes an incompatible
    tool.
    """


class MCPDiscoveryError(Exception):
    """Raised when MCP tool discovery fails before any registration.

    Reasons include:

    * duplicate remote tool names within the same discovery page;
    * a generated namespaced name already exists in the target
      ``ToolRegistry``.

    Discovery is atomic: when this is raised, no partial registration
    has occurred.
    """
