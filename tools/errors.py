"""Errors for the tool runtime layer (Phase 2 Step 1).

Two kinds of errors live here:

* ``ToolReportedFailure`` — the small, controlled exception a Tool
  handler raises to signal a structured TRANSIENT / PERMANENT / EXECUTION
  failure to the runtime. The runtime converts it into a
  ``ToolError`` data object inside a ``ToolExecutionResult``.

* ``DuplicateToolError`` — a configuration / programmer error raised by
  ``ToolRegistry`` when a tool name is registered twice.

All other expected tool failures (not found, validation, permission,
timeout, ordinary handler exception) are produced by the runtime
directly as ``ToolError`` data inside ``ToolExecutionResult`` rather than
as raised exceptions. The only exception that propagates out of
``ToolRuntime.execute`` is ``asyncio.CancelledError`` (external
cancellation), which must never be converted into a structured result.
"""

from __future__ import annotations

from typing import Any, Optional

from tools.models import ToolErrorType


class ToolReportedFailure(Exception):
    """Controlled failure reported by a Tool handler.

    Only ``TRANSIENT``, ``PERMANENT`` and ``EXECUTION`` error types are
    accepted — these are the types a handler is allowed to self-report.
    NOT_FOUND / VALIDATION / PERMISSION / TIMEOUT are produced by the
    runtime itself and must not be raised by handlers.

    The runtime catches this and converts it into a ``ToolError`` data
    object. ``retryable`` is derived from ``error_type`` per the default
    table (see ``tools.models``).
    """

    _ALLOWED = frozenset(
        {ToolErrorType.TRANSIENT, ToolErrorType.PERMANENT, ToolErrorType.EXECUTION}
    )

    def __init__(
        self,
        error_type: ToolErrorType,
        message: str,
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        if error_type not in self._ALLOWED:
            raise ValueError(
                f"ToolReportedFailure error_type must be one of "
                f"{sorted(t.value for t in self._ALLOWED)}, got {error_type.value!r}"
            )
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        super().__init__(message)


class DuplicateToolError(Exception):
    """Raised when a tool name is registered more than once.

    This is a configuration / programmer error, not a runtime tool
    failure. The registry never silently overwrites an existing tool.
    """
