"""Benchmark-only logical invocation boundary (Phase 7 Step 1 Fix).

This module provides ``FaultAwareToolInvoker`` — a thin benchmark-specific
boundary that wraps ``ToolRuntime.execute()`` to allocate a logical
invocation identity ONCE before execution, so that all retry attempts
within a single ``ToolRuntime.execute()`` call share the same logical
identity.

Problem it fixes
----------------

Without this boundary, the fault-injecting tool handler (e.g.
``_RunTestsHandler``) would call ``FaultInjector.next_invocation()`` inside
its ``__call__`` method. Because ``ToolRuntime`` invokes the handler once
per retry attempt, each retry would advance the logical invocation
counter, corrupting fault placement:

::

    logical call #1
        attempt 1  ->  next_invocation() -> logical #1  (correct)
        attempt 2  ->  next_invocation() -> logical #2  (WRONG)

Correct model
-------------

::

    Scripted / Agent logical action
            |
            v
    FaultAwareToolInvoker.execute(call, ctx)
            |
            v
    allocate logical invocation identity ONCE  (begin_invocation)
            |
            v
    ToolRuntime.execute(call, ctx)
            |
            v
        attempt 1  ->  current_invocation() -> logical #1
        attempt 2  ->  current_invocation() -> logical #1
        attempt 3  ->  current_invocation() -> logical #1
            |
            v
    clear active invocation  (end_invocation)
            |
            v
    next Agent logical call -> logical #2

Scope
-----

This is NOT a Naive Runner, NOT a Reliable Harness Runner, and NOT a
full benchmark runner. It is only the logical-call boundary for fault
injection accounting. Production ``ToolRuntime`` is unchanged and remains
benchmark-agnostic.
"""

from __future__ import annotations

from typing import Optional

from evaluation.faults import FaultInjector
from tools import (
    ToolCall,
    ToolExecutionContext,
    ToolExecutionResult,
    ToolRuntime,
)


class FaultAwareToolInvoker:
    """Thin benchmark-only boundary that allocates a logical invocation
    identity exactly once per ``ToolRuntime.execute()`` call.

    When an injector is present, ``begin_invocation(tool_name)`` is called
    before delegating to ``ToolRuntime.execute()``, and
    ``end_invocation(tool_name)`` is called in a ``finally`` block
    afterwards. This ensures all retry attempts within that execution
    share the same logical invocation index.

    When no injector is present (``injector=None``), this is a pure
    pass-through to ``ToolRuntime.execute()`` with no overhead.

    This class does NOT add benchmark-specific behavior to
    ``ToolRuntime``. It wraps it from the outside.
    """

    def __init__(
        self,
        runtime: ToolRuntime,
        injector: Optional[FaultInjector] = None,
    ) -> None:
        if not isinstance(runtime, ToolRuntime):
            raise ValueError("runtime must be a ToolRuntime")
        if injector is not None and not isinstance(injector, FaultInjector):
            raise ValueError("injector must be a FaultInjector or None")
        self._runtime = runtime
        self._injector = injector

    @property
    def runtime(self) -> ToolRuntime:
        """The underlying ToolRuntime (for direct access when needed)."""
        return self._runtime

    @property
    def injector(self) -> Optional[FaultInjector]:
        """The fault injector, or None if no fault injection is active."""
        return self._injector

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        """Execute a logical tool call, allocating the logical invocation
        identity exactly once before delegating to ToolRuntime.

        All retry attempts inside ``ToolRuntime.execute()`` share the
        same logical invocation identity allocated here.
        """
        if self._injector is None:
            return await self._runtime.execute(call, context)

        self._injector.begin_invocation(call.tool_name)
        try:
            return await self._runtime.execute(call, context)
        finally:
            self._injector.end_invocation(call.tool_name)
