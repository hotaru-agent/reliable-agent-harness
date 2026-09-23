"""Deterministic fault injection model (Phase 7 Step 1).

Provides a deterministic, reproducible fault injection abstraction that
targets LOGICAL tool invocations (not retry attempts). The same
``FaultPlan`` always produces the same fault sequence for the same
invocation order.

Key distinction:
- ``logical_invocation`` — the Nth time a tool is called by the agent
  (one logical action = one logical invocation, even if retried).
- ``attempt`` — a single handler execution within the ToolRuntime retry
  loop (one logical invocation may have multiple attempts).

FaultPlan targets logical invocations. If attempt-level control is
needed in the future, a separate mechanism should be introduced with
explicit naming.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from tools.errors import ToolReportedFailure
from tools.models import ToolErrorType


# ---------------------------------------------------------------------------
# Fault types
# ---------------------------------------------------------------------------


class FaultType(str, Enum):
    """Types of faults that can be injected into tool invocations.

    * ``TRANSIENT_FAILURE`` — the tool invocation fails with a
      ``ToolReportedFailure(TRANSIENT)``. A subsequent invocation (or
      retry) may succeed.
    * ``PERMANENT_FAILURE`` — the tool invocation fails with a
      ``ToolReportedFailure(PERMANENT)``. No retry will help.
    * ``TIMEOUT`` — the tool handler sleeps longer than the tool's
      timeout, causing the ToolRuntime to produce a TIMEOUT error.
    * ``LARGE_OUTPUT`` — the tool returns a deterministic large payload
      instead of its normal output.
    * ``PROCESS_INTERRUPTION`` — raises ``InjectedProcessInterruption``,
      a control signal distinct from ToolError. The benchmark runner
      interprets this as "process terminated between checkpoints."
    """

    TRANSIENT_FAILURE = "transient_failure"
    PERMANENT_FAILURE = "permanent_failure"
    TIMEOUT = "timeout"
    LARGE_OUTPUT = "large_output"
    PROCESS_INTERRUPTION = "process_interruption"


# ---------------------------------------------------------------------------
# Process interruption control signal
# ---------------------------------------------------------------------------


class InjectedProcessInterruption(BaseException):
    """Control signal raised by PROCESS_INTERRUPTION fault injection.

    This is NOT a ``ToolError`` and NOT a ``ToolReportedFailure``. It is
    a ``BaseException`` (not ``Exception``) so it is semantically
    distinct from ordinary tool failures and will not be caught by
    generic ``except Exception`` handlers.

    The benchmark runner interprets this as "the process was terminated
    between checkpoints" and should trigger resume-from-checkpoint
    logic in a future phase.

    This is NOT a real OS kill / machine crash. It is a deterministic
    in-process control signal.
    """


# ---------------------------------------------------------------------------
# Fault specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaultSpec:
    """Specification of a single fault to inject.

    Attributes:
        fault_type: The type of fault.
        tool_name: The tool to inject the fault into. If ``"*"``, the
            fault applies to any tool.
        logical_invocation: The 1-based index of the logical invocation
            to fault. Must be >= 1.
        large_output_size: For ``LARGE_OUTPUT`` faults, the size in
            bytes of the deterministic payload. Ignored for other types.
        timeout_sleep_seconds: For ``TIMEOUT`` faults, how long the
            handler should sleep. Should be larger than the tool's
            timeout_seconds. Ignored for other types.
    """

    fault_type: FaultType
    tool_name: str = "*"
    logical_invocation: int = 1
    large_output_size: int = 4096
    timeout_sleep_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.fault_type, FaultType):
            raise ValueError("fault_type must be a FaultType")
        if not self.tool_name or not self.tool_name.strip():
            raise ValueError("tool_name must be a non-empty str")
        if (
            not isinstance(self.logical_invocation, int)
            or isinstance(self.logical_invocation, bool)
            or self.logical_invocation < 1
        ):
            raise ValueError("logical_invocation must be an int >= 1")
        if (
            not isinstance(self.large_output_size, int)
            or isinstance(self.large_output_size, bool)
            or self.large_output_size < 1
        ):
            raise ValueError("large_output_size must be an int >= 1")
        if (
            not isinstance(self.timeout_sleep_seconds, (int, float))
            or self.timeout_sleep_seconds <= 0
        ):
            raise ValueError("timeout_sleep_seconds must be a number > 0")


# ---------------------------------------------------------------------------
# Fault plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaultPlan:
    """Immutable plan of faults to inject during a benchmark run.

    A ``FaultPlan`` is frozen and immutable. The ``FaultInjector``
    maintains per-run mutable counters, so the same plan can be reused
    across multiple runs without cross-run contamination.

    Attributes:
        plan_id: Stable identifier for this plan.
        faults: Tuple of ``FaultSpec`` instances, in deterministic order.
    """

    plan_id: str
    faults: tuple[FaultSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.plan_id or not self.plan_id.strip():
            raise ValueError("plan_id must be a non-empty str")
        if not isinstance(self.faults, tuple):
            raise ValueError("faults must be a tuple")

    @classmethod
    def no_faults(cls, plan_id: str = "none") -> FaultPlan:
        """Return a plan with no faults (clean run)."""
        return cls(plan_id=plan_id, faults=())


# ---------------------------------------------------------------------------
# Fault injector
# ---------------------------------------------------------------------------


class FaultInjector:
    """Per-run fault injection controller.

    The injector maintains mutable per-run invocation counters. Each
    call to ``maybe_inject`` checks whether a fault should be fired for
    the given tool at the current logical invocation count.

    IMPORTANT: The injector counts LOGICAL invocations, not retry
    attempts. One logical invocation may have multiple retry attempts
    inside the ToolRuntime; the injector fires at most once per logical
    invocation.

    Usage pattern (inside a benchmark tool handler)::

        injector.maybe_inject(tool_name="run_tests",
                              logical_invocation=injector.next_invocation("run_tests"))
        # if no fault fires, proceed with normal handler logic

    The injector is created fresh for each benchmark run from the
    immutable ``FaultPlan``. This ensures deterministic reset.
    """

    def __init__(self, plan: FaultPlan) -> None:
        if not isinstance(plan, FaultPlan):
            raise ValueError("plan must be a FaultPlan")
        self._plan = plan
        # Per-tool logical invocation counters.
        self._invocation_counts: dict[str, int] = {}
        # Track which faults have been fired (by index).
        self._fired_faults: set[int] = set()
        # Active logical invocation per tool (set by begin_invocation,
        # cleared by end_invocation). All retry attempts within one
        # logical invocation share the same identity.
        self._active_invocations: dict[str, int] = {}

    @property
    def plan(self) -> FaultPlan:
        return self._plan

    @property
    def fired_fault_indices(self) -> tuple[int, ...]:
        """Indices of faults that have been fired, in sorted order."""
        return tuple(sorted(self._fired_faults))

    def begin_invocation(self, tool_name: str) -> int:
        """Allocate a logical invocation identity ONCE for a tool call.

        This must be called BEFORE ``ToolRuntime.execute()`` so that all
        retry attempts within that execution share the same logical
        invocation index. The identity remains active until
        ``end_invocation`` is called.

        Returns the 1-based logical invocation index.
        """
        if not tool_name or not tool_name.strip():
            raise ValueError("tool_name must be a non-empty str")
        idx = self._invocation_counts.get(tool_name, 0) + 1
        self._invocation_counts[tool_name] = idx
        self._active_invocations[tool_name] = idx
        return idx

    def end_invocation(self, tool_name: str) -> None:
        """Clear the active logical invocation for ``tool_name``.

        Must be called AFTER ``ToolRuntime.execute()`` returns (or
        raises). This ends the logical invocation scope so the next
        ``begin_invocation`` allocates a new index.
        """
        if not tool_name or not tool_name.strip():
            raise ValueError("tool_name must be a non-empty str")
        self._active_invocations.pop(tool_name, None)

    def current_invocation(self, tool_name: str) -> int:
        """Return the active logical invocation index for ``tool_name``.

        This is the identity allocated by ``begin_invocation``. All retry
        attempts within the same logical invocation see the same index.

        Returns 0 if no invocation is currently active (e.g. the handler
        was called without going through the invoker boundary). In that
        case, ``maybe_inject`` with index 0 will not match any fault
        (faults are configured for index >= 1), so the handler proceeds
        normally — a safe failure mode.
        """
        if not tool_name or not tool_name.strip():
            raise ValueError("tool_name must be a non-empty str")
        return self._active_invocations.get(tool_name, 0)

    def next_invocation(self, tool_name: str) -> int:
        """Increment and return the logical invocation count for
        ``tool_name``.

        .. deprecated::
            This method is retained for direct unit testing of
            ``FaultInjector`` in isolation. Benchmark tool handlers
            should use ``begin_invocation`` / ``current_invocation`` /
            ``end_invocation`` via ``FaultAwareToolInvoker`` to ensure
            retry attempts do not advance the logical invocation counter.

        Returns the 1-based invocation index.
        """
        if not tool_name or not tool_name.strip():
            raise ValueError("tool_name must be a non-empty str")
        self._invocation_counts[tool_name] = (
            self._invocation_counts.get(tool_name, 0) + 1
        )
        return self._invocation_counts[tool_name]

    def peek_invocation(self, tool_name: str) -> int:
        """Return the current logical invocation count for ``tool_name``
        WITHOUT incrementing.

        Returns the 1-based invocation index that the NEXT call to
        ``next_invocation`` will produce.
        """
        return self._invocation_counts.get(tool_name, 0) + 1

    def find_fault(
        self,
        tool_name: str,
        logical_invocation: int,
    ) -> Optional[FaultSpec]:
        """Find the fault spec matching the given tool and invocation,
        or None if no fault is scheduled.

        A fault matches if:
        - ``spec.tool_name == tool_name`` or ``spec.tool_name == "*"``
        - ``spec.logical_invocation == logical_invocation``
        - The fault has not already been fired.

        If multiple faults match, the first one (by tuple order) is
        returned.
        """
        for i, spec in enumerate(self._plan.faults):
            if i in self._fired_faults:
                continue
            if spec.tool_name not in ("*", tool_name):
                continue
            if spec.logical_invocation != logical_invocation:
                continue
            self._fired_faults.add(i)
            return spec
        return None

    async def maybe_inject(
        self,
        tool_name: str,
        logical_invocation: int,
        *,
        sleep_fn: Any = asyncio.sleep,
    ) -> Optional[FaultSpec]:
        """Check for a fault at the given invocation and inject it if
        found.

        If a fault is found, it is executed (exception raised, sleep
        performed, or large output returned). If no fault is found,
        returns None and the caller should proceed normally.

        For ``LARGE_OUTPUT``, the caller is responsible for returning
        the generated payload — ``maybe_inject`` returns the ``FaultSpec``
        so the caller can call ``generate_large_output(spec)``.

        For ``TRANSIENT_FAILURE`` / ``PERMANENT_FAILURE`` /
        ``PROCESS_INTERRUPTION``, the exception is raised directly.

        For ``TIMEOUT``, the handler sleeps for
        ``spec.timeout_sleep_seconds``, which should exceed the tool's
        timeout. The ToolRuntime will cancel the handler and produce a
        TIMEOUT error.

        Returns the ``FaultSpec`` if a fault was found (and not one that
        raises), None if no fault.
        """
        spec = self.find_fault(tool_name, logical_invocation)
        if spec is None:
            return None

        if spec.fault_type is FaultType.TRANSIENT_FAILURE:
            raise ToolReportedFailure(
                error_type=ToolErrorType.TRANSIENT,
                message="injected transient failure",
            )
        if spec.fault_type is FaultType.PERMANENT_FAILURE:
            raise ToolReportedFailure(
                error_type=ToolErrorType.PERMANENT,
                message="injected permanent failure",
            )
        if spec.fault_type is FaultType.PROCESS_INTERRUPTION:
            raise InjectedProcessInterruption(
                "injected process interruption"
            )
        if spec.fault_type is FaultType.TIMEOUT:
            await sleep_fn(spec.timeout_sleep_seconds)
            # If we get here, the timeout didn't fire (e.g. no timeout
            # configured). Return normally — the caller handles it.
            return spec
        if spec.fault_type is FaultType.LARGE_OUTPUT:
            return spec

        raise ValueError(f"unknown fault type: {spec.fault_type}")

    @staticmethod
    def generate_large_output(spec: FaultSpec) -> str:
        """Generate a deterministic large output payload for
        ``LARGE_OUTPUT`` faults.

        The payload is a repeating pattern of known content, sized to
        ``spec.large_output_size`` bytes. It is deterministic: the same
        spec always produces the same payload.
        """
        if spec.fault_type is not FaultType.LARGE_OUTPUT:
            raise ValueError("generate_large_output requires LARGE_OUTPUT fault type")
        pattern = "BENCHMARK_LARGE_OUTPUT_"
        payload = (pattern * (spec.large_output_size // len(pattern) + 1))
        return payload[:spec.large_output_size]
