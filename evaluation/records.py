"""Benchmark execution records and events (Phase 7 Step 1).

Provides lightweight, deterministic accounting structures for benchmark
runs. These are NOT OpenTelemetry spans — they are evaluation records
that the benchmark runner produces directly from structured runtime
results.

The benchmark source-of-truth comes from:
- the benchmark runner (which drives execution)
- the repository oracle (which determines completion)
- runtime structured results (ToolExecutionResult, AgentActionResult, etc.)

NOT from trace parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Benchmark event types
# ---------------------------------------------------------------------------


class BenchmarkEventType(str, Enum):
    """Types of events recorded during a benchmark run.

    These are lightweight evaluation accounting events, NOT OpenTelemetry
    spans. They exist for deterministic record-keeping only.
    """

    LOGICAL_ACTION = "logical_action"
    TOOL_INVOCATION = "tool_invocation"
    TOOL_ATTEMPT = "tool_attempt"
    CHECKPOINT = "checkpoint"
    RESUME = "resume"
    LOOP_DETECTED = "loop_detected"
    REPLAN_REQUIRED = "replan_required"
    REPLAN_BLOCKED = "replan_blocked"
    REPLAN_ACKNOWLEDGED = "replan_acknowledged"
    EXTERNALIZATION = "externalization"
    INTERRUPTION = "interruption"
    COMPLETION_CHECK = "completion_check"
    # Phase 7 Step 5 — context growth & large output externalization.
    CONTEXT_ASSEMBLED = "context_assembled"
    CONTEXT_LIMIT_EXCEEDED = "context_limit_exceeded"
    OUTPUT_EXTERNALIZED = "output_externalized"


# ---------------------------------------------------------------------------
# Benchmark event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkEvent:
    """A single deterministic benchmark event.

    Attributes:
        event_type: The type of event.
        sequence: Monotonically increasing event sequence number.
        tool_name: Tool name if relevant, else None.
        success: True if the event represents a success, False if
            failure, None if not applicable.
        details: Small dict of event-specific metadata. Only safe,
            non-sensitive metadata should be stored here (counts,
            indices, error types — NOT tool arguments, outputs, or
            context content).
    """

    event_type: BenchmarkEventType
    sequence: int
    tool_name: Optional[str] = None
    success: Optional[bool] = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, BenchmarkEventType):
            raise ValueError("event_type must be a BenchmarkEventType")
        if (
            not isinstance(self.sequence, int)
            or isinstance(self.sequence, bool)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be an int >= 0")
        if self.tool_name is not None and (
            not isinstance(self.tool_name, str) or not self.tool_name
        ):
            raise ValueError("tool_name must be a non-empty str or None")
        if self.success is not None and not isinstance(self.success, bool):
            raise ValueError("success must be a bool or None")
        if not isinstance(self.details, dict):
            raise ValueError("details must be a dict")


# ---------------------------------------------------------------------------
# Benchmark run record
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRunRecord:
    """Structured execution record for a single benchmark run.

    This is the RAW execution record — it captures what happened during
    a run, not derived metrics. Derived metrics (completion rate,
    recovery success, etc.) are computed from this record by a future
    metric-aggregation phase.

    Only fields that can be reliably computed from the benchmark runner
    and runtime structured results are included. No fake data.

    Attributes:
        scenario_id: The scenario that was run.
        task_id: The task that was attempted.
        completed: True if the oracle declared the task complete.
        logical_action_count: Number of logical agent actions executed.
        tool_invocation_count: Number of tool invocations (logical).
        tool_attempt_count: Total number of handler attempts (including
            retries).
        retry_count: Number of retries that occurred (attempts beyond
            the first per invocation).
        loop_detection_count: Number of times loop detection fired.
        replan_count: Number of replan-required signals.
        checkpoint_count: Number of checkpoints saved.
        resume_count: Number of resume operations.
        duplicate_action_count: Number of duplicate logical actions
            (same tool + same canonical arguments).
        externalized_output_count: Number of tool outputs externalized
            to artifact store.
        processed_output_count: Number of tool outputs successfully
            processed (inline or externalized). Used as the denominator
            for the externalized-output ratio.
        estimated_context_tokens: Estimated context tokens used (if
            context assembly was performed).
        peak_context_tokens: Peak estimated token count at the context
            presented/attempted at an action decision boundary. For
            Naive append-all, this may exceed capacity. For Reliable,
            this must be <= capacity.
        failure_reason: Human-readable failure reason if the run did
            not complete, else None.
        events: Tuple of ``BenchmarkEvent`` instances recorded during
            the run.
        wall_clock_seconds: Wall-clock duration of the run. NOT used
            for deterministic unit-test assertions.
    """

    scenario_id: str
    task_id: str
    completed: bool = False
    logical_action_count: int = 0
    tool_invocation_count: int = 0
    tool_attempt_count: int = 0
    retry_count: int = 0
    loop_detection_count: int = 0
    replan_count: int = 0
    checkpoint_count: int = 0
    resume_count: int = 0
    duplicate_action_count: int = 0
    externalized_output_count: int = 0
    processed_output_count: int = 0
    estimated_context_tokens: int = 0
    peak_context_tokens: int = 0
    failure_reason: Optional[str] = None
    events: tuple[BenchmarkEvent, ...] = ()
    wall_clock_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty str")
        if not self.task_id or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty str")
        if not isinstance(self.completed, bool):
            raise ValueError("completed must be a bool")
        for field_name in (
            "logical_action_count",
            "tool_invocation_count",
            "tool_attempt_count",
            "retry_count",
            "loop_detection_count",
            "replan_count",
            "checkpoint_count",
            "resume_count",
            "duplicate_action_count",
            "externalized_output_count",
            "processed_output_count",
            "estimated_context_tokens",
            "peak_context_tokens",
        ):
            val = getattr(self, field_name)
            if (
                not isinstance(val, int)
                or isinstance(val, bool)
                or val < 0
            ):
                raise ValueError(f"{field_name} must be an int >= 0")
        if self.failure_reason is not None and (
            not isinstance(self.failure_reason, str)
        ):
            raise ValueError("failure_reason must be a str or None")
        if not isinstance(self.events, tuple):
            raise ValueError("events must be a tuple")
        if (
            not isinstance(self.wall_clock_seconds, (int, float))
            or self.wall_clock_seconds < 0
        ):
            raise ValueError("wall_clock_seconds must be a number >= 0")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict snapshot of this record.

        The returned dict is a shallow copy — mutating it does not
        affect the record. Events are included as-is (they are frozen).
        """
        return {
            "scenario_id": self.scenario_id,
            "task_id": self.task_id,
            "completed": self.completed,
            "logical_action_count": self.logical_action_count,
            "tool_invocation_count": self.tool_invocation_count,
            "tool_attempt_count": self.tool_attempt_count,
            "retry_count": self.retry_count,
            "loop_detection_count": self.loop_detection_count,
            "replan_count": self.replan_count,
            "checkpoint_count": self.checkpoint_count,
            "resume_count": self.resume_count,
            "duplicate_action_count": self.duplicate_action_count,
            "externalized_output_count": self.externalized_output_count,
            "processed_output_count": self.processed_output_count,
            "estimated_context_tokens": self.estimated_context_tokens,
            "peak_context_tokens": self.peak_context_tokens,
            "failure_reason": self.failure_reason,
            "wall_clock_seconds": self.wall_clock_seconds,
        }


# ---------------------------------------------------------------------------
# Benchmark result (derived from record)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkResult:
    """Derived evaluation result from a ``BenchmarkRunRecord``.

    This separates raw execution records from derived metrics. The
    ``BenchmarkResult`` is what a future comparison phase would use to
    compare Naive Runner vs Reliable Harness Runner.

    Attributes:
        scenario_id: The scenario that was run.
        task_completed: True if the oracle declared completion.
        steps_to_completion: Logical action count if completed, else None.
        failure_reason: Failure reason if not completed, else None.
    """

    scenario_id: str
    task_completed: bool
    steps_to_completion: Optional[int] = None
    failure_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty str")
        if not isinstance(self.task_completed, bool):
            raise ValueError("task_completed must be a bool")
        if self.steps_to_completion is not None:
            if (
                not isinstance(self.steps_to_completion, int)
                or isinstance(self.steps_to_completion, bool)
                or self.steps_to_completion < 0
            ):
                raise ValueError(
                    "steps_to_completion must be an int >= 0 or None"
                )
        if self.failure_reason is not None and (
            not isinstance(self.failure_reason, str)
        ):
            raise ValueError("failure_reason must be a str or None")

    @classmethod
    def from_record(cls, record: BenchmarkRunRecord) -> BenchmarkResult:
        """Derive a ``BenchmarkResult`` from a ``BenchmarkRunRecord``."""
        return cls(
            scenario_id=record.scenario_id,
            task_completed=record.completed,
            steps_to_completion=(
                record.logical_action_count if record.completed else None
            ),
            failure_reason=record.failure_reason,
        )
