"""Benchmark data model contracts (Phase 7 Step 1).

Defines the benchmark workload contract: tasks, scenarios, and metric
definitions. These are frozen data contracts — no runtime mutable state,
no LLM, no network.

The benchmark is designed to be runner-agnostic: a Naive Runner and a
Reliable Harness Runner should both be able to execute the same
``BenchmarkScenario`` and produce a ``BenchmarkRunRecord`` that can be
compared fairly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Optional


# ---------------------------------------------------------------------------
# Benchmark errors
# ---------------------------------------------------------------------------


class BenchmarkConfigurationError(Exception):
    """Raised when a benchmark scenario or task is misconfigured."""


class UnknownScenarioError(BenchmarkConfigurationError):
    """Raised when a scenario_id is not recognised by the fixture registry."""


# ---------------------------------------------------------------------------
# BenchmarkTask
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkTask:
    """A single benchmark workload definition.

    A task describes WHAT should be accomplished, not HOW. The runner
    decides how to execute it; the oracle decides whether it succeeded.

    Attributes:
        task_id: Stable, unique identifier for the task.
        goal: Human-readable goal description (not parsed by the runner).
        scenario_id: The scenario fixture this task belongs to.
        max_logical_actions: Upper bound on logical agent actions before
            the run is considered exhausted. Must be > 0.
    """

    task_id: str
    goal: str
    scenario_id: str
    max_logical_actions: int

    def __post_init__(self) -> None:
        if not self.task_id or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty str")
        if not self.goal or not self.goal.strip():
            raise ValueError("goal must be a non-empty str")
        if not self.scenario_id or not self.scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty str")
        if (
            not isinstance(self.max_logical_actions, int)
            or isinstance(self.max_logical_actions, bool)
            or self.max_logical_actions < 1
        ):
            raise ValueError("max_logical_actions must be an int >= 1")


# ---------------------------------------------------------------------------
# BenchmarkScenario
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkScenario:
    """A complete benchmark scenario: task + fixture + fault plan.

    A scenario describes WHAT should happen:
    - which repository fixture to start from
    - which fault plan to inject
    - which task to accomplish

    It does NOT describe HOW to execute (that is the runner's job).

    Attributes:
        scenario_id: Stable, unique identifier.
        task: The benchmark task to accomplish.
        fixture_id: Identifier for the repository fixture.
        fault_plan_id: Identifier for the fault plan (or ``"none"``).
        description: Human-readable scenario description.
    """

    scenario_id: str
    task: BenchmarkTask
    fixture_id: str
    fault_plan_id: str = "none"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty str")
        if not isinstance(self.task, BenchmarkTask):
            raise ValueError("task must be a BenchmarkTask")
        if not self.fixture_id or not self.fixture_id.strip():
            raise ValueError("fixture_id must be a non-empty str")
        if not self.fault_plan_id or not self.fault_plan_id.strip():
            raise ValueError("fault_plan_id must be a non-empty str")
        if not isinstance(self.description, str):
            raise ValueError("description must be a str")
        # Cross-check: task.scenario_id must match scenario_id.
        if self.task.scenario_id != self.scenario_id:
            raise ValueError(
                f"task.scenario_id {self.task.scenario_id!r} does not match "
                f"scenario_id {self.scenario_id!r}"
            )


# ---------------------------------------------------------------------------
# Metric definitions (contracts only — no computed values here)
# ---------------------------------------------------------------------------


class MetricName(str, Enum):
    """Names of evaluation metrics that future benchmark runners will
    compute from ``BenchmarkRunRecord`` data.

    These are CONTRACT definitions — the mathematical semantics are
    documented here, but actual computation belongs to a future
    metric-aggregation phase.

    Semantics:

    * ``TASK_COMPLETION`` — 1 if oracle says complete, 0 otherwise.
    * ``RECOVERY_SUCCESS`` — 1 if an interruption occurred AND execution
      resumed AND task eventually completed; 0 otherwise. A run with no
      interruption is NOT a successful recovery.
    * ``DUPLICATE_EXECUTION`` — count of duplicate logical agent actions
      (same tool + same canonical arguments), NOT counting ToolRuntime
      retry attempts.
    * ``LOOP_ESCAPE`` — 1 if a loop was detected and the controller
      escaped it (via replan acknowledgement) and eventually completed;
      0 otherwise.
    * ``UNNECESSARY_RETRY`` — count of retry attempts that violated
      policy or retried a non-retryable result. Does NOT count legitimate
      transient retries.
    * ``TOOL_FAILURE_RECOVERY`` — 1 if at least one tool invocation
      failed (transient/timeout) and the logical action still succeeded
      (via retry); 0 if no failure occurred.
    * ``CONTEXT_USAGE`` — estimated context tokens used (from
      ContextAssembler).
    * ``EXTERNALIZED_OUTPUT_RATIO`` — externalized outputs / total tool
      outputs (0.0 to 1.0).
    * ``STEPS_TO_COMPLETION`` — logical action count to reach completion
      (only meaningful if completed).
    * ``WALL_CLOCK_LATENCY`` — wall-clock seconds from run start to run
      end. NOT used for deterministic unit-test assertions.
    """

    TASK_COMPLETION = "task_completion"
    RECOVERY_SUCCESS = "recovery_success"
    DUPLICATE_EXECUTION = "duplicate_execution"
    LOOP_ESCAPE = "loop_escape"
    UNNECESSARY_RETRY = "unnecessary_retry"
    TOOL_FAILURE_RECOVERY = "tool_failure_recovery"
    CONTEXT_USAGE = "context_usage"
    EXTERNALIZED_OUTPUT_RATIO = "externalized_output_ratio"
    STEPS_TO_COMPLETION = "steps_to_completion"
    WALL_CLOCK_LATENCY = "wall_clock_latency"


@dataclass(frozen=True)
class MetricDefinition:
    """Static definition of a metric: name, description, and whether it
    requires an interruption to be meaningful."""

    name: MetricName
    description: str
    requires_interruption: bool = False
    higher_is_better: bool = True


METRIC_DEFINITIONS: dict[MetricName, MetricDefinition] = {
    MetricName.TASK_COMPLETION: MetricDefinition(
        name=MetricName.TASK_COMPLETION,
        description="1 if oracle says complete, 0 otherwise.",
    ),
    MetricName.RECOVERY_SUCCESS: MetricDefinition(
        name=MetricName.RECOVERY_SUCCESS,
        description=(
            "1 if an interruption occurred AND execution resumed AND "
            "task eventually completed; 0 otherwise."
        ),
        requires_interruption=True,
    ),
    MetricName.DUPLICATE_EXECUTION: MetricDefinition(
        name=MetricName.DUPLICATE_EXECUTION,
        description=(
            "Count of duplicate logical agent actions (same tool + same "
            "canonical arguments). Does NOT count ToolRuntime retry attempts."
        ),
        higher_is_better=False,
    ),
    MetricName.LOOP_ESCAPE: MetricDefinition(
        name=MetricName.LOOP_ESCAPE,
        description=(
            "1 if a loop was detected and the controller escaped it and "
            "eventually completed; 0 otherwise."
        ),
    ),
    MetricName.UNNECESSARY_RETRY: MetricDefinition(
        name=MetricName.UNNECESSARY_RETRY,
        description=(
            "Count of retry attempts that violated policy or retried a "
            "non-retryable result. Does NOT count legitimate transient retries."
        ),
        higher_is_better=False,
    ),
    MetricName.TOOL_FAILURE_RECOVERY: MetricDefinition(
        name=MetricName.TOOL_FAILURE_RECOVERY,
        description=(
            "1 if at least one tool invocation failed and the logical "
            "action still succeeded via retry; 0 if no failure occurred."
        ),
    ),
    MetricName.CONTEXT_USAGE: MetricDefinition(
        name=MetricName.CONTEXT_USAGE,
        description="Estimated context tokens used (from ContextAssembler).",
    ),
    MetricName.EXTERNALIZED_OUTPUT_RATIO: MetricDefinition(
        name=MetricName.EXTERNALIZED_OUTPUT_RATIO,
        description="Externalized outputs / total tool outputs (0.0 to 1.0).",
    ),
    MetricName.STEPS_TO_COMPLETION: MetricDefinition(
        name=MetricName.STEPS_TO_COMPLETION,
        description=(
            "Logical action count to reach completion (only meaningful "
            "if completed)."
        ),
    ),
    MetricName.WALL_CLOCK_LATENCY: MetricDefinition(
        name=MetricName.WALL_CLOCK_LATENCY,
        description=(
            "Wall-clock seconds from run start to run end. NOT used for "
            "deterministic unit-test assertions."
        ),
    ),
}
