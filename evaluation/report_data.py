"""Final benchmark report data (Phase 7 Step 10).

This module is the single source of truth for the final benchmark
report's machine-readable data. It provides:

- The metric dictionary (definitions, what is counted, what is not).
- The suite registry (frozen suite IDs and scenario IDs).
- The scenario matrix builder (derives outcomes from actual
  BenchmarkRunRecord field semantics, NOT from hardcoded numbers).
- The per-suite aggregate builder (sums actual scenario outcomes).

The final report (``FINAL_BENCHMARK_REPORT.md``) and the reporting
tests (``tests/test_phase7_step10_reporting.py``) both reference this
module to avoid stale accounting drift.

No execution logic lives here — this module only collects, defines,
and presents. The actual benchmark execution is performed by the
runners in ``evaluation/runners.py`` and
``evaluation/filesystem_runners.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from evaluation.records import BenchmarkRunRecord
from evaluation.suites import (
    CONTROLLED_V1,
    FILESYSTEM_PYTEST_V1,
    INTEGRATED_FILESYSTEM_V1,
)


# ---------------------------------------------------------------------------
# Suite registry
# ---------------------------------------------------------------------------

ALL_SUITES = (CONTROLLED_V1, FILESYSTEM_PYTEST_V1, INTEGRATED_FILESYSTEM_V1)


# ---------------------------------------------------------------------------
# Metric dictionary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricDefinition:
    """Formal definition of a benchmark metric.

    Attributes:
        name: The field name in ``BenchmarkRunRecord``.
        definition: What exactly is counted.
        not_counted: What is explicitly NOT counted.
        scenarios: Which scenarios populate this metric.
        deterministic: Whether the value is deterministic across runs.
    """

    name: str
    definition: str
    not_counted: str
    scenarios: str
    deterministic: bool


METRIC_DICTIONARY: dict[str, MetricDefinition] = {
    "task_completed": MetricDefinition(
        name="task_completed",
        definition=(
            "True if the objective completion oracle declared the task "
            "complete. For filesystem scenarios, the oracle is a real "
            "pytest subprocess with returncode == 0. For controlled "
            "scenarios, the oracle is an in-process repository state check."
        ),
        not_counted="Harness COMPLETED is NOT oracle completion.",
        scenarios="All scenarios.",
        deterministic=True,
    ),
    "logical_action_count": MetricDefinition(
        name="logical_action_count",
        definition=(
            "Number of actual logical Agent action executions. Each "
            "scripted action is one logical action. Legal recovery "
            "replay of an interrupted uncheckpointed step counts as "
            "another logical action execution."
        ),
        not_counted=(
            "ToolRuntime retry attempts do NOT increment "
            "logical_action_count. Context preparation, output "
            "processing, and oracle execution are NOT logical actions."
        ),
        scenarios="All scenarios.",
        deterministic=True,
    ),
    "tool_invocation_count": MetricDefinition(
        name="tool_invocation_count",
        definition=(
            "Number of logical ToolRuntime.execute() invocations. One "
            "per logical action. Internal retry attempts share the same "
            "logical invocation."
        ),
        not_counted="Internal retry attempts do NOT create new invocations.",
        scenarios="All scenarios.",
        deterministic=True,
    ),
    "tool_attempt_count": MetricDefinition(
        name="tool_attempt_count",
        definition=(
            "Total number of handler attempts: initial attempt plus "
            "internal retry attempts. Always >= tool_invocation_count."
        ),
        not_counted="Oracle execution is NOT an agent tool attempt.",
        scenarios="All scenarios.",
        deterministic=True,
    ),
    "retry_count": MetricDefinition(
        name="retry_count",
        definition=(
            "Number of retries: attempts beyond the first per logical "
            "invocation. Derived from actual ToolRuntime retry history."
        ),
        not_counted=(
            "Resume replay is NOT a retry. Legal recovery replay is NOT "
            "a retry."
        ),
        scenarios="Scenarios with transient/timeout faults.",
        deterministic=True,
    ),
    "checkpoint_count": MetricDefinition(
        name="checkpoint_count",
        definition=(
            "Number of real Harness checkpoints: successful logical "
            "Harness steps plus the COMPLETE checkpoint."
        ),
        not_counted=(
            "Failed internal ToolRuntime attempts and interrupted "
            "uncheckpointed steps do NOT produce checkpoints."
        ),
        scenarios="Reliable scenarios with HarnessRuntime.",
        deterministic=True,
    ),
    "resume_count": MetricDefinition(
        name="resume_count",
        definition=(
            "Number of times HarnessRuntime.resume() was called and a "
            "new resume Run was created."
        ),
        not_counted="Retry and fresh start are NOT resume.",
        scenarios="Recovery and integrated scenarios.",
        deterministic=True,
    ),
    "loop_detection_count": MetricDefinition(
        name="loop_detection_count",
        definition=(
            "Number of times loop detection fired (duplicate, repeating "
            "sequence, or no-progress)."
        ),
        not_counted="Blocked gate probes are NOT loop detections.",
        scenarios="Loop/replan controlled scenario.",
        deterministic=True,
    ),
    "replan_count": MetricDefinition(
        name="replan_count",
        definition="Number of REPLAN_REQUIRED signals emitted.",
        not_counted="Loop detections without replan are NOT replan counts.",
        scenarios="Loop/replan controlled scenario.",
        deterministic=True,
    ),
    "blocked_call_count": MetricDefinition(
        name="blocked_call_count",
        definition=(
            "Number of tool calls blocked by the replan gate before "
            "reaching ToolRuntime. Blocked calls consume no action index."
        ),
        not_counted="Executed actions are NOT blocked calls.",
        scenarios="Loop/replan controlled scenario.",
        deterministic=True,
    ),
    "processed_output_count": MetricDefinition(
        name="processed_output_count",
        definition=(
            "Number of tool outputs successfully processed (inline or "
            "externalized). Used as the denominator for the "
            "externalized-output ratio."
        ),
        not_counted="Tool attempts that failed are NOT processed outputs.",
        scenarios="Context/externalization controlled scenario.",
        deterministic=True,
    ),
    "externalized_output_count": MetricDefinition(
        name="externalized_output_count",
        definition=(
            "Number of tool outputs externalized to ArtifactStore instead "
            "of inlined into context."
        ),
        not_counted="Inline outputs are NOT externalized.",
        scenarios="Context/externalization controlled scenario.",
        deterministic=True,
    ),
    "externalized_output_ratio": MetricDefinition(
        name="externalized_output_ratio",
        definition=(
            "externalized_output_count / processed_output_count. "
            "Fraction of processed outputs that were externalized."
        ),
        not_counted="Denominator is processed outputs, NOT tool attempts.",
        scenarios="Context/externalization controlled scenario.",
        deterministic=True,
    ),
    "peak_context_tokens": MetricDefinition(
        name="peak_context_tokens",
        definition=(
            "Peak estimated context tokens at an actual decision "
            "boundary. For Reliable: selected/presented context. For "
            "Naive: attempted append-all context (may exceed capacity)."
        ),
        not_counted="Final accumulated history is NOT peak context.",
        scenarios="Context/externalization controlled scenario.",
        deterministic=True,
    ),
    "wall_clock_seconds": MetricDefinition(
        name="wall_clock_seconds",
        definition="Real wall-clock duration of the run in seconds.",
        not_counted=(
            "NOT part of deterministic functional equality. Real pytest "
            "timing varies across runs."
        ),
        scenarios="All scenarios.",
        deterministic=False,
    ),
}


# ---------------------------------------------------------------------------
# Scenario matrix entry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioMatrixEntry:
    """One row in the final scenario matrix.

    Attributes:
        suite_id: The suite this scenario belongs to.
        scenario_id: The scenario identifier.
        naive_outcome: "PASS" or "FAIL".
        reliable_outcome: "PASS" or "FAIL".
        mechanism: Human-readable mechanism description.
        real_filesystem: Whether this scenario uses a real temp filesystem.
        real_subprocess: Whether this scenario spawns a real pytest subprocess.
        resume: Whether this scenario involves checkpoint/resume.
    """

    suite_id: str
    scenario_id: str
    naive_outcome: str
    reliable_outcome: str
    mechanism: str
    real_filesystem: bool
    real_subprocess: bool
    resume: bool


# ---------------------------------------------------------------------------
# Scenario metadata (static, derived from scenario definitions)
# ---------------------------------------------------------------------------

_SCENARIO_METADATA: dict[str, dict[str, object]] = {
    # controlled_v1
    "clean_success": {
        "mechanism": "No faults; baseline execution.",
        "real_filesystem": False,
        "real_subprocess": False,
        "resume": False,
    },
    "transient_failure": {
        "mechanism": "Transient tool failure; RetryPolicy retry.",
        "real_filesystem": False,
        "real_subprocess": False,
        "resume": False,
    },
    "timeout": {
        "mechanism": "Tool timeout; RetryPolicy retry.",
        "real_filesystem": False,
        "real_subprocess": False,
        "resume": False,
    },
    "permanent_failure": {
        "mechanism": "Permanent tool failure; no retry.",
        "real_filesystem": False,
        "real_subprocess": False,
        "resume": False,
    },
    "checkpoint_recovery": {
        "mechanism": "Process interruption; checkpoint/resume.",
        "real_filesystem": False,
        "real_subprocess": False,
        "resume": True,
    },
    "loop_replan": {
        "mechanism": "Loop detection; replan gate.",
        "real_filesystem": False,
        "real_subprocess": False,
        "resume": False,
    },
    "context_growth": {
        "mechanism": "Context budget; bounded selection.",
        "real_filesystem": False,
        "real_subprocess": False,
        "resume": False,
    },
    "large_output_externalization": {
        "mechanism": "Large output; artifact externalization.",
        "real_filesystem": False,
        "real_subprocess": False,
        "resume": False,
    },
    # filesystem_pytest_v1
    "filesystem_pytest_clean": {
        "mechanism": "Real filesystem + real pytest.",
        "real_filesystem": True,
        "real_subprocess": True,
        "resume": False,
    },
    "filesystem_pytest_transient": {
        "mechanism": "Transient retry on real pytest subprocess.",
        "real_filesystem": True,
        "real_subprocess": True,
        "resume": False,
    },
    "filesystem_pytest_timeout": {
        "mechanism": "Real subprocess timeout + cleanup + retry.",
        "real_filesystem": True,
        "real_subprocess": True,
        "resume": False,
    },
    "filesystem_pytest_recovery": {
        "mechanism": "Filesystem checkpoint/resume.",
        "real_filesystem": True,
        "real_subprocess": True,
        "resume": True,
    },
    # integrated_filesystem_v1
    "filesystem_integrated_multifault": {
        "mechanism": "Transient + timeout + checkpoint/resume in one trial.",
        "real_filesystem": True,
        "real_subprocess": True,
        "resume": True,
    },
}


def _outcome(completed: bool) -> str:
    return "PASS" if completed else "FAIL"


def build_scenario_matrix(
    naive_records: dict[str, BenchmarkRunRecord],
    reliable_records: dict[str, BenchmarkRunRecord],
) -> list[ScenarioMatrixEntry]:
    """Build the scenario matrix from actual benchmark run records.

    Args:
        naive_records: Map of scenario_id → BenchmarkRunRecord for Naive.
        reliable_records: Map of scenario_id → BenchmarkRunRecord for
            Reliable.

    Returns:
        List of ScenarioMatrixEntry, one per scenario, in suite order.
    """
    entries: list[ScenarioMatrixEntry] = []
    for suite in ALL_SUITES:
        for sid in suite.scenario_ids:
            meta = _SCENARIO_METADATA.get(sid, {})
            n_rec = naive_records.get(sid)
            r_rec = reliable_records.get(sid)
            entries.append(ScenarioMatrixEntry(
                suite_id=suite.suite_id,
                scenario_id=sid,
                naive_outcome=_outcome(n_rec.completed if n_rec else False),
                reliable_outcome=_outcome(r_rec.completed if r_rec else False),
                mechanism=str(meta.get("mechanism", "")),
                real_filesystem=bool(meta.get("real_filesystem", False)),
                real_subprocess=bool(meta.get("real_subprocess", False)),
                resume=bool(meta.get("resume", False)),
            ))
    return entries


# ---------------------------------------------------------------------------
# Suite aggregate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuiteAggregate:
    """Per-suite completion aggregate.

    Attributes:
        suite_id: The suite identifier.
        total_scenarios: Number of scenarios in the suite.
        naive_completed: Number of scenarios Naive completed.
        reliable_completed: Number of scenarios Reliable completed.
        advantage: Number of scenarios where Reliable completed but
            Naive did not.
    """

    suite_id: str
    total_scenarios: int
    naive_completed: int
    reliable_completed: int
    advantage: int


def build_suite_aggregates(
    naive_records: dict[str, BenchmarkRunRecord],
    reliable_records: dict[str, BenchmarkRunRecord],
) -> list[SuiteAggregate]:
    """Build per-suite aggregates from actual benchmark run records.

    The aggregates are derived from actual completion results, NOT
    hardcoded constants.
    """
    aggregates: list[SuiteAggregate] = []
    for suite in ALL_SUITES:
        n_completed = 0
        r_completed = 0
        advantage = 0
        for sid in suite.scenario_ids:
            n = naive_records.get(sid)
            r = reliable_records.get(sid)
            n_pass = n.completed if n else False
            r_pass = r.completed if r else False
            if n_pass:
                n_completed += 1
            if r_pass:
                r_completed += 1
            if r_pass and not n_pass:
                advantage += 1
        aggregates.append(SuiteAggregate(
            suite_id=suite.suite_id,
            total_scenarios=suite.size,
            naive_completed=n_completed,
            reliable_completed=r_completed,
            advantage=advantage,
        ))
    return aggregates


# ---------------------------------------------------------------------------
# Total scenario count (for inventory, NOT for a combined success rate)
# ---------------------------------------------------------------------------

def total_scenario_count() -> int:
    """Return the total number of deterministic benchmark scenarios
    across all suites.

    This is for inventory only. Per-suite aggregates must be reported
    separately — the suites are NOT the same sampling population.
    """
    return sum(s.size for s in ALL_SUITES)


# ---------------------------------------------------------------------------
# Suite disjointness check
# ---------------------------------------------------------------------------

def suites_are_disjoint() -> bool:
    """Return True if all suites share no scenario IDs."""
    seen: set[str] = set()
    for suite in ALL_SUITES:
        for sid in suite.scenario_ids:
            if sid in seen:
                return False
            seen.add(sid)
    return True
