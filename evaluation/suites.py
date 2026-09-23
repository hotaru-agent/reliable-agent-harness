"""Benchmark suite separation (Phase 7 Steps 6–10).

Provides explicit ``BenchmarkSuite`` definitions so that different
benchmark suites (controlled in-memory, filesystem+pytest, integrated
multi-fault) are summarized independently. Suite membership is
explicit — aggregate counts are derived from the actual scenario
list, not a stale constant.

Suites:
- ``controlled_v1`` — the 8 in-memory controlled scenarios from
  Phase 7 Steps 1–5. Aggregate: Naive 1/8, Reliable 7/8, advantage 6/8.
- ``filesystem_pytest_v1`` — the 4 real-filesystem + real-pytest
  scenarios from Phase 7 Steps 6–8. Aggregate: Naive 1/4, Reliable 4/4,
  advantage 3/4.
- ``integrated_filesystem_v1`` — the 1 integrated multi-fault
  scenario from Phase 7 Step 9. Aggregate: Naive 0/1, Reliable 1/1,
  advantage 1/1.

All offline, deterministic, no LLM, no network.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSuite:
    """An explicit, named set of benchmark scenario IDs.

    Attributes:
        suite_id: Stable identifier for this suite.
        scenario_ids: Tuple of scenario IDs that belong to this suite.
            Membership is explicit and frozen.
    """

    suite_id: str
    scenario_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.suite_id or not self.suite_id.strip():
            raise ValueError("suite_id must be a non-empty str")
        if not isinstance(self.scenario_ids, tuple):
            raise ValueError("scenario_ids must be a tuple")
        if not self.scenario_ids:
            raise ValueError("scenario_ids must not be empty")
        for sid in self.scenario_ids:
            if not isinstance(sid, str) or not sid.strip():
                raise ValueError("each scenario_id must be a non-empty str")

    @property
    def size(self) -> int:
        """Number of scenarios in this suite."""
        return len(self.scenario_ids)

    def contains(self, scenario_id: str) -> bool:
        """Return True if ``scenario_id`` is in this suite."""
        return scenario_id in self.scenario_ids


# ---------------------------------------------------------------------------
# Suite definitions
# ---------------------------------------------------------------------------

# The 8 controlled in-memory scenarios (Phase 7 Steps 1–5).
CONTROLLED_V1 = BenchmarkSuite(
    suite_id="controlled_v1",
    scenario_ids=(
        "clean_success",
        "transient_failure",
        "timeout",
        "permanent_failure",
        "checkpoint_recovery",
        "loop_replan",
        "context_growth",
        "large_output_externalization",
    ),
)

# The 4 real-filesystem + real-pytest scenarios (Phase 7 Steps 6–8).
FILESYSTEM_PYTEST_V1 = BenchmarkSuite(
    suite_id="filesystem_pytest_v1",
    scenario_ids=(
        "filesystem_pytest_clean",
        "filesystem_pytest_transient",
        "filesystem_pytest_timeout",
        "filesystem_pytest_recovery",
    ),
)

# The 1 integrated multi-fault scenario (Phase 7 Step 9).
INTEGRATED_FILESYSTEM_V1 = BenchmarkSuite(
    suite_id="integrated_filesystem_v1",
    scenario_ids=(
        "filesystem_integrated_multifault",
    ),
)


def make_standard_suites() -> dict[str, BenchmarkSuite]:
    """Return all standard benchmark suites keyed by suite_id."""
    return {
        CONTROLLED_V1.suite_id: CONTROLLED_V1,
        FILESYSTEM_PYTEST_V1.suite_id: FILESYSTEM_PYTEST_V1,
        INTEGRATED_FILESYSTEM_V1.suite_id: INTEGRATED_FILESYSTEM_V1,
    }
