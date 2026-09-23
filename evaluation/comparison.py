"""Benchmark comparison model and comparator (Phase 7 Step 2).

Provides a small side-by-side comparison model and a comparator that
runs both ``NaiveBenchmarkRunner`` and ``ReliableHarnessBenchmarkRunner``
on the same ``BenchmarkScenario`` with fresh state for each.

This is a controlled deterministic comparison, NOT a statistical study.
No cross-run aggregation, no plots, no real-world claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from evaluation.models import BenchmarkScenario
from evaluation.records import BenchmarkResult, BenchmarkRunRecord
from evaluation.runners import BenchmarkRunner


@dataclass(frozen=True)
class BenchmarkComparison:
    """Side-by-side comparison of two runners on the same scenario.

    Attributes:
        scenario_id: The scenario that was compared.
        naive: Derived ``BenchmarkResult`` from the naive run.
        reliable: Derived ``BenchmarkResult`` from the reliable run.
        naive_record: Raw ``BenchmarkRunRecord`` from the naive run.
        reliable_record: Raw ``BenchmarkRunRecord`` from the reliable run.
    """

    scenario_id: str
    naive: BenchmarkResult
    reliable: BenchmarkResult
    naive_record: BenchmarkRunRecord
    reliable_record: BenchmarkRunRecord

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty str")
        if self.naive.scenario_id != self.scenario_id:
            raise ValueError(
                f"naive result scenario_id {self.naive.scenario_id!r} "
                f"does not match {self.scenario_id!r}"
            )
        if self.reliable.scenario_id != self.scenario_id:
            raise ValueError(
                f"reliable result scenario_id {self.reliable.scenario_id!r} "
                f"does not match {self.scenario_id!r}"
            )

    @property
    def naive_completed(self) -> bool:
        """True if the naive runner completed the task."""
        return self.naive.task_completed

    @property
    def reliable_completed(self) -> bool:
        """True if the reliable runner completed the task."""
        return self.reliable.task_completed

    @property
    def reliable_advantage(self) -> bool:
        """True if reliable completed but naive did not.

        This is the controlled mechanism validation result: the reliable
        runner recovered from a retryable failure that the naive runner
        could not.
        """
        return self.reliable_completed and not self.naive_completed


class BenchmarkComparator:
    """Runs both runners on the same scenario and returns a comparison.

    Each runner creates its own fresh state (repository, fault injector,
    scripted action source) from the same immutable scenario blueprint.
    The comparator does NOT share state between the two runs.

    The order of execution (naive first or reliable first) does not
    affect the result because each runner creates fresh state.
    """

    def __init__(
        self,
        naive_runner: BenchmarkRunner,
        reliable_runner: BenchmarkRunner,
    ) -> None:
        self._naive = naive_runner
        self._reliable = reliable_runner

    async def compare(self, scenario: BenchmarkScenario) -> BenchmarkComparison:
        """Run both runners on the same scenario and return a comparison.

        Each runner gets a fresh run with fresh state. The scenario
        blueprint is not mutated.
        """
        naive_record = await self._naive.run(scenario)
        reliable_record = await self._reliable.run(scenario)

        return BenchmarkComparison(
            scenario_id=scenario.scenario_id,
            naive=BenchmarkResult.from_record(naive_record),
            reliable=BenchmarkResult.from_record(reliable_record),
            naive_record=naive_record,
            reliable_record=reliable_record,
        )

    async def compare_reversed(
        self, scenario: BenchmarkScenario
    ) -> BenchmarkComparison:
        """Run reliable first, then naive — for order-independence tests.

        The result should be functionally identical to ``compare`` because
        each runner creates fresh state. This method exists to make the
        order-independence test explicit and self-documenting.
        """
        reliable_record = await self._reliable.run(scenario)
        naive_record = await self._naive.run(scenario)

        return BenchmarkComparison(
            scenario_id=scenario.scenario_id,
            naive=BenchmarkResult.from_record(naive_record),
            reliable=BenchmarkResult.from_record(reliable_record),
            naive_record=naive_record,
            reliable_record=reliable_record,
        )
