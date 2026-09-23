"""Phase 7 Step 2 — Benchmark runner contract, naive, reliable, fairness,
and accounting regression tests.

Covers Sections 4-13, 19-23, 35, 46-53, 54-59 of the Phase 7 Step 2
specification.
"""

from __future__ import annotations

import asyncio

import pytest

from evaluation.comparison import BenchmarkComparison, BenchmarkComparator
from evaluation.faults import FaultInjector, FaultPlan, FaultSpec, FaultType
from evaluation.models import BenchmarkScenario, BenchmarkTask
from evaluation.records import BenchmarkResult, BenchmarkRunRecord
from evaluation.repository import RepositoryOracle
from evaluation.runners import (
    BenchmarkRunner,
    BenchmarkStepExecutor,
    BenchmarkStepFailure,
    MaxActionsExceededError,
    NaiveBenchmarkRunner,
    ReliableHarnessBenchmarkRunner,
)
from evaluation.scenarios.controlled_repo import (
    CORRECTED_CALCULATOR_CONTENT,
    ScriptedAction,
    ScriptedActionSource,
    make_calculator_fixture,
    make_fix_calculator_actions,
    make_repository_tool_handlers,
    make_repository_tool_specs,
    make_standard_fault_plans,
    make_standard_fixtures,
    make_standard_scenarios,
)
from tools import (
    RetryPolicy,
    ToolCall,
    ToolExecutionContext,
    ToolRegistry,
    ToolRuntime,
)
from tools.models import ToolErrorType


def _run(coro):
    return asyncio.run(coro)


class _FakeSleeper:
    """Deterministic sleep that records durations without sleeping."""

    def __init__(self):
        self.sleeps: list[float] = []

    async def __call__(self, seconds: float):
        self.sleeps.append(seconds)


# Use a very small timeout so timeout tests are fast. The timeout fault
# sleeps 10s via real asyncio.sleep, which asyncio.wait_for cancels after
# this timeout.
_TIMEOUT_SECONDS = 0.01


def _make_naive():
    return NaiveBenchmarkRunner(
        run_tests_timeout_seconds=_TIMEOUT_SECONDS,
        sleep=_FakeSleeper(),
    )


def _make_reliable():
    return ReliableHarnessBenchmarkRunner(
        run_tests_timeout_seconds=_TIMEOUT_SECONDS,
        sleep=_FakeSleeper(),
    )


def _make_comparator():
    return BenchmarkComparator(_make_naive(), _make_reliable())


def _get_scenario(scenario_id: str) -> BenchmarkScenario:
    return make_standard_scenarios()[scenario_id]


# ===========================================================================
# Section 54 — Runner contract tests
# ===========================================================================


class TestRunnerContract:
    """The BenchmarkRunner protocol and its implementations."""

    def test_naive_implements_protocol(self):
        """NaiveBenchmarkRunner satisfies the BenchmarkRunner protocol."""
        runner = _make_naive()
        assert hasattr(runner, "run")

    def test_reliable_implements_protocol(self):
        """ReliableHarnessBenchmarkRunner satisfies the BenchmarkRunner protocol."""
        runner = _make_reliable()
        assert hasattr(runner, "run")

    def test_runner_creates_fresh_state_per_run(self):
        """Two consecutive runs of the same scenario produce fresh state.

        The second run must not see the first run's repository changes.
        """
        runner = _make_naive()
        scenario = _get_scenario("clean_success")

        r1 = _run(runner.run(scenario))
        r2 = _run(runner.run(scenario))

        # Both should complete (fresh repo each time).
        assert r1.completed is True
        assert r2.completed is True
        # Both have the same logical action count (fresh script each time).
        assert r1.logical_action_count == r2.logical_action_count

    def test_scenario_input_not_mutated(self):
        """The runner must not mutate the scenario definition."""
        runner = _make_naive()
        scenario = _get_scenario("clean_success")
        original_scenario_id = scenario.scenario_id
        original_task_id = scenario.task.task_id
        original_fixture_id = scenario.fixture_id
        original_fault_plan_id = scenario.fault_plan_id
        original_max_actions = scenario.task.max_logical_actions

        _run(runner.run(scenario))

        assert scenario.scenario_id == original_scenario_id
        assert scenario.task.task_id == original_task_id
        assert scenario.fixture_id == original_fixture_id
        assert scenario.fault_plan_id == original_fault_plan_id
        assert scenario.task.max_logical_actions == original_max_actions

    def test_max_logical_actions_respected(self):
        """The runner respects max_logical_actions."""
        # Create a scenario with max_logical_actions=2 (less than the 4
        # scripted actions).
        task = BenchmarkTask(
            task_id="fix_calculator_add",
            goal="Fix the buggy add() function.",
            scenario_id="max_actions_test",
            max_logical_actions=2,
        )
        scenario = BenchmarkScenario(
            scenario_id="max_actions_test",
            task=task,
            fixture_id="calculator_buggy_add",
            fault_plan_id="none",
        )
        runner = _make_naive()
        record = _run(runner.run(scenario))

        # The run should not complete (only 2 of 4 actions executed).
        assert record.completed is False
        assert record.logical_action_count == 2
        assert record.failure_reason == "MAX_ACTIONS"

    def test_script_exhaustion_with_incomplete_oracle_fails(self):
        """If the script is exhausted but the oracle says incomplete,
        completed must be False."""
        # Create a scenario where the script doesn't fix the bug.
        # Use a script that only runs tests (doesn't write the fix).
        from evaluation.scenarios.controlled_repo import ScriptedAction

        task = BenchmarkTask(
            task_id="incomplete_test",
            goal="Fix the buggy add() function.",
            scenario_id="incomplete_test",
            max_logical_actions=10,
        )
        scenario = BenchmarkScenario(
            scenario_id="incomplete_test",
            task=task,
            fixture_id="calculator_buggy_add",
            fault_plan_id="none",
        )
        # Use a custom action factory that only runs tests (no fix).
        runner = NaiveBenchmarkRunner(
            run_tests_timeout_seconds=_TIMEOUT_SECONDS,
            sleep=_FakeSleeper(),
            action_factory=lambda: [
                ScriptedAction(
                    tool_name="run_tests",
                    arguments={},
                    description="Run tests (no fix applied).",
                ),
            ],
        )
        record = _run(runner.run(scenario))

        # Script exhausted, but oracle says incomplete (bug not fixed).
        assert record.completed is False
        assert record.failure_reason == "ORACLE_INCOMPLETE"
        assert record.logical_action_count == 1

    def test_oracle_determines_final_completion(self):
        """Completion is always determined by RepositoryOracle."""
        runner = _make_naive()
        scenario = _get_scenario("clean_success")
        record = _run(runner.run(scenario))

        # Oracle says complete → completed=True.
        assert record.completed is True

    def test_records_are_produced_deterministically(self):
        """Running the same scenario twice produces the same functional
        record (except wall_clock_seconds which is always 0.0)."""
        runner = _make_naive()
        scenario = _get_scenario("clean_success")

        r1 = _run(runner.run(scenario))
        r2 = _run(runner.run(scenario))

        assert r1.completed == r2.completed
        assert r1.logical_action_count == r2.logical_action_count
        assert r1.tool_invocation_count == r2.tool_invocation_count
        assert r1.tool_attempt_count == r2.tool_attempt_count
        assert r1.retry_count == r2.retry_count
        assert r1.failure_reason == r2.failure_reason


# ===========================================================================
# Section 55 — Naive runner tests
# ===========================================================================


class TestNaiveRunner:
    """NaiveBenchmarkRunner specific behavior."""

    def test_clean_scenario_completes(self):
        runner = _make_naive()
        record = _run(runner.run(_get_scenario("clean_success")))
        assert record.completed is True
        assert record.logical_action_count == 4
        assert record.tool_invocation_count == 4
        assert record.tool_attempt_count == 4
        assert record.retry_count == 0
        assert record.checkpoint_count == 0
        assert record.failure_reason is None

    def test_transient_scenario_stops_on_first_failure(self):
        runner = _make_naive()
        record = _run(runner.run(_get_scenario("transient_failure")))
        assert record.completed is False
        assert record.logical_action_count == 1  # only the first run_tests
        assert record.retry_count == 0
        assert record.checkpoint_count == 0
        assert record.failure_reason == "TRANSIENT"

    def test_timeout_scenario_stops_on_first_timeout(self):
        runner = _make_naive()
        record = _run(runner.run(_get_scenario("timeout")))
        assert record.completed is False
        assert record.logical_action_count == 1
        assert record.retry_count == 0
        assert record.checkpoint_count == 0
        assert record.failure_reason == "TIMEOUT"

    def test_permanent_scenario_stops(self):
        runner = _make_naive()
        record = _run(runner.run(_get_scenario("permanent_failure")))
        assert record.completed is False
        assert record.logical_action_count == 1
        assert record.retry_count == 0
        assert record.checkpoint_count == 0
        assert record.failure_reason == "PERMANENT"

    def test_retry_count_always_zero(self):
        """Naive runner never retries — retry_count is always 0."""
        for scenario_id in ("clean_success", "transient_failure",
                            "permanent_failure", "timeout"):
            runner = _make_naive()
            record = _run(runner.run(_get_scenario(scenario_id)))
            assert record.retry_count == 0, f"{scenario_id}: retry_count != 0"

    def test_checkpoint_count_always_zero(self):
        """Naive runner has no HarnessRuntime — checkpoint_count is always 0."""
        for scenario_id in ("clean_success", "transient_failure",
                            "permanent_failure", "timeout"):
            runner = _make_naive()
            record = _run(runner.run(_get_scenario(scenario_id)))
            assert record.checkpoint_count == 0, (
                f"{scenario_id}: checkpoint_count != 0"
            )


# ===========================================================================
# Section 56 — Reliable runner tests
# ===========================================================================


class TestReliableRunner:
    """ReliableHarnessBenchmarkRunner specific behavior."""

    def test_clean_scenario_completes_through_harness(self):
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("clean_success")))
        assert record.completed is True
        assert record.logical_action_count == 4
        assert record.tool_invocation_count == 4
        assert record.tool_attempt_count == 4
        assert record.retry_count == 0
        # 4 action steps + 1 COMPLETE step = 5 checkpoints.
        assert record.checkpoint_count == 5
        assert record.failure_reason is None

    def test_transient_scenario_recovers_through_retry(self):
        """The key comparison: reliable runner recovers from transient
        failure via ToolRuntime retry."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("transient_failure")))
        assert record.completed is True
        assert record.logical_action_count == 4
        assert record.tool_invocation_count == 4
        # First run_tests: 2 attempts (1 transient + 1 success).
        # Rest: 1 attempt each. Total: 2 + 1 + 1 + 1 = 5.
        assert record.tool_attempt_count == 5
        assert record.retry_count == 1
        assert record.failure_reason is None

    def test_timeout_scenario_recovers_through_retry(self):
        """Reliable runner recovers from timeout via ToolRuntime retry."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("timeout")))
        assert record.completed is True
        assert record.logical_action_count == 4
        assert record.tool_attempt_count == 5
        assert record.retry_count == 1
        assert record.failure_reason is None

    def test_permanent_scenario_fails_without_retry(self):
        """The critical negative proof: reliable runner does NOT retry
        permanent failures."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("permanent_failure")))
        assert record.completed is False
        assert record.logical_action_count == 1
        assert record.tool_attempt_count == 1  # no retry
        assert record.retry_count == 0  # no retry
        assert record.failure_reason == "PERMANENT"

    def test_retry_attempt_does_not_become_new_logical_action(self):
        """A retry attempt within one logical call does NOT increment
        logical_action_count."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("transient_failure")))
        # 4 logical actions, not 5 (the retry is not a new action).
        assert record.logical_action_count == 4
        assert record.tool_invocation_count == 4
        # But 5 attempts (the retry added an attempt).
        assert record.tool_attempt_count == 5

    def test_successful_steps_create_checkpoint_progression(self):
        """Each successful logical step creates a real checkpoint."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("clean_success")))
        # 4 action steps + 1 COMPLETE step = 5 checkpoints.
        assert record.checkpoint_count == 5
        assert record.checkpoint_count > 0

    def test_oracle_determines_completed_field(self):
        """The completed field comes from RepositoryOracle, not from
        HarnessRuntime COMPLETED status."""
        runner = _make_reliable()
        # In the permanent failure scenario, the harness run fails, but
        # even if it had completed, the oracle would say incomplete (bug
        # not fixed).
        record = _run(runner.run(_get_scenario("permanent_failure")))
        assert record.completed is False


# ===========================================================================
# Section 57 — Fairness tests
# ===========================================================================


class TestFairness:
    """Both runners see the same fault placement and use fresh state."""

    def test_same_scenario_definition_used_for_both(self):
        """Both runners receive the same immutable scenario."""
        comparator = _make_comparator()
        scenario = _get_scenario("transient_failure")
        comparison = _run(comparator.compare(scenario))
        assert comparison.naive_record.scenario_id == scenario.scenario_id
        assert comparison.reliable_record.scenario_id == scenario.scenario_id

    def test_fresh_repository_for_both(self):
        """Each runner gets a fresh repository — no cross-contamination."""
        comparator = _make_comparator()
        scenario = _get_scenario("clean_success")
        comparison = _run(comparator.compare(scenario))
        # Both should complete (fresh repo each time).
        assert comparison.naive_completed is True
        assert comparison.reliable_completed is True

    def test_same_transient_fault_logical_placement(self):
        """Both runners see the transient fault on the same logical
        invocation (run_tests #1)."""
        comparator = _make_comparator()
        scenario = _get_scenario("transient_failure")
        comparison = _run(comparator.compare(scenario))

        # Naive: fails on run_tests #1 (transient, no retry).
        assert comparison.naive_record.completed is False
        assert comparison.naive_record.failure_reason == "TRANSIENT"
        assert comparison.naive_record.logical_action_count == 1

        # Reliable: run_tests #1 retries and succeeds, continues.
        assert comparison.reliable_record.completed is True
        assert comparison.reliable_record.logical_action_count == 4
        assert comparison.reliable_record.retry_count == 1

    def test_same_timeout_fault_logical_placement(self):
        """Both runners see the timeout fault on the same logical
        invocation (run_tests #1)."""
        comparator = _make_comparator()
        scenario = _get_scenario("timeout")
        comparison = _run(comparator.compare(scenario))

        assert comparison.naive_record.completed is False
        assert comparison.naive_record.failure_reason == "TIMEOUT"

        assert comparison.reliable_record.completed is True
        assert comparison.reliable_record.retry_count == 1

    def test_comparison_order_does_not_change_result(self):
        """Running naive→reliable vs reliable→naive produces the same
        functional result."""
        comparator = _make_comparator()
        scenario = _get_scenario("transient_failure")

        c1 = _run(comparator.compare(scenario))  # naive first
        c2 = _run(comparator.compare_reversed(scenario))  # reliable first

        assert c1.naive_completed == c2.naive_completed
        assert c1.reliable_completed == c2.reliable_completed
        assert c1.naive_record.failure_reason == c2.naive_record.failure_reason
        assert c1.reliable_record.failure_reason == c2.reliable_record.failure_reason
        assert c1.naive_record.logical_action_count == c2.naive_record.logical_action_count
        assert c1.reliable_record.logical_action_count == c2.reliable_record.logical_action_count


# ===========================================================================
# Section 58 — Accounting tests
# ===========================================================================


class TestAccounting:
    """Logical action, invocation, attempt, and retry count accounting."""

    def test_transient_reliable_attempt_count_relationship(self):
        """For the transient reliable run:
        tool_attempt_count > tool_invocation_count
        retry_count = tool_attempt_count - tool_invocation_count
        """
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("transient_failure")))
        assert record.tool_attempt_count > record.tool_invocation_count
        assert record.retry_count == (
            record.tool_attempt_count - record.tool_invocation_count
        )

    def test_clean_naive_attempt_equals_invocation(self):
        """For the clean naive run: attempt_count == invocation_count."""
        runner = _make_naive()
        record = _run(runner.run(_get_scenario("clean_success")))
        assert record.tool_attempt_count == record.tool_invocation_count
        assert record.retry_count == 0

    def test_one_logical_action_one_invocation(self):
        """One scripted action = one logical action = one tool invocation."""
        runner = _make_naive()
        record = _run(runner.run(_get_scenario("clean_success")))
        assert record.logical_action_count == record.tool_invocation_count

    def test_reliable_transient_logical_action_not_inflated(self):
        """The transient retry does NOT inflate logical_action_count."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("transient_failure")))
        # 4 logical actions, not 5.
        assert record.logical_action_count == 4
        # 4 tool invocations, not 5.
        assert record.tool_invocation_count == 4
        # 5 attempts (the retry added one).
        assert record.tool_attempt_count == 5


# ===========================================================================
# Section 59 — Permanent failure safety tests
# ===========================================================================


class TestPermanentFailureSafety:
    """The reliable runner must NOT retry permanent failures."""

    def test_permanent_no_additional_attempt(self):
        """PERMANENT failure → no additional attempt."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("permanent_failure")))
        assert record.tool_attempt_count == 1
        assert record.retry_count == 0

    def test_permanent_both_runners_fail(self):
        """Both naive and reliable fail on permanent failure."""
        comparator = _make_comparator()
        comparison = _run(comparator.compare(_get_scenario("permanent_failure")))
        assert comparison.naive_completed is False
        assert comparison.reliable_completed is False
        assert comparison.naive_record.failure_reason == "PERMANENT"
        assert comparison.reliable_record.failure_reason == "PERMANENT"

    def test_permanent_reliable_no_retry_advantage(self):
        """Reliable runner does NOT gain an advantage on permanent failure
        — it fails just like naive."""
        comparator = _make_comparator()
        comparison = _run(comparator.compare(_get_scenario("permanent_failure")))
        # No reliable advantage on permanent failure.
        assert not comparison.reliable_advantage
        # Both have retry_count == 0.
        assert comparison.naive_record.retry_count == 0
        assert comparison.reliable_record.retry_count == 0


# ===========================================================================
# Section 35 — Retry metrics tests
# ===========================================================================


class TestRetryMetrics:
    """Verify retry_count for each scenario and runner."""

    def test_transient_naive_retry_count_zero(self):
        runner = _make_naive()
        record = _run(runner.run(_get_scenario("transient_failure")))
        assert record.retry_count == 0

    def test_transient_reliable_retry_count_at_least_one(self):
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("transient_failure")))
        assert record.retry_count >= 1

    def test_permanent_naive_retry_count_zero(self):
        runner = _make_naive()
        record = _run(runner.run(_get_scenario("permanent_failure")))
        assert record.retry_count == 0

    def test_permanent_reliable_retry_count_zero(self):
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("permanent_failure")))
        assert record.retry_count == 0

    def test_timeout_naive_retry_count_zero(self):
        runner = _make_naive()
        record = _run(runner.run(_get_scenario("timeout")))
        assert record.retry_count == 0

    def test_timeout_reliable_retry_count_at_least_one(self):
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("timeout")))
        assert record.retry_count >= 1


# ===========================================================================
# Section 36 — Checkpoint metrics tests
# ===========================================================================


class TestCheckpointMetrics:
    """Verify checkpoint_count for each runner."""

    def test_naive_checkpoint_count_zero(self):
        """Naive runner has no HarnessRuntime — checkpoint_count == 0."""
        for scenario_id in ("clean_success", "transient_failure",
                            "permanent_failure", "timeout"):
            runner = _make_naive()
            record = _run(runner.run(_get_scenario(scenario_id)))
            assert record.checkpoint_count == 0

    def test_reliable_clean_has_checkpoints(self):
        """Reliable runner produces real checkpoints on clean success."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("clean_success")))
        assert record.checkpoint_count > 0

    def test_reliable_transient_has_checkpoints(self):
        """Reliable runner produces checkpoints even with a transient fault."""
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("transient_failure")))
        assert record.checkpoint_count > 0


# ===========================================================================
# Section 46 — Clean scenario equivalence test
# ===========================================================================


class TestCleanScenarioEquivalence:
    """Clean scenario: both runners complete with the same functional
    result."""

    def test_clean_both_complete(self):
        comparator = _make_comparator()
        comparison = _run(comparator.compare(_get_scenario("clean_success")))
        assert comparison.naive_completed is True
        assert comparison.reliable_completed is True

    def test_clean_same_logical_action_count(self):
        comparator = _make_comparator()
        comparison = _run(comparator.compare(_get_scenario("clean_success")))
        assert (
            comparison.naive_record.logical_action_count
            == comparison.reliable_record.logical_action_count
        )

    def test_clean_same_tool_invocation_count(self):
        comparator = _make_comparator()
        comparison = _run(comparator.compare(_get_scenario("clean_success")))
        assert (
            comparison.naive_record.tool_invocation_count
            == comparison.reliable_record.tool_invocation_count
        )

    def test_clean_both_retry_count_zero(self):
        comparator = _make_comparator()
        comparison = _run(comparator.compare(_get_scenario("clean_success")))
        assert comparison.naive_record.retry_count == 0
        assert comparison.reliable_record.retry_count == 0


# ===========================================================================
# Section 48 — Determinism test
# ===========================================================================


class TestDeterminism:
    """Same runner + same scenario run twice → same functional record."""

    def test_naive_deterministic(self):
        runner = _make_naive()
        scenario = _get_scenario("transient_failure")
        r1 = _run(runner.run(scenario))
        r2 = _run(runner.run(scenario))
        assert r1.completed == r2.completed
        assert r1.logical_action_count == r2.logical_action_count
        assert r1.tool_invocation_count == r2.tool_invocation_count
        assert r1.tool_attempt_count == r2.tool_attempt_count
        assert r1.retry_count == r2.retry_count
        assert r1.failure_reason == r2.failure_reason

    def test_reliable_deterministic(self):
        runner = _make_reliable()
        scenario = _get_scenario("transient_failure")
        r1 = _run(runner.run(scenario))
        r2 = _run(runner.run(scenario))
        assert r1.completed == r2.completed
        assert r1.logical_action_count == r2.logical_action_count
        assert r1.tool_invocation_count == r2.tool_invocation_count
        assert r1.tool_attempt_count == r2.tool_attempt_count
        assert r1.retry_count == r2.retry_count
        assert r1.failure_reason == r2.failure_reason
        assert r1.checkpoint_count == r2.checkpoint_count


# ===========================================================================
# Section 47 — Fault placement equivalence test
# ===========================================================================


class TestFaultPlacementEquivalence:
    """Both runners observe the same fault placement (same tool, same
    logical invocation, same FaultSpec)."""

    def test_transient_fault_on_run_tests_logical_1_both(self):
        """The transient fault fires on run_tests logical invocation #1
        for both runners."""
        # Naive: the fault fires on the first (and only) attempt.
        # The naive runner sees a TRANSIENT failure on run_tests #1.
        runner_n = _make_naive()
        record_n = _run(runner_n.run(_get_scenario("transient_failure")))
        assert record_n.logical_action_count == 1  # only run_tests #1
        assert record_n.failure_reason == "TRANSIENT"

        # Reliable: the fault fires on run_tests #1 attempt 1, then
        # attempt 2 succeeds. The reliable runner continues.
        runner_r = _make_reliable()
        record_r = _run(runner_r.run(_get_scenario("transient_failure")))
        assert record_r.logical_action_count == 4  # all 4 actions
        assert record_r.retry_count == 1  # one retry on run_tests #1
        assert record_r.completed is True

    def test_timeout_fault_on_run_tests_logical_1_both(self):
        """The timeout fault fires on run_tests logical invocation #1
        for both runners."""
        runner_n = _make_naive()
        record_n = _run(runner_n.run(_get_scenario("timeout")))
        assert record_n.logical_action_count == 1
        assert record_n.failure_reason == "TIMEOUT"

        runner_r = _make_reliable()
        record_r = _run(runner_r.run(_get_scenario("timeout")))
        assert record_r.logical_action_count == 4
        assert record_r.retry_count == 1
        assert record_r.completed is True


# ===========================================================================
# Section 8 — Fail-open protection
# ===========================================================================


class TestFailOpenProtection:
    """Non-empty fault plans must go through FaultAwareToolInvoker. No
    silent fault bypass."""

    def test_naive_uses_invoker_for_faults(self):
        """The naive runner uses FaultAwareToolInvoker when faults exist.

        We verify this by checking that the transient fault actually
        fires (the naive runner fails with TRANSIENT, not silently
        succeeding).
        """
        runner = _make_naive()
        record = _run(runner.run(_get_scenario("transient_failure")))
        # If the invoker were bypassed, the fault would not fire and the
        # run would succeed. The fact that it fails with TRANSIENT proves
        # the invoker was used.
        assert record.completed is False
        assert record.failure_reason == "TRANSIENT"

    def test_reliable_uses_invoker_for_faults(self):
        """The reliable runner uses FaultAwareToolInvoker when faults
        exist.

        We verify this by checking that the transient fault fires on
        attempt 1 (causing a retry) rather than being silently skipped.
        """
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("transient_failure")))
        # If the invoker were bypassed, retry_count would be 0. The fact
        # that retry_count == 1 proves the fault fired (and was retried).
        assert record.retry_count == 1

    def test_no_fault_plan_succeeds_without_invoker_issues(self):
        """The 'none' fault plan works correctly — no faults, no
        invoker-related issues."""
        runner = _make_naive()
        record = _run(runner.run(_get_scenario("clean_success")))
        assert record.completed is True
        assert record.failure_reason is None


# ===========================================================================
# Section 52-53 — Oracle always wins / Harness COMPLETE is not oracle
# ===========================================================================


class TestOracleAlwaysWins:
    """The oracle always determines completion, not the runner or harness."""

    def test_oracle_overrides_harness_success(self):
        """If the harness completes but the oracle says incomplete,
        completed is False."""
        # Create a scenario where the script completes all actions but
        # doesn't fix the bug.
        task = BenchmarkTask(
            task_id="incomplete_test",
            goal="Fix the buggy add() function.",
            scenario_id="incomplete_test",
            max_logical_actions=10,
        )
        scenario = BenchmarkScenario(
            scenario_id="incomplete_test",
            task=task,
            fixture_id="calculator_buggy_add",
            fault_plan_id="none",
        )
        runner = ReliableHarnessBenchmarkRunner(
            run_tests_timeout_seconds=_TIMEOUT_SECONDS,
            sleep=_FakeSleeper(),
            action_factory=lambda: [
                ScriptedAction(
                    tool_name="run_tests",
                    arguments={},
                    description="Run tests (no fix applied).",
                ),
            ],
        )
        record = _run(runner.run(scenario))
        # Harness run completed, but oracle says incomplete.
        assert record.completed is False
        assert record.failure_reason == "ORACLE_INCOMPLETE"


# ===========================================================================
# Section 26 — BenchmarkRunRecord population
# ===========================================================================


class TestRecordPopulation:
    """BenchmarkRunRecord is populated from real execution results."""

    def test_all_count_fields_populated(self):
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("transient_failure")))
        assert record.scenario_id == "transient_failure"
        assert record.task_id == "fix_calculator_add"
        assert isinstance(record.completed, bool)
        assert record.logical_action_count >= 0
        assert record.tool_invocation_count >= 0
        assert record.tool_attempt_count >= 0
        assert record.retry_count >= 0
        assert record.checkpoint_count >= 0
        assert isinstance(record.events, tuple)

    def test_events_recorded(self):
        runner = _make_reliable()
        record = _run(runner.run(_get_scenario("clean_success")))
        # 4 logical actions → 4 events.
        assert len(record.events) == 4
        for event in record.events:
            assert event.tool_name is not None

    def test_failure_reason_structured(self):
        """Failure reason is a stable string, not a raw exception message."""
        runner = _make_naive()
        record = _run(runner.run(_get_scenario("transient_failure")))
        assert record.failure_reason in (
            "TRANSIENT", "PERMANENT", "TIMEOUT", "EXECUTION",
            "MAX_ACTIONS", "ORACLE_INCOMPLETE",
        )


# ===========================================================================
# Section 50 — Max logical actions (reliable runner)
# ===========================================================================


class TestMaxActionsReliable:
    """The reliable runner also respects max_logical_actions."""

    def test_reliable_max_actions_respected(self):
        task = BenchmarkTask(
            task_id="fix_calculator_add",
            goal="Fix the buggy add() function.",
            scenario_id="max_actions_test",
            max_logical_actions=2,
        )
        scenario = BenchmarkScenario(
            scenario_id="max_actions_test",
            task=task,
            fixture_id="calculator_buggy_add",
            fault_plan_id="none",
        )
        runner = _make_reliable()
        record = _run(runner.run(scenario))
        assert record.completed is False
        assert record.logical_action_count == 2
        assert record.failure_reason == "MAX_ACTIONS"
