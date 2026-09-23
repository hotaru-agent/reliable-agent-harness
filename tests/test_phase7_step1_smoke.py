"""Phase 7 Step 1 smoke test — Controlled Repository Benchmark.

Proves the controlled repository maintenance workload is functional:

1. Create controlled repository (buggy calculator).
2. Oracle says incomplete.
3. Scripted run_tests → failure observed.
4. Scripted read_file → read buggy source.
5. Scripted write_file → write corrected source.
6. Scripted run_tests → all pass.
7. Oracle says complete.

All tools go through existing ToolRegistry / ToolRuntime contract.
All offline, deterministic, no LLM, no network, no real filesystem.
"""

from __future__ import annotations

import asyncio

import pytest

from evaluation.faults import FaultInjector, FaultPlan, FaultSpec, FaultType
from evaluation.invoker import FaultAwareToolInvoker
from evaluation.records import (
    BenchmarkEvent,
    BenchmarkEventType,
    BenchmarkResult,
    BenchmarkRunRecord,
)
from evaluation.repository import ControlledRepository, RepositoryOracle
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
    ToolCall,
    ToolExecutionContext,
    ToolRegistry,
    ToolRuntime,
    ToolSideEffect,
)
from tools.errors import ToolReportedFailure
from tools.models import ToolErrorType


def _run(coro):
    return asyncio.run(coro)


class _FakeSleeper:
    """Deterministic sleep that records durations without actually sleeping."""

    def __init__(self):
        self.sleeps: list[float] = []

    async def __call__(self, seconds: float):
        self.sleeps.append(seconds)


# ===========================================================================
# Section 39 — Controlled Repository Smoke Scenario
# ===========================================================================


class TestControlledRepositorySmokeScenario:
    """The core smoke scenario: fix the calculator bug via scripted actions."""

    def test_smoke_scenario_completes(self):
        """Full smoke scenario:
        1. Create repo (buggy).
        2. Oracle says incomplete.
        3. run_tests → failure.
        4. read_file → read buggy source.
        5. write_file → write corrected source.
        6. run_tests → all pass.
        7. Oracle says complete.
        """
        # --- Setup ---
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        oracle = RepositoryOracle()
        sleeper = _FakeSleeper()

        registry = ToolRegistry()
        specs = make_repository_tool_specs()
        handlers = make_repository_tool_handlers(repo, injector=None)
        for name in specs:
            registry.register(specs[name], handlers[name])

        runtime = ToolRuntime(registry=registry, sleep=sleeper)
        ctx = ToolExecutionContext()
        source = ScriptedActionSource(make_fix_calculator_actions())

        # --- Step 1: Oracle says incomplete ---
        assert oracle.is_complete(repo) is False

        # --- Step 2: run_tests → failure ---
        call = source.next_call()
        assert call.tool_name == "run_tests"
        result = _run(runtime.execute(call, ctx))
        assert result.success is True  # tool executed successfully
        assert result.output["passed"] is False
        assert "test_add" in result.output["failing_tests"]

        # --- Step 3: read_file → read buggy source ---
        call = source.next_call()
        assert call.tool_name == "read_file"
        result = _run(runtime.execute(call, ctx))
        assert result.success is True
        assert "return a - b" in result.output["content"]  # the bug

        # --- Step 4: write_file → write corrected source ---
        call = source.next_call()
        assert call.tool_name == "write_file"
        result = _run(runtime.execute(call, ctx))
        assert result.success is True

        # --- Step 5: run_tests → all pass ---
        call = source.next_call()
        assert call.tool_name == "run_tests"
        result = _run(runtime.execute(call, ctx))
        assert result.success is True
        assert result.output["passed"] is True
        assert result.output["failed_count"] == 0

        # --- Step 6: Oracle says complete ---
        assert oracle.is_complete(repo) is True

        # --- Source exhausted ---
        assert not source.has_next()

    def test_all_tools_go_through_runtime(self):
        """Verify all four repository tools are registered and execute
        through the existing ToolRuntime contract."""
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        registry = ToolRegistry()
        specs = make_repository_tool_specs()
        handlers = make_repository_tool_handlers(repo)
        for name in specs:
            registry.register(specs[name], handlers[name])

        # All four tools are registered.
        assert registry.contains("list_files")
        assert registry.contains("read_file")
        assert registry.contains("write_file")
        assert registry.contains("run_tests")

        runtime = ToolRuntime(registry=registry, sleep=_FakeSleeper())
        ctx = ToolExecutionContext()

        # Execute each tool once.
        for tool_name, args in [
            ("list_files", {}),
            ("read_file", {"path": "calculator.py"}),
            ("run_tests", {}),
        ]:
            result = _run(runtime.execute(ToolCall(tool_name=tool_name, arguments=args), ctx))
            assert result.success is True, f"{tool_name} failed"

    def test_tool_side_effect_metadata(self):
        """Verify side-effect classifications are correct."""
        specs = make_repository_tool_specs()
        assert specs["list_files"].side_effect is ToolSideEffect.READ_ONLY
        assert specs["read_file"].side_effect is ToolSideEffect.READ_ONLY
        assert specs["write_file"].side_effect is ToolSideEffect.IDEMPOTENT
        assert specs["run_tests"].side_effect is ToolSideEffect.READ_ONLY


# ===========================================================================
# Section 40 — Smoke scenario does not compare Naive vs Harness
# ===========================================================================


class TestNoNaiveVsHarnessComparison:
    """This phase does NOT compare Naive Runner vs Reliable Harness.
    It only establishes the workload foundation."""

    def test_smoke_scenario_is_runner_agnostic(self):
        """The scenario fixture, fault plan, and scripted actions are
        independent of any specific runner. They can be reused by
        future runners."""
        scenarios = make_standard_scenarios()
        fixtures = make_standard_fixtures()
        fault_plans = make_standard_fault_plans()

        # All scenarios have valid fixture and fault plan references.
        for scenario in scenarios.values():
            assert scenario.fixture_id in fixtures
            assert scenario.fault_plan_id in fault_plans


# ===========================================================================
# Section 42 — Benchmark determinism
# ===========================================================================


class TestBenchmarkDeterminism:
    def test_same_scenario_produces_same_functional_result(self):
        """Running the same scenario twice produces the same functional
        result (completion status, test results)."""
        for _ in range(2):
            fixture = make_calculator_fixture()
            repo = fixture.create_repository()
            oracle = RepositoryOracle()
            sleeper = _FakeSleeper()

            registry = ToolRegistry()
            specs = make_repository_tool_specs()
            handlers = make_repository_tool_handlers(repo)
            for name in specs:
                registry.register(specs[name], handlers[name])

            runtime = ToolRuntime(registry=registry, sleep=sleeper)
            ctx = ToolExecutionContext()
            source = ScriptedActionSource(make_fix_calculator_actions())

            completed = False
            while source.has_next():
                call = source.next_call()
                result = _run(runtime.execute(call, ctx))
                if not result.success:
                    break
            else:
                completed = oracle.is_complete(repo)

            assert completed is True

    def test_fresh_fixture_each_run(self):
        """Each run gets a fresh repository — no cross-run contamination."""
        fixture = make_calculator_fixture()

        # Run A: fix the bug.
        repo_a = fixture.create_repository()
        repo_a.write_file("calculator.py", CORRECTED_CALCULATOR_CONTENT)
        assert repo_a.run_tests().passed is True

        # Run B: fresh fixture — bug is back.
        repo_b = fixture.create_repository()
        assert repo_b.run_tests().passed is False


# ===========================================================================
# Section 43-44 — Fault consumption and snapshot semantics
# ===========================================================================


class TestFaultConsumptionAndSnapshots:
    def test_fresh_injector_per_run(self):
        """Each run gets a fresh FaultInjector — fault plan is not
        consumed across runs."""
        plan = FaultPlan(
            plan_id="p1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        )

        # Run A: fault fires.
        injector_a = FaultInjector(plan)
        inv = injector_a.next_invocation("run_tests")
        with pytest.raises(ToolReportedFailure):
            _run(injector_a.maybe_inject("run_tests", inv))
        assert injector_a.fired_fault_indices == (0,)

        # Run B: fresh injector — fault fires again.
        injector_b = FaultInjector(plan)
        inv = injector_b.next_invocation("run_tests")
        with pytest.raises(ToolReportedFailure):
            _run(injector_b.maybe_inject("run_tests", inv))
        assert injector_b.fired_fault_indices == (0,)


# ===========================================================================
# Section 20 — Transient failure with retry through ToolRuntime
# ===========================================================================


class TestTransientFailureWithRetry:
    def test_transient_failure_then_retry_succeeds(self):
        """A transient fault on run_tests invocation #1 is retried by
        the ToolRuntime, and the second attempt succeeds."""
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        oracle = RepositoryOracle()
        sleeper = _FakeSleeper()

        plan = FaultPlan(
            plan_id="transient_run_tests_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        )
        injector = FaultInjector(plan)

        registry = ToolRegistry()
        specs = make_repository_tool_specs()
        handlers = make_repository_tool_handlers(repo, injector=injector)
        for name in specs:
            registry.register(specs[name], handlers[name])

        runtime = ToolRuntime(registry=registry, sleep=sleeper)
        invoker = FaultAwareToolInvoker(runtime, injector=injector)
        ctx = ToolExecutionContext()

        # run_tests with transient fault on first logical invocation.
        # The ToolRuntime should retry (attempt 2) and succeed.
        result = _run(
            invoker.execute(
                ToolCall(tool_name="run_tests", arguments={}),
                ctx,
            )
        )

        # The tool should succeed after retry.
        assert result.success is True
        assert result.attempt_count == 2  # first attempt failed, second succeeded
        assert len(result.retry_history) == 1  # one retry
        assert result.output["passed"] is False  # tests still fail (bug not fixed)
        # Logical invocation counter must still be 1 (retry did not advance it).
        assert injector.current_invocation("run_tests") == 0  # ended after execute
        # The next invocation would be #2, not #3.
        assert injector.peek_invocation("run_tests") == 2


# ===========================================================================
# Section 22 — Permanent failure through ToolRuntime
# ===========================================================================


class TestPermanentFailureNoRetry:
    def test_permanent_failure_not_retried(self):
        """A permanent fault on run_tests is not retried."""
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        sleeper = _FakeSleeper()

        plan = FaultPlan(
            plan_id="permanent_run_tests_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.PERMANENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        )
        injector = FaultInjector(plan)

        registry = ToolRegistry()
        specs = make_repository_tool_specs()
        handlers = make_repository_tool_handlers(repo, injector=injector)
        for name in specs:
            registry.register(specs[name], handlers[name])

        runtime = ToolRuntime(registry=registry, sleep=sleeper)
        invoker = FaultAwareToolInvoker(runtime, injector=injector)
        ctx = ToolExecutionContext()

        result = _run(
            invoker.execute(
                ToolCall(tool_name="run_tests", arguments={}),
                ctx,
            )
        )

        assert result.success is False
        assert result.attempt_count == 1  # no retry
        assert result.error is not None
        assert result.error.error_type is ToolErrorType.PERMANENT
        # Logical invocation counter must still be 1 (no retry happened).
        assert injector.current_invocation("run_tests") == 0  # ended after execute
        assert injector.peek_invocation("run_tests") == 2


# ===========================================================================
# Section 21 — Timeout through ToolRuntime
# ===========================================================================


class TestTimeoutThroughRuntime:
    def test_timeout_produces_timeout_error(self):
        """A timeout fault causes the ToolRuntime to produce a TIMEOUT
        error on the first attempt. The retry (attempt 2) succeeds
        because the fault was already consumed.

        This proves the timeout fault goes through the existing
        ToolRuntime timeout boundary (not a fake TIMEOUT error).
        """
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        sleeper = _FakeSleeper()

        plan = FaultPlan(
            plan_id="timeout_run_tests_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TIMEOUT,
                    tool_name="run_tests",
                    logical_invocation=1,
                    timeout_sleep_seconds=10.0,
                ),
            ),
        )
        injector = FaultInjector(plan)

        registry = ToolRegistry()
        # Use a very small timeout so the test is fast.
        specs = make_repository_tool_specs(run_tests_timeout_seconds=0.01)
        handlers = make_repository_tool_handlers(repo, injector=injector)
        for name in specs:
            registry.register(specs[name], handlers[name])

        runtime = ToolRuntime(registry=registry, sleep=sleeper)
        invoker = FaultAwareToolInvoker(runtime, injector=injector)
        ctx = ToolExecutionContext()

        result = _run(
            invoker.execute(
                ToolCall(tool_name="run_tests", arguments={}),
                ctx,
            )
        )

        # The first attempt timed out (TIMEOUT in retry_history),
        # the retry succeeded.
        assert result.success is True
        assert result.attempt_count == 2
        assert len(result.retry_history) == 1
        assert result.retry_history[0].error_type is ToolErrorType.TIMEOUT
        # Logical invocation counter must still be 1 (retry did not advance it).
        assert injector.current_invocation("run_tests") == 0  # ended after execute
        assert injector.peek_invocation("run_tests") == 2


# ===========================================================================
# Section 23 — Large output
# ===========================================================================


class TestLargeOutput:
    def test_large_output_returned(self):
        """A large output fault returns a deterministic large payload."""
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        sleeper = _FakeSleeper()

        plan = FaultPlan(
            plan_id="large_output_run_tests_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.LARGE_OUTPUT,
                    tool_name="run_tests",
                    logical_invocation=1,
                    large_output_size=4096,
                ),
            ),
        )
        injector = FaultInjector(plan)

        registry = ToolRegistry()
        specs = make_repository_tool_specs()
        handlers = make_repository_tool_handlers(repo, injector=injector)
        for name in specs:
            registry.register(specs[name], handlers[name])

        runtime = ToolRuntime(registry=registry, sleep=sleeper)
        invoker = FaultAwareToolInvoker(runtime, injector=injector)
        ctx = ToolExecutionContext()

        result = _run(
            invoker.execute(
                ToolCall(tool_name="run_tests", arguments={}),
                ctx,
            )
        )

        assert result.success is True
        assert isinstance(result.output, str)
        assert len(result.output) == 4096
        # Logical invocation counter must still be 1 (no retry for large output).
        assert injector.current_invocation("run_tests") == 0  # ended after execute
        assert injector.peek_invocation("run_tests") == 2


# ===========================================================================
# Section 24 — Process interruption
# ===========================================================================


class TestProcessInterruption:
    def test_interruption_propagates_as_base_exception(self):
        """A process interruption fault raises
        InjectedProcessInterruption (a BaseException, not Exception)."""
        from evaluation.faults import InjectedProcessInterruption

        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        sleeper = _FakeSleeper()

        plan = FaultPlan(
            plan_id="interruption_run_tests_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.PROCESS_INTERRUPTION,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        )
        injector = FaultInjector(plan)

        registry = ToolRegistry()
        specs = make_repository_tool_specs()
        handlers = make_repository_tool_handlers(repo, injector=injector)
        for name in specs:
            registry.register(specs[name], handlers[name])

        runtime = ToolRuntime(registry=registry, sleep=sleeper)
        invoker = FaultAwareToolInvoker(runtime, injector=injector)
        ctx = ToolExecutionContext()

        with pytest.raises(InjectedProcessInterruption):
            _run(
                invoker.execute(
                    ToolCall(tool_name="run_tests", arguments={}),
                    ctx,
                )
            )
        # Even though the exception propagated, the logical invocation
        # scope must have been cleaned up by the invoker's finally block.
        assert injector.current_invocation("run_tests") == 0
        assert injector.peek_invocation("run_tests") == 2


# ===========================================================================
# Section 52 — Benchmark record from smoke scenario
# ===========================================================================


class TestBenchmarkRecordFromSmoke:
    def test_record_built_from_smoke_run(self):
        """Build a BenchmarkRunRecord from the smoke scenario run."""
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        oracle = RepositoryOracle()
        sleeper = _FakeSleeper()

        registry = ToolRegistry()
        specs = make_repository_tool_specs()
        handlers = make_repository_tool_handlers(repo)
        for name in specs:
            registry.register(specs[name], handlers[name])

        runtime = ToolRuntime(registry=registry, sleep=sleeper)
        ctx = ToolExecutionContext()
        source = ScriptedActionSource(make_fix_calculator_actions())

        events: list[BenchmarkEvent] = []
        seq = 0
        tool_invocations = 0
        tool_attempts = 0
        retries = 0

        while source.has_next():
            call = source.next_call()
            tool_invocations += 1
            result = _run(runtime.execute(call, ctx))
            tool_attempts += result.attempt_count
            retries += max(0, result.attempt_count - 1)

            events.append(BenchmarkEvent(
                event_type=BenchmarkEventType.LOGICAL_ACTION,
                sequence=seq,
                tool_name=call.tool_name,
                success=result.success,
            ))
            seq += 1

        completed = oracle.is_complete(repo)

        record = BenchmarkRunRecord(
            scenario_id="clean_success",
            task_id="fix_calculator_add",
            completed=completed,
            logical_action_count=len(events),
            tool_invocation_count=tool_invocations,
            tool_attempt_count=tool_attempts,
            retry_count=retries,
            events=tuple(events),
        )

        assert record.completed is True
        assert record.logical_action_count == 4
        assert record.tool_invocation_count == 4
        assert record.tool_attempt_count == 4  # no retries in clean run
        assert record.retry_count == 0

        result = BenchmarkResult.from_record(record)
        assert result.task_completed is True
        assert result.steps_to_completion == 4


# ===========================================================================
# Section 26 — Standard scenario fixtures exist
# ===========================================================================


class TestStandardScenarios:
    def test_all_standard_scenarios_exist(self):
        scenarios = make_standard_scenarios()
        expected = {
            "clean_success",
            "transient_failure",
            "permanent_failure",
            "timeout",
            "large_output",
            "interruption",
            "checkpoint_recovery",
            "loop_replan",
            "context_growth",
            "large_output_externalization",
        }
        assert set(scenarios.keys()) == expected

    def test_all_standard_fault_plans_exist(self):
        plans = make_standard_fault_plans()
        expected = {
            "none",
            "transient_run_tests_1",
            "permanent_run_tests_1",
            "timeout_run_tests_1",
            "large_output_run_tests_1",
            "interruption_run_tests_1",
            "interruption_read_file_1",
        }
        assert set(plans.keys()) == expected

    def test_all_standard_fixtures_exist(self):
        fixtures = make_standard_fixtures()
        assert "calculator_buggy_add" in fixtures
        assert "calculator_buggy_add_large_log" in fixtures

    def test_scenario_fixture_fault_plan_association(self):
        scenarios = make_standard_scenarios()
        fixtures = make_standard_fixtures()
        plans = make_standard_fault_plans()

        for scenario in scenarios.values():
            assert scenario.fixture_id in fixtures
            assert scenario.fault_plan_id in plans


# ===========================================================================
# Section 15 — ScriptedActionSource is not an AI agent
# ===========================================================================


class TestScriptedActionSourceNotAI:
    def test_action_source_is_deterministic(self):
        """ScriptedActionSource produces the same sequence every time."""
        actions = make_fix_calculator_actions()
        source1 = ScriptedActionSource(actions)
        source2 = ScriptedActionSource(actions)

        calls1 = [c for c in source1]
        calls2 = [c for c in source2]

        assert len(calls1) == len(calls2)
        for c1, c2 in zip(calls1, calls2):
            assert c1.tool_name == c2.tool_name
            assert c1.arguments == c2.arguments

    def test_action_source_reset(self):
        source = ScriptedActionSource(make_fix_calculator_actions())
        first_call = source.next_call()
        assert first_call.tool_name == "run_tests"

        source.reset()
        reset_call = source.next_call()
        assert reset_call.tool_name == "run_tests"

    def test_action_source_exhaustion(self):
        source = ScriptedActionSource(make_fix_calculator_actions())
        for _ in range(4):
            source.next_call()
        assert not source.has_next()
        with pytest.raises(StopIteration):
            source.next_call()

    def test_empty_actions_rejected(self):
        with pytest.raises(ValueError, match="actions"):
            ScriptedActionSource([])


# ===========================================================================
# Section 11 — Oracle determines completion
# ===========================================================================


class TestOracleCompletion:
    def test_oracle_says_incomplete_for_bug(self):
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        oracle = RepositoryOracle()
        assert oracle.is_complete(repo) is False

    def test_oracle_says_complete_after_fix(self):
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        repo.write_file("calculator.py", CORRECTED_CALCULATOR_CONTENT)
        oracle = RepositoryOracle()
        assert oracle.is_complete(repo) is True

    def test_oracle_says_incomplete_for_partial_fix(self):
        """If we write a file that doesn't fix the bug, oracle still
        says incomplete."""
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        repo.write_file("calculator.py", "def add(a, b):\n    return a * b\n")
        oracle = RepositoryOracle()
        assert oracle.is_complete(repo) is False
