"""Phase 7 Step 4 — Loop Detection, Replan Gate & Loop Escape Benchmark
tests.

Covers:

* Progress provider (Section 58)
* Loop detection integration (Section 59)
* Replan gate (Section 60)
* Replan (Section 61)
* Post-replan execution (Section 62)
* Naive loop behavior (Section 63)
* Reliable loop escape (Section 64)
* Fairness (Section 65)
* Determinism (Section 66)
* Existing regression (Section 67)

All offline, deterministic, no LLM, no network, no real filesystem.
"""

from __future__ import annotations

import asyncio

import pytest

from evaluation.loop_workload import (
    DeterministicReplanStrategy,
    LoopBenchmarkStepExecutor,
    ReplanAwareActionSource,
    ReplanPlan,
    RepositoryProgressProvider,
    make_loop_looping_call,
    make_loop_repair_calls,
)
from evaluation.repository import ControlledRepository, RepositoryOracle
from evaluation.runners import (
    NaiveBenchmarkRunner,
    ReliableHarnessBenchmarkRunner,
)
from evaluation.scenarios.controlled_repo import (
    CORRECTED_CALCULATOR_CONTENT,
    make_calculator_fixture,
    make_standard_scenarios,
)
from harness.action_controller import (
    ActionControllerState,
    AgentActionController,
    ReplanRequiredError,
)
from harness.loop_detection import (
    ActionHistory,
    LoopDetectionReason,
    LoopDetector,
    LoopDetectorConfig,
)
from evaluation.workloads import (
    LoopWorkload,
    ToolStack,
    WorkloadRegistry,
    make_default_workload_registry,
)
from tools import (
    RetryPolicy,
    ToolCall,
    ToolExecutionContext,
    ToolRegistry,
    ToolRuntime,
)
from evaluation.scenarios.controlled_repo import (
    make_repository_tool_handlers,
    make_repository_tool_specs,
)
from evaluation.faults import FaultInjector, FaultPlan
from evaluation.invoker import FaultAwareToolInvoker


# ===========================================================================
# Helpers
# ===========================================================================


def _run(coro):
    return asyncio.run(coro)


def _make_loop_scenario():
    """Return the loop_replan scenario."""
    scenarios = make_standard_scenarios()
    return scenarios["loop_replan"]


def _make_fresh_repo():
    """Return a fresh buggy calculator repository."""
    fixture = make_calculator_fixture()
    return fixture.create_repository()


def _make_tool_stack(repo, *, sleep=None):
    """Build a ToolStack for the loop scenario (no faults)."""
    injector = FaultInjector(FaultPlan.no_faults())
    registry = ToolRegistry()
    specs = make_repository_tool_specs()
    handlers = make_repository_tool_handlers(repo, injector=injector)
    for name in specs:
        registry.register(specs[name], handlers[name])
    runtime = ToolRuntime(registry=registry, sleep=sleep or asyncio.sleep)
    invoker = FaultAwareToolInvoker(runtime, injector=injector)
    ctx = ToolExecutionContext()
    return ToolStack(runtime=runtime, invoker=invoker, ctx=ctx), injector


class _FakeSleeper:
    async def __call__(self, seconds: float) -> None:
        pass


# ===========================================================================
# Section 58 — Progress Provider
# ===========================================================================


class TestRepositoryProgressProvider:
    def test_same_unchanged_repo_same_token(self):
        """Repeated read_file on unchanged repo → same progress token."""
        repo = _make_fresh_repo()
        provider = RepositoryProgressProvider(repo)
        call = make_loop_looping_call()

        token1 = provider.get_progress_token(call, None)
        token2 = provider.get_progress_token(call, None)
        token3 = provider.get_progress_token(call, None)

        assert token1 == token2 == token3
        assert token1 == "failing-tests:1"

    def test_write_file_fix_changes_token(self):
        """After write_file with corrected content, token changes."""
        repo = _make_fresh_repo()
        provider = RepositoryProgressProvider(repo)
        call = make_loop_looping_call()

        token_before = provider.get_progress_token(call, None)
        assert token_before == "failing-tests:1"

        repo.write_file("calculator.py", CORRECTED_CALCULATOR_CONTENT)

        token_after = provider.get_progress_token(call, None)
        assert token_after == "failing-tests:0"
        assert token_before != token_after

    def test_provider_deterministic(self):
        """Same repo state → same token, every time."""
        repo1 = _make_fresh_repo()
        repo2 = _make_fresh_repo()
        provider1 = RepositoryProgressProvider(repo1)
        provider2 = RepositoryProgressProvider(repo2)
        call = make_loop_looping_call()

        assert provider1.get_progress_token(call, None) == \
            provider2.get_progress_token(call, None)

    def test_provider_call_count_increments(self):
        """Provider call_count increments per call."""
        repo = _make_fresh_repo()
        provider = RepositoryProgressProvider(repo)
        call = make_loop_looping_call()

        assert provider.call_count == 0
        provider.get_progress_token(call, None)
        assert provider.call_count == 1
        provider.get_progress_token(call, None)
        assert provider.call_count == 2


# ===========================================================================
# Section 59 — Loop Detection (benchmark integration)
# ===========================================================================


class TestLoopDetectionIntegration:
    def test_loop_replan_scenario_exists(self):
        scenarios = make_standard_scenarios()
        assert "loop_replan" in scenarios
        s = scenarios["loop_replan"]
        assert s.fault_plan_id == "none"
        assert s.task.max_logical_actions == 6

    def test_first_two_reads_no_loop(self):
        """First two read_file actions do not trigger loop detection."""
        repo = _make_fresh_repo()
        tool_stack, _ = _make_tool_stack(repo, sleep=_FakeSleeper())
        workload = LoopWorkload()
        source = workload.create_source()
        executor = workload.create_step_executor(
            source=source,
            tool_stack=tool_stack,
            repo=repo,
            max_logical_actions=6,
        )

        # Execute two steps.
        from harness.execution import ExecutionState
        from harness.state import Run, RunStatus, Task, TaskStatus

        task = Task(task_id="t", goal="g", created_at=None)
        run = Run(run_id="R-001", task_id="t")
        state = ExecutionState()

        _run(executor.execute_step(task, run, state))
        assert executor.loop_detection_count == 0

        _run(executor.execute_step(task, run, state))
        assert executor.loop_detection_count == 0

    def test_third_read_triggers_duplicate_call(self):
        """Third read_file triggers DUPLICATE_CALL loop detection."""
        repo = _make_fresh_repo()
        tool_stack, _ = _make_tool_stack(repo, sleep=_FakeSleeper())
        workload = LoopWorkload()
        source = workload.create_source()
        executor = workload.create_step_executor(
            source=source,
            tool_stack=tool_stack,
            repo=repo,
            max_logical_actions=6,
        )

        from harness.execution import ExecutionState
        from harness.state import Run, Task

        task = Task(task_id="t", goal="g", created_at=None)
        run = Run(run_id="R-001", task_id="t")
        state = ExecutionState()

        _run(executor.execute_step(task, run, state))
        _run(executor.execute_step(task, run, state))
        _run(executor.execute_step(task, run, state))

        assert executor.loop_detection_count == 1

    def test_replan_signal_exists(self):
        """ReplanSignal is captured after loop detection."""
        repo = _make_fresh_repo()
        tool_stack, _ = _make_tool_stack(repo, sleep=_FakeSleeper())
        workload = LoopWorkload()
        source = workload.create_source()
        executor = workload.create_step_executor(
            source=source,
            tool_stack=tool_stack,
            repo=repo,
            max_logical_actions=6,
        )

        from harness.execution import ExecutionState
        from harness.state import Run, Task

        task = Task(task_id="t", goal="g", created_at=None)
        run = Run(run_id="R-001", task_id="t")
        state = ExecutionState()

        _run(executor.execute_step(task, run, state))
        _run(executor.execute_step(task, run, state))
        _run(executor.execute_step(task, run, state))

        signal = executor.last_replan_signal
        assert signal is not None

    def test_replan_signal_reason_correct(self):
        """Signal reason is DUPLICATE_CALL."""
        repo = _make_fresh_repo()
        tool_stack, _ = _make_tool_stack(repo, sleep=_FakeSleeper())
        workload = LoopWorkload()
        source = workload.create_source()
        executor = workload.create_step_executor(
            source=source,
            tool_stack=tool_stack,
            repo=repo,
            max_logical_actions=6,
        )

        from harness.execution import ExecutionState
        from harness.state import Run, Task

        task = Task(task_id="t", goal="g", created_at=None)
        run = Run(run_id="R-001", task_id="t")
        state = ExecutionState()

        _run(executor.execute_step(task, run, state))
        _run(executor.execute_step(task, run, state))
        _run(executor.execute_step(task, run, state))

        assert executor.last_replan_signal.reason is \
            LoopDetectionReason.DUPLICATE_CALL

    def test_replan_signal_evidence_correct(self):
        """Signal evidence contains (0, 1, 2)."""
        repo = _make_fresh_repo()
        tool_stack, _ = _make_tool_stack(repo, sleep=_FakeSleeper())
        workload = LoopWorkload()
        source = workload.create_source()
        executor = workload.create_step_executor(
            source=source,
            tool_stack=tool_stack,
            repo=repo,
            max_logical_actions=6,
        )

        from harness.execution import ExecutionState
        from harness.state import Run, Task

        task = Task(task_id="t", goal="g", created_at=None)
        run = Run(run_id="R-001", task_id="t")
        state = ExecutionState()

        _run(executor.execute_step(task, run, state))
        _run(executor.execute_step(task, run, state))
        _run(executor.execute_step(task, run, state))

        assert executor.last_replan_signal.evidence_action_indices == (0, 1, 2)

    def test_controller_state_after_detection(self):
        """After loop detection + replan, controller is READY."""
        repo = _make_fresh_repo()
        tool_stack, _ = _make_tool_stack(repo, sleep=_FakeSleeper())
        workload = LoopWorkload()
        source = workload.create_source()
        executor = workload.create_step_executor(
            source=source,
            tool_stack=tool_stack,
            repo=repo,
            max_logical_actions=6,
        )

        from harness.execution import ExecutionState
        from harness.state import Run, Task

        task = Task(task_id="t", goal="g", created_at=None)
        run = Run(run_id="R-001", task_id="t")
        state = ExecutionState()

        _run(executor.execute_step(task, run, state))
        _run(executor.execute_step(task, run, state))
        _run(executor.execute_step(task, run, state))

        # After the third step (which includes gate probe + replan +
        # acknowledge), controller should be READY.
        assert executor._controller.state is ActionControllerState.READY
        assert executor._controller.pending_replan_signal is None


# ===========================================================================
# Section 60 — Gate
# ===========================================================================


class TestReplanGate:
    def test_blocked_call_count_is_one(self):
        """Exactly one blocked call (the gate probe)."""
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        _run(runner.run(scenario))

        executor = runner.last_trial_state.source_executor
        assert executor.blocked_call_count == 1

    def test_blocked_call_not_in_execution_history(self):
        """The blocked call does not appear in execution history."""
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        _run(runner.run(scenario))

        executor = runner.last_trial_state.source_executor
        # 5 actions: 3 reads + 1 write + 1 run_tests
        assert len(executor.execution_history) == 5
        # All entries are success
        for _, _, _, outcome in executor.execution_history:
            assert outcome == "success"

    def test_blocked_call_does_not_increment_action_count(self):
        """Blocked call does not count as logical action."""
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        rec = _run(runner.run(scenario))

        # 5 logical actions, not 6
        assert rec.logical_action_count == 5
        assert rec.tool_invocation_count == 5

    def test_gate_event_present(self):
        """REPLAN_BLOCKED event is present in events."""
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        rec = _run(runner.run(scenario))

        from evaluation.records import BenchmarkEventType
        event_types = [e.event_type for e in rec.events]
        assert BenchmarkEventType.REPLAN_BLOCKED in event_types

    def test_gate_blocks_before_toolruntime(self):
        """Gate blocks the call before ToolRuntime executes it.

        We verify this by checking that the handler call count equals
        the number of successful logical actions (5), NOT 6 (which
        would include the blocked call).
        """
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        rec = _run(runner.run(scenario))

        # tool_attempt_count == tool_invocation_count == 5
        # If the gate didn't block, we'd have 6 invocations.
        assert rec.tool_attempt_count == 5
        assert rec.tool_invocation_count == 5


# ===========================================================================
# Section 61 — Replan
# ===========================================================================


class TestReplan:
    def test_deterministic_strategy_produces_plan(self):
        """DeterministicReplanStrategy produces a ReplanPlan."""
        strategy = DeterministicReplanStrategy()
        signal = type(
            "FakeSignal",
            (),
            {"reason": LoopDetectionReason.DUPLICATE_CALL,
             "evidence_action_indices": (0, 1, 2),
             "message": "test"},
        )()
        # ReplanSignal is frozen, need to construct properly
        from harness.action_controller import ReplanSignal
        signal = ReplanSignal(
            reason=LoopDetectionReason.DUPLICATE_CALL,
            evidence_action_indices=(0, 1, 2),
            message="test",
        )
        plan = strategy.replan(signal)
        assert isinstance(plan, ReplanPlan)

    def test_source_enters_repair_phase(self):
        """After replan, source phase is REPAIR."""
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        _run(runner.run(scenario))

        executor = runner.last_trial_state.source_executor
        assert executor._source.phase == "REPAIR"

    def test_acknowledge_replan_called(self):
        """Controller is READY after replan (acknowledge was called)."""
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        _run(runner.run(scenario))

        executor = runner.last_trial_state.source_executor
        assert executor._controller.state is ActionControllerState.READY
        assert executor._controller.pending_replan_signal is None

    def test_detector_episode_reset(self):
        """LoopDetector episode is reset after acknowledge.

        After acknowledge_replan(), the detector history is cleared.
        New post-replan actions (write_file, run_tests) are observed
        fresh — they do NOT trigger another loop detection from the
        old 3-read evidence.
        """
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        rec = _run(runner.run(scenario))

        # Only 1 loop detection (not 2) — old evidence was cleared.
        assert rec.loop_detection_count == 1

        executor = runner.last_trial_state.source_executor
        detector = executor._controller.loop_detector
        # History has only the 2 post-replan records (write_file,
        # run_tests), not the old 3 read_file records.
        assert len(detector.history) == 2
        # Verify the remaining records are post-replan actions.
        for record in detector.history.records():
            assert record.fingerprint.tool_name in ("write_file", "run_tests")

    def test_replan_count_is_one(self):
        """Reliable replan_count == 1."""
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        rec = _run(runner.run(scenario))
        assert rec.replan_count == 1

    def test_replan_acknowledged_event_present(self):
        """REPLAN_ACKNOWLEDGED event is present."""
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        rec = _run(runner.run(scenario))

        from evaluation.records import BenchmarkEventType
        event_types = [e.event_type for e in rec.events]
        assert BenchmarkEventType.REPLAN_ACKNOWLEDGED in event_types


# ===========================================================================
# Section 62 — Post-Replan Execution
# ===========================================================================


class TestPostReplanExecution:
    def test_write_file_executes(self):
        """write_file appears in execution history after replan."""
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        _run(runner.run(scenario))

        executor = runner.last_trial_state.source_executor
        tool_names = [h[2] for h in executor.execution_history]
        assert "write_file" in tool_names

    def test_repository_state_changes(self):
        """Repository is repaired after write_file."""
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        _run(runner.run(scenario))

        repo = runner.last_trial_state.repo
        oracle = RepositoryOracle()
        assert oracle.is_complete(repo)

    def test_run_tests_executes(self):
        """run_tests appears in execution history after replan."""
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        _run(runner.run(scenario))

        executor = runner.last_trial_state.source_executor
        tool_names = [h[2] for h in executor.execution_history]
        assert "run_tests" in tool_names

    def test_oracle_complete(self):
        """Oracle is complete after Reliable run."""
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        rec = _run(runner.run(scenario))
        assert rec.completed is True

    def test_post_replan_indices_continue_monotonically(self):
        """Action indices continue after replan (no reset to 0)."""
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        _run(runner.run(scenario))

        executor = runner.last_trial_state.source_executor
        controller = executor._controller
        # After 5 actions (3 reads + 1 write + 1 run_tests),
        # next_action_index should be 5.
        assert controller.next_action_index == 5

    def test_execution_history_order(self):
        """Execution history has correct order: 3 reads, write, run_tests."""
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        _run(runner.run(scenario))

        executor = runner.last_trial_state.source_executor
        tool_names = [h[2] for h in executor.execution_history]
        assert tool_names == ["read_file", "read_file", "read_file",
                              "write_file", "run_tests"]


# ===========================================================================
# Section 63 — Naive Loop Behavior
# ===========================================================================


class TestNaiveLoop:
    def test_naive_loops_until_max_actions(self):
        """Naive loops until max_logical_actions."""
        scenario = _make_loop_scenario()
        runner = NaiveBenchmarkRunner()
        rec = _run(runner.run(scenario))

        assert rec.logical_action_count == 6
        assert rec.completed is False

    def test_naive_no_replan(self):
        """Naive replan_count == 0."""
        scenario = _make_loop_scenario()
        runner = NaiveBenchmarkRunner()
        rec = _run(runner.run(scenario))
        assert rec.replan_count == 0

    def test_naive_no_loop_detection(self):
        """Naive loop_detection_count == 0."""
        scenario = _make_loop_scenario()
        runner = NaiveBenchmarkRunner()
        rec = _run(runner.run(scenario))
        assert rec.loop_detection_count == 0

    def test_naive_failure_reason_max_actions(self):
        """Naive failure_reason == MAX_ACTIONS."""
        scenario = _make_loop_scenario()
        runner = NaiveBenchmarkRunner()
        rec = _run(runner.run(scenario))
        assert rec.failure_reason == "MAX_ACTIONS"

    def test_naive_oracle_incomplete(self):
        """Naive oracle is incomplete."""
        scenario = _make_loop_scenario()
        runner = NaiveBenchmarkRunner()
        rec = _run(runner.run(scenario))
        assert rec.completed is False

    def test_naive_all_read_file(self):
        """Naive executes only read_file (stuck in looping phase)."""
        scenario = _make_loop_scenario()
        runner = NaiveBenchmarkRunner()
        rec = _run(runner.run(scenario))

        # All 6 actions should be read_file.
        read_file_events = [
            e for e in rec.events
            if e.tool_name == "read_file"
        ]
        assert len(read_file_events) == 6


# ===========================================================================
# Section 64 — Reliable Loop Escape
# ===========================================================================


class TestReliableLoopEscape:
    def test_reliable_completes(self):
        """Reliable completes the loop_replan scenario."""
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        rec = _run(runner.run(scenario))
        assert rec.completed is True

    def test_reliable_loop_detection_count_one(self):
        """Reliable loop_detection_count == 1."""
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        rec = _run(runner.run(scenario))
        assert rec.loop_detection_count == 1

    def test_reliable_replan_count_one(self):
        """Reliable replan_count == 1."""
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        rec = _run(runner.run(scenario))
        assert rec.replan_count == 1

    def test_reliable_within_budget(self):
        """Reliable action count < max_logical_actions."""
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        rec = _run(runner.run(scenario))
        assert rec.logical_action_count == 5
        assert rec.logical_action_count < scenario.task.max_logical_actions

    def test_reliable_no_failure_reason(self):
        """Reliable has no failure reason."""
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        rec = _run(runner.run(scenario))
        assert rec.failure_reason is None

    def test_reliable_blocked_call_did_not_execute(self):
        """Blocked call did not execute (handler count == 5, not 6)."""
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        rec = _run(runner.run(scenario))
        # 5 invocations, not 6 (blocked call not counted)
        assert rec.tool_invocation_count == 5

    def test_loop_escape_true_for_reliable(self):
        """Loop Escape metric is True for Reliable.

        Loop Escape = loop detected AND replan/gate handled AND
        execution leaves looping policy AND task eventually completes.
        """
        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner()
        rec = _run(runner.run(scenario))

        loop_detected = rec.loop_detection_count > 0
        replan_handled = rec.replan_count > 0
        task_completed = rec.completed
        # "leaves looping policy" = write_file or run_tests in history
        executor = runner.last_trial_state.source_executor
        left_loop = any(
            h[2] not in ("read_file",) for h in executor.execution_history
        )

        loop_escape = loop_detected and replan_handled and left_loop and task_completed
        assert loop_escape is True

    def test_loop_escape_false_for_naive(self):
        """Loop Escape metric is False for Naive."""
        scenario = _make_loop_scenario()
        runner = NaiveBenchmarkRunner()
        rec = _run(runner.run(scenario))

        loop_detected = rec.loop_detection_count > 0
        replan_handled = rec.replan_count > 0
        task_completed = rec.completed

        loop_escape = loop_detected and replan_handled and task_completed
        assert loop_escape is False


# ===========================================================================
# Section 65 — Fairness
# ===========================================================================


class TestFairness:
    def test_same_workload_blueprint(self):
        """Both runners use the same workload blueprint."""
        scenario = _make_loop_scenario()
        registry = make_default_workload_registry(
            action_factory=lambda: []
        )
        workload = registry.get(scenario.scenario_id)
        assert isinstance(workload, LoopWorkload)

    def test_same_max_logical_actions(self):
        """Both runners have the same max_logical_actions."""
        scenario = _make_loop_scenario()
        assert scenario.task.max_logical_actions == 6

    def test_same_initial_repo(self):
        """Both runners start with the same buggy calculator fixture."""
        scenario = _make_loop_scenario()
        assert scenario.fixture_id == "calculator_buggy_add"

    def test_same_fault_plan(self):
        """Both runners use the same no-fault plan."""
        scenario = _make_loop_scenario()
        assert scenario.fault_plan_id == "none"

    def test_fresh_state_per_runner(self):
        """Each runner gets fresh mutable state."""
        scenario = _make_loop_scenario()
        naive = NaiveBenchmarkRunner()
        reliable = ReliableHarnessBenchmarkRunner()

        n_rec = _run(naive.run(scenario))
        r_rec = _run(reliable.run(scenario))

        # Both should produce results (fresh state, no leakage).
        assert n_rec.scenario_id == "loop_replan"
        assert r_rec.scenario_id == "loop_replan"

    def test_comparison_order_independence(self):
        """Order of naive → reliable vs reliable → naive doesn't matter."""
        scenario = _make_loop_scenario()

        # naive → reliable
        naive1 = NaiveBenchmarkRunner()
        reliable1 = ReliableHarnessBenchmarkRunner()
        n1 = _run(naive1.run(scenario))
        r1 = _run(reliable1.run(scenario))

        # reliable → naive
        reliable2 = ReliableHarnessBenchmarkRunner()
        naive2 = NaiveBenchmarkRunner()
        r2 = _run(reliable2.run(scenario))
        n2 = _run(naive2.run(scenario))

        assert n1.completed == n2.completed
        assert n1.logical_action_count == n2.logical_action_count
        assert n1.failure_reason == n2.failure_reason

        assert r1.completed == r2.completed
        assert r1.logical_action_count == r2.logical_action_count
        assert r1.loop_detection_count == r2.loop_detection_count
        assert r1.replan_count == r2.replan_count


# ===========================================================================
# Section 66 — Determinism
# ===========================================================================


class TestDeterminism:
    def test_reliable_repeated_same_results(self):
        """Repeated Reliable runs produce same results (except IDs)."""
        scenario = _make_loop_scenario()

        results = []
        for _ in range(3):
            runner = ReliableHarnessBenchmarkRunner()
            rec = _run(runner.run(scenario))
            results.append(rec)

        for rec in results:
            assert rec.completed is True
            assert rec.logical_action_count == 5
            assert rec.loop_detection_count == 1
            assert rec.replan_count == 1
            assert rec.failure_reason is None

    def test_naive_repeated_same_results(self):
        """Repeated Naive runs produce same results."""
        scenario = _make_loop_scenario()

        results = []
        for _ in range(3):
            runner = NaiveBenchmarkRunner()
            rec = _run(runner.run(scenario))
            results.append(rec)

        for rec in results:
            assert rec.completed is False
            assert rec.logical_action_count == 6
            assert rec.loop_detection_count == 0
            assert rec.replan_count == 0
            assert rec.failure_reason == "MAX_ACTIONS"

    def test_reliable_same_controller_evidence(self):
        """Repeated Reliable runs produce same controller evidence."""
        scenario = _make_loop_scenario()

        evidences = []
        for _ in range(3):
            runner = ReliableHarnessBenchmarkRunner()
            _run(runner.run(scenario))
            executor = runner.last_trial_state.source_executor
            evidences.append(executor.last_replan_signal.evidence_action_indices)

        assert all(e == (0, 1, 2) for e in evidences)

    def test_reliable_same_final_repo_state(self):
        """Repeated Reliable runs produce same final repo state."""
        scenario = _make_loop_scenario()

        for _ in range(3):
            runner = ReliableHarnessBenchmarkRunner()
            _run(runner.run(scenario))
            repo = runner.last_trial_state.repo
            oracle = RepositoryOracle()
            assert oracle.is_complete(repo)


# ===========================================================================
# Section 67 — Existing Regression (5-scenario matrix unchanged)
# ===========================================================================


class TestExistingRegression:
    """Verify the 5 existing scenarios still produce the same results."""

    def test_clean_success(self):
        from evaluation.scenarios.controlled_repo import make_standard_scenarios
        scenarios = make_standard_scenarios()
        s = scenarios["clean_success"]

        naive = NaiveBenchmarkRunner()
        reliable = ReliableHarnessBenchmarkRunner()
        n = _run(naive.run(s))
        r = _run(reliable.run(s))

        assert n.completed is True
        assert r.completed is True

    def test_transient_failure(self):
        from evaluation.scenarios.controlled_repo import make_standard_scenarios
        scenarios = make_standard_scenarios()
        s = scenarios["transient_failure"]

        naive = NaiveBenchmarkRunner()
        reliable = ReliableHarnessBenchmarkRunner()
        n = _run(naive.run(s))
        r = _run(reliable.run(s))

        assert n.completed is False
        assert r.completed is True

    def test_timeout(self):
        from evaluation.scenarios.controlled_repo import make_standard_scenarios
        scenarios = make_standard_scenarios()
        s = scenarios["timeout"]

        naive = NaiveBenchmarkRunner()
        reliable = ReliableHarnessBenchmarkRunner()
        n = _run(naive.run(s))
        r = _run(reliable.run(s))

        assert n.completed is False
        assert r.completed is True

    def test_permanent_failure(self):
        from evaluation.scenarios.controlled_repo import make_standard_scenarios
        scenarios = make_standard_scenarios()
        s = scenarios["permanent_failure"]

        naive = NaiveBenchmarkRunner()
        reliable = ReliableHarnessBenchmarkRunner()
        n = _run(naive.run(s))
        r = _run(reliable.run(s))

        assert n.completed is False
        assert r.completed is False

    def test_checkpoint_recovery(self):
        from evaluation.scenarios.controlled_repo import make_standard_scenarios
        scenarios = make_standard_scenarios()
        s = scenarios["checkpoint_recovery"]

        naive = NaiveBenchmarkRunner()
        reliable = ReliableHarnessBenchmarkRunner()
        n = _run(naive.run(s))
        r = _run(reliable.run(s))

        assert n.completed is False
        assert r.completed is True
        assert r.resume_count == 1


# ===========================================================================
# Section 52 — New Matrix: loop_replan adds N FAIL / R PASS
# ===========================================================================


class TestScenarioMatrix:
    def test_loop_replan_naive_fail_reliable_pass(self):
        """loop_replan: Naive FAIL, Reliable PASS."""
        scenario = _make_loop_scenario()

        naive = NaiveBenchmarkRunner()
        reliable = ReliableHarnessBenchmarkRunner()
        n = _run(naive.run(scenario))
        r = _run(reliable.run(scenario))

        assert n.completed is False
        assert r.completed is True

    def test_full_matrix(self):
        """Full 6-scenario matrix: clean, transient, timeout, permanent,
        checkpoint_recovery, loop_replan."""
        scenarios = make_standard_scenarios()

        # Expected results: (naive_completed, reliable_completed)
        expected = {
            "clean_success": (True, True),
            "transient_failure": (False, True),
            "timeout": (False, True),
            "permanent_failure": (False, False),
            "checkpoint_recovery": (False, True),
            "loop_replan": (False, True),
        }

        for scenario_id, (n_expected, r_expected) in expected.items():
            s = scenarios[scenario_id]
            naive = NaiveBenchmarkRunner()
            reliable = ReliableHarnessBenchmarkRunner()
            n = _run(naive.run(s))
            r = _run(reliable.run(s))
            assert n.completed is n_expected, (
                f"{scenario_id}: naive expected {n_expected}, got {n.completed}"
            )
            assert r.completed is r_expected, (
                f"{scenario_id}: reliable expected {r_expected}, got {r.completed}"
            )


# ===========================================================================
# Section 52 — Reliable failure case (negative test)
# ===========================================================================


class TestReliableFailureProtection:
    def test_reliable_fails_if_no_replan_possible(self):
        """If loop detected but replan strategy can't produce a plan,
        Reliable should fail deterministically.

        We simulate this by using a workload where the source has no
        repair actions (empty repair branch). The replan still fires,
        but the source immediately exhausts, producing COMPLETE without
        repairing the repo. The oracle then says incomplete.
        """
        # Create a custom workload with empty repair branch.
        from evaluation.workloads import LoopWorkload
        from evaluation.scenarios.controlled_repo import (
            CORRECTED_CALCULATOR_CONTENT,
        )

        # Custom source with no repair actions — after replan, source
        # is exhausted, so HarnessRuntime completes but repo is still
        # buggy.
        class EmptyRepairLoopWorkload(LoopWorkload):
            def create_source(self) -> ReplanAwareActionSource:
                return ReplanAwareActionSource(
                    looping_call=make_loop_looping_call(),
                    repair_calls=[],  # empty — no repair
                )

        registry = WorkloadRegistry(
            default_workload=LoopWorkload(),
        )
        registry.register("loop_replan", EmptyRepairLoopWorkload())

        scenario = _make_loop_scenario()
        runner = ReliableHarnessBenchmarkRunner(
            workload_registry=registry,
        )
        rec = _run(runner.run(scenario))

        # Loop was detected, replan happened, but repo is still buggy.
        assert rec.loop_detection_count == 1
        assert rec.replan_count == 1
        assert rec.completed is False
        assert rec.failure_reason == "ORACLE_INCOMPLETE"
