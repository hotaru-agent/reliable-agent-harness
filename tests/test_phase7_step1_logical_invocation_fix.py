"""Phase 7 Step 1 Fix — Logical Invocation Accounting regression tests.

These tests prove the core invariant fixed in this phase:

    1 logical invocation
        |
        v
    ToolRuntime.execute()
        |
        v
    attempt 1
    attempt 2
    attempt 3
        |
        v
    all attempts share SAME logical invocation identity

Regardless of retry count, the logical invocation count remains one.

The key acceptance test (Section 25):

    If the first ``run_tests`` logical ToolCall experiences transient
    failure + retry success, and FaultPlan configures ``run_tests
    logical invocation #2 -> TIMEOUT``, then TIMEOUT must occur on the
    second actual ``run_tests`` logical ToolCall — not on the first
    call's retry attempt, and not skipped because the retry advanced
    the counter.

Answer (test-supported): YES
"""

from __future__ import annotations

import asyncio

import pytest

from evaluation.faults import FaultInjector, FaultPlan, FaultSpec, FaultType
from evaluation.invoker import FaultAwareToolInvoker
from evaluation.repository import ControlledRepository, RepositoryOracle
from evaluation.scenarios.controlled_repo import (
    CORRECTED_CALCULATOR_CONTENT,
    ScriptedAction,
    ScriptedActionSource,
    make_calculator_fixture,
    make_fix_calculator_actions,
    make_repository_tool_handlers,
    make_repository_tool_specs,
)
from tools import (
    ToolCall,
    ToolExecutionContext,
    ToolRegistry,
    ToolRuntime,
)
from tools.models import ToolErrorType


def _run(coro):
    return asyncio.run(coro)


class _FakeSleeper:
    """Deterministic sleep that records durations without actually sleeping."""

    def __init__(self):
        self.sleeps: list[float] = []

    async def __call__(self, seconds: float):
        self.sleeps.append(seconds)


def _make_runtime_and_invoker(
    repo: ControlledRepository,
    injector: FaultInjector,
    *,
    run_tests_timeout_seconds: float = 5.0,
):
    """Build a registry + runtime + invoker for testing."""
    registry = ToolRegistry()
    specs = make_repository_tool_specs(
        run_tests_timeout_seconds=run_tests_timeout_seconds,
    )
    handlers = make_repository_tool_handlers(repo, injector=injector)
    for name in specs:
        registry.register(specs[name], handlers[name])
    runtime = ToolRuntime(registry=registry, sleep=_FakeSleeper())
    invoker = FaultAwareToolInvoker(runtime, injector=injector)
    return runtime, invoker


# ===========================================================================
# Section 4 — Logical invocation identity allocated once before execute
# ===========================================================================


class TestLogicalInvocationAllocatedOnce:
    """The logical invocation identity must be allocated exactly once
    before ``ToolRuntime.execute()``, not once per handler attempt."""

    def test_begin_invocation_allocates_index_once(self):
        """begin_invocation allocates the index and it stays active."""
        injector = FaultInjector(FaultPlan.no_faults())
        idx = injector.begin_invocation("run_tests")
        assert idx == 1
        # current_invocation returns the active identity.
        assert injector.current_invocation("run_tests") == 1
        # Without end_invocation, the counter has advanced but the
        # active identity is still 1.
        assert injector.peek_invocation("run_tests") == 2

    def test_end_invocation_clears_active_identity(self):
        """end_invocation clears the active identity."""
        injector = FaultInjector(FaultPlan.no_faults())
        injector.begin_invocation("run_tests")
        assert injector.current_invocation("run_tests") == 1
        injector.end_invocation("run_tests")
        # After end, current_invocation returns 0 (no active invocation).
        assert injector.current_invocation("run_tests") == 0
        # The counter still records that one invocation happened.
        assert injector.peek_invocation("run_tests") == 2

    def test_next_begin_invocation_gets_next_index(self):
        """After ending one invocation, the next begin gets index 2."""
        injector = FaultInjector(FaultPlan.no_faults())
        idx1 = injector.begin_invocation("run_tests")
        injector.end_invocation("run_tests")
        idx2 = injector.begin_invocation("run_tests")
        injector.end_invocation("run_tests")
        assert idx1 == 1
        assert idx2 == 2

    def test_current_invocation_zero_when_no_active(self):
        """current_invocation returns 0 when no invocation is active."""
        injector = FaultInjector(FaultPlan.no_faults())
        assert injector.current_invocation("run_tests") == 0


# ===========================================================================
# Section 10 — Retry attempts share the same logical identity
# ===========================================================================


class TestRetryDoesNotAdvanceLogicalInvocation:
    """A ToolRuntime retry attempt must NOT advance the logical
    invocation counter. All attempts within one logical call share the
    same logical identity."""

    def test_transient_retry_does_not_advance_counter(self):
        """Transient fault on logical #1 + retry success: the next
        logical call should still get index 2, not 3."""
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()

        plan = FaultPlan(
            plan_id="transient_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        )
        injector = FaultInjector(plan)
        _, invoker = _make_runtime_and_invoker(repo, injector)

        ctx = ToolExecutionContext()
        result = _run(invoker.execute(
            ToolCall(tool_name="run_tests", arguments={}), ctx,
        ))

        # Retry succeeded.
        assert result.success is True
        assert result.attempt_count == 2
        # The logical invocation counter must be 1, not 2.
        # After end_invocation, current is 0 and peek is 2.
        assert injector.current_invocation("run_tests") == 0
        assert injector.peek_invocation("run_tests") == 2

    def test_three_attempts_still_one_logical_invocation(self):
        """Two transient faults at logical #1 cause 3 attempts (2
        retries), but the logical invocation count is still 1."""
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()

        plan = FaultPlan(
            plan_id="double_transient_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        )
        injector = FaultInjector(plan)
        _, invoker = _make_runtime_and_invoker(repo, injector)

        ctx = ToolExecutionContext()
        result = _run(invoker.execute(
            ToolCall(tool_name="run_tests", arguments={}), ctx,
        ))

        # Three attempts: 2 transient failures + 1 success.
        assert result.success is True
        assert result.attempt_count == 3
        assert len(result.retry_history) == 2
        # Both retries were transient.
        for err in result.retry_history:
            assert err.error_type is ToolErrorType.TRANSIENT
        # Logical invocation count is still 1.
        assert injector.current_invocation("run_tests") == 0
        assert injector.peek_invocation("run_tests") == 2

    def test_second_logical_call_gets_index_2(self):
        """After a first logical call (with retries), the second logical
        call receives index 2."""
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()

        # Transient on logical #1 so the first call retries.
        plan = FaultPlan(
            plan_id="transient_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        )
        injector = FaultInjector(plan)
        _, invoker = _make_runtime_and_invoker(repo, injector)

        ctx = ToolExecutionContext()
        # First logical call.
        r1 = _run(invoker.execute(
            ToolCall(tool_name="run_tests", arguments={}), ctx,
        ))
        assert r1.attempt_count == 2  # retried once

        # Second logical call — should get index 2.
        # No fault configured for index 2, so it succeeds on attempt 1.
        r2 = _run(invoker.execute(
            ToolCall(tool_name="run_tests", arguments={}), ctx,
        ))
        assert r2.success is True
        assert r2.attempt_count == 1  # no retry needed
        # After two logical calls, the next would be index 3.
        assert injector.peek_invocation("run_tests") == 3


# ===========================================================================
# Section 25 — Key acceptance test
# ===========================================================================


class TestKeyAcceptanceTest:
    """The key acceptance test from Section 25.

    If the first ``run_tests`` logical ToolCall experiences transient
    failure + retry success, and FaultPlan configures ``run_tests
    logical invocation #2 -> TIMEOUT``, then TIMEOUT must occur on the
    second actual ``run_tests`` logical ToolCall — not on the first
    call's retry attempt, and not skipped because the retry advanced
    the counter.
    """

    def test_timeout_fires_on_second_logical_call_not_first_retry(self):
        """The definitive acceptance test."""
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()

        plan = FaultPlan(
            plan_id="transient_1_timeout_2",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
                FaultSpec(
                    fault_type=FaultType.TIMEOUT,
                    tool_name="run_tests",
                    logical_invocation=2,
                    timeout_sleep_seconds=10.0,
                ),
            ),
        )
        injector = FaultInjector(plan)
        _, invoker = _make_runtime_and_invoker(
            repo, injector, run_tests_timeout_seconds=0.01,
        )

        ctx = ToolExecutionContext()

        # --- First logical call ---
        # Attempt 1: TRANSIENT fires (logical #1).
        # Attempt 2: no fault at logical #1 (transient consumed), succeeds.
        r1 = _run(invoker.execute(
            ToolCall(tool_name="run_tests", arguments={}), ctx,
        ))
        assert r1.success is True
        assert r1.attempt_count == 2
        assert len(r1.retry_history) == 1
        assert r1.retry_history[0].error_type is ToolErrorType.TRANSIENT
        # The TIMEOUT must NOT have fired during the first call.
        # (If it had, r1.retry_history would contain TIMEOUT, not TRANSIENT.)

        # --- Second logical call ---
        # Attempt 1: TIMEOUT fires (logical #2).
        # Attempt 2: no fault at logical #2 (timeout consumed), succeeds.
        r2 = _run(invoker.execute(
            ToolCall(tool_name="run_tests", arguments={}), ctx,
        ))
        assert r2.success is True
        assert r2.attempt_count == 2
        assert len(r2.retry_history) == 1
        # The TIMEOUT must have fired on the second logical call.
        assert r2.retry_history[0].error_type is ToolErrorType.TIMEOUT

        # --- Verification ---
        # Both faults were fired.
        assert injector.fired_fault_indices == (0, 1)
        # The logical invocation counter advanced exactly twice.
        assert injector.peek_invocation("run_tests") == 3

    def test_timeout_not_consumed_by_first_call_retry_attempt_2(self):
        """Explicitly verify that the TIMEOUT fault (logical #2) is NOT
        consumed by retry attempt #2 of logical call #1."""
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()

        plan = FaultPlan(
            plan_id="transient_1_timeout_2",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
                FaultSpec(
                    fault_type=FaultType.TIMEOUT,
                    tool_name="run_tests",
                    logical_invocation=2,
                    timeout_sleep_seconds=10.0,
                ),
            ),
        )
        injector = FaultInjector(plan)
        _, invoker = _make_runtime_and_invoker(
            repo, injector, run_tests_timeout_seconds=0.01,
        )

        ctx = ToolExecutionContext()

        # First logical call: transient fires, retry succeeds.
        r1 = _run(invoker.execute(
            ToolCall(tool_name="run_tests", arguments={}), ctx,
        ))
        # Only the transient fault (index 0) should be fired.
        # The timeout fault (index 1) must NOT be fired.
        assert injector.fired_fault_indices == (0,)
        assert r1.retry_history[0].error_type is ToolErrorType.TRANSIENT

        # Second logical call: timeout fires.
        r2 = _run(invoker.execute(
            ToolCall(tool_name="run_tests", arguments={}), ctx,
        ))
        # Now both faults are fired.
        assert injector.fired_fault_indices == (0, 1)
        assert r2.retry_history[0].error_type is ToolErrorType.TIMEOUT

    def test_timeout_not_skipped_after_retries(self):
        """Verify that the TIMEOUT fault is not skipped because the
        first call's retries advanced the counter past #2."""
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()

        # Two transient faults at logical #1 → 3 attempts (counter
        # would advance to #3 without the fix, skipping #2).
        plan = FaultPlan(
            plan_id="double_transient_1_timeout_2",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
                FaultSpec(
                    fault_type=FaultType.TIMEOUT,
                    tool_name="run_tests",
                    logical_invocation=2,
                    timeout_sleep_seconds=10.0,
                ),
            ),
        )
        injector = FaultInjector(plan)
        _, invoker = _make_runtime_and_invoker(
            repo, injector, run_tests_timeout_seconds=0.01,
        )

        ctx = ToolExecutionContext()

        # First logical call: 2 transient faults → 3 attempts.
        r1 = _run(invoker.execute(
            ToolCall(tool_name="run_tests", arguments={}), ctx,
        ))
        assert r1.attempt_count == 3
        # Only the two transient faults should be fired, NOT the timeout.
        assert injector.fired_fault_indices == (0, 1)

        # Second logical call: timeout fires on logical #2.
        r2 = _run(invoker.execute(
            ToolCall(tool_name="run_tests", arguments={}), ctx,
        ))
        assert r2.attempt_count == 2
        assert r2.retry_history[0].error_type is ToolErrorType.TIMEOUT
        assert injector.fired_fault_indices == (0, 1, 2)


# ===========================================================================
# Section 13 — Fault consumption vs invocation accounting
# ===========================================================================


class TestFaultConsumptionVsInvocationAccounting:
    """Fault consumption (fired=True) and logical invocation identity
    are separate concepts. Both invariants must hold simultaneously."""

    def test_fault_consumed_but_invocation_still_one(self):
        """A transient fault fires (consumed) on attempt 1, but the
        logical invocation is still #1 for attempt 2."""
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()

        plan = FaultPlan(
            plan_id="transient_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        )
        injector = FaultInjector(plan)
        _, invoker = _make_runtime_and_invoker(repo, injector)

        ctx = ToolExecutionContext()
        result = _run(invoker.execute(
            ToolCall(tool_name="run_tests", arguments={}), ctx,
        ))

        # The fault was consumed (fired).
        assert injector.fired_fault_indices == (0,)
        # But the logical invocation count is still 1.
        assert injector.peek_invocation("run_tests") == 2
        # And the result shows 2 attempts (1 retry) for 1 logical call.
        assert result.attempt_count == 2

    def test_no_fault_does_not_consume_or_advance(self):
        """When no fault is configured, nothing is consumed and the
        counter still advances by exactly 1 per logical call."""
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()

        injector = FaultInjector(FaultPlan.no_faults())
        _, invoker = _make_runtime_and_invoker(repo, injector)

        ctx = ToolExecutionContext()
        _run(invoker.execute(
            ToolCall(tool_name="run_tests", arguments={}), ctx,
        ))
        _run(invoker.execute(
            ToolCall(tool_name="run_tests", arguments={}), ctx,
        ))

        assert injector.fired_fault_indices == ()
        assert injector.peek_invocation("run_tests") == 3


# ===========================================================================
# Section 8 — Fresh injector resets logical counters
# ===========================================================================


class TestFreshInjectorResetsCounters:
    """A fresh FaultInjector resets all logical invocation counters."""

    def test_fresh_injector_starts_at_index_1(self):
        """A fresh injector allocates index 1 for the first call."""
        plan = FaultPlan(
            plan_id="p",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        )

        # First injector: use it.
        inj1 = FaultInjector(plan)
        assert inj1.begin_invocation("run_tests") == 1
        inj1.end_invocation("run_tests")
        assert inj1.begin_invocation("run_tests") == 2
        inj1.end_invocation("run_tests")

        # Fresh injector: starts at 1 again.
        inj2 = FaultInjector(plan)
        assert inj2.begin_invocation("run_tests") == 1
        inj2.end_invocation("run_tests")

    def test_fresh_injector_refires_same_fault(self):
        """A fresh injector can fire the same fault again."""
        plan = FaultPlan(
            plan_id="p",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        )

        # First injector: fault fires.
        inj1 = FaultInjector(plan)
        inj1.begin_invocation("run_tests")
        with pytest.raises(Exception):
            _run(inj1.maybe_inject("run_tests", 1))
        inj1.end_invocation("run_tests")
        assert inj1.fired_fault_indices == (0,)

        # Fresh injector: fault fires again.
        inj2 = FaultInjector(plan)
        inj2.begin_invocation("run_tests")
        with pytest.raises(Exception):
            _run(inj2.maybe_inject("run_tests", 1))
        inj2.end_invocation("run_tests")
        assert inj2.fired_fault_indices == (0,)


# ===========================================================================
# Section 20 — Same FaultPlan deterministic with internal retries
# ===========================================================================


class TestDeterminismWithRetries:
    """The same scenario, repository snapshot, ScriptedActionSource, and
    FaultPlan must yield the same fault placement and functional result
    regardless of internal retry count."""

    def test_same_plan_same_result_across_runs(self):
        """Running the same fault plan twice (with fresh injector + repo
        each time) produces the same results."""
        plan = FaultPlan(
            plan_id="transient_1_timeout_2",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
                FaultSpec(
                    fault_type=FaultType.TIMEOUT,
                    tool_name="run_tests",
                    logical_invocation=2,
                    timeout_sleep_seconds=10.0,
                ),
            ),
        )

        results = []
        for _ in range(2):
            fixture = make_calculator_fixture()
            repo = fixture.create_repository()
            injector = FaultInjector(plan)
            _, invoker = _make_runtime_and_invoker(
                repo, injector, run_tests_timeout_seconds=0.01,
            )
            ctx = ToolExecutionContext()

            r1 = _run(invoker.execute(
                ToolCall(tool_name="run_tests", arguments={}), ctx,
            ))
            r2 = _run(invoker.execute(
                ToolCall(tool_name="run_tests", arguments={}), ctx,
            ))
            results.append((r1, r2, injector.fired_fault_indices))

        # Both runs produce the same results.
        for i in range(2):
            r1, r2, fired = results[i]
            assert r1.attempt_count == 2
            assert r1.retry_history[0].error_type is ToolErrorType.TRANSIENT
            assert r2.attempt_count == 2
            assert r2.retry_history[0].error_type is ToolErrorType.TIMEOUT
            assert fired == (0, 1)

        # Both runs have identical fired fault indices.
        assert results[0][2] == results[1][2]


# ===========================================================================
# Section 16 — Logical action count vs attempt count independence
# ===========================================================================


class TestLogicalActionVsAttemptCount:
    """Logical action count and tool attempt count are independent
    concepts. One logical action with three attempts counts as one
    logical action, not three."""

    def test_one_logical_action_three_attempts(self):
        """Two transient faults → 3 attempts, but logical action count
        is 1 and tool invocation count is 1."""
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()

        plan = FaultPlan(
            plan_id="double_transient_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        )
        injector = FaultInjector(plan)
        _, invoker = _make_runtime_and_invoker(repo, injector)

        ctx = ToolExecutionContext()
        result = _run(invoker.execute(
            ToolCall(tool_name="run_tests", arguments={}), ctx,
        ))

        # 3 attempts, 2 retries.
        assert result.attempt_count == 3
        # But only 1 logical invocation was consumed.
        assert injector.peek_invocation("run_tests") == 2
        # The logical action count (from the caller's perspective) is 1.
        # The tool invocation count is 1 (one execute() call).
        # The tool attempt count is 3.
        # The retry count is 2.
        # These are all independent.

    def test_duplicate_action_separate_from_retry(self):
        """A duplicate logical action (calling run_tests twice) is
        separate from retry attempts within one call."""
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()

        # Transient on logical #1 only.
        plan = FaultPlan(
            plan_id="transient_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        )
        injector = FaultInjector(plan)
        _, invoker = _make_runtime_and_invoker(repo, injector)

        ctx = ToolExecutionContext()

        # First logical call: transient + retry success (2 attempts).
        r1 = _run(invoker.execute(
            ToolCall(tool_name="run_tests", arguments={}), ctx,
        ))
        assert r1.attempt_count == 2

        # Second logical call: no fault, 1 attempt.
        r2 = _run(invoker.execute(
            ToolCall(tool_name="run_tests", arguments={}), ctx,
        ))
        assert r2.attempt_count == 1

        # Two logical actions, two tool invocations, three total attempts.
        # The retry in the first call is NOT a duplicate action.
        assert injector.peek_invocation("run_tests") == 3  # 2 logical calls


# ===========================================================================
# Section 17 — Duplicate semantics: retry != duplicate action
# ===========================================================================


class TestRetryNotDuplicateAction:
    """Retry attempts are NOT duplicate logical actions. A call with
    transient + retry success has duplicate_action_count = 0."""

    def test_retry_does_not_count_as_duplicate(self):
        """The smoke scenario with a transient fault on run_tests #1
        still completes with 4 logical actions (not 5)."""
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        oracle = RepositoryOracle()

        plan = FaultPlan(
            plan_id="transient_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        )
        injector = FaultInjector(plan)
        runtime, invoker = _make_runtime_and_invoker(repo, injector)

        ctx = ToolExecutionContext()
        source = ScriptedActionSource(make_fix_calculator_actions())

        logical_action_count = 0
        tool_invocation_count = 0
        tool_attempt_count = 0
        retry_count = 0

        while source.has_next():
            call = source.next_call()
            logical_action_count += 1
            tool_invocation_count += 1
            result = _run(invoker.execute(call, ctx))
            tool_attempt_count += result.attempt_count
            retry_count += max(0, result.attempt_count - 1)

        # 4 logical actions, 4 tool invocations.
        assert logical_action_count == 4
        assert tool_invocation_count == 4
        # The first run_tests had 2 attempts (1 retry), the rest had 1.
        # Total attempts: 2 + 1 + 1 + 1 = 5.
        assert tool_attempt_count == 5
        assert retry_count == 1
        # The task still completes.
        assert oracle.is_complete(repo) is True
        # No duplicate actions — the retry is not a duplicate.
        # (duplicate_action_count would be 0, tracked by AgentActionController
        # in the Harness layer, not here. Here we verify the counts are
        # consistent: 4 logical actions, not 5.)


# ===========================================================================
# Section 9 — FaultAwareToolInvoker boundary properties
# ===========================================================================


class TestFaultAwareToolInvokerBoundary:
    """The invoker boundary allocates logical identity before execute
    and clears it after, even on exceptions."""

    def test_invoker_passes_through_without_injector(self):
        """Without an injector, the invoker is a pure pass-through."""
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()

        registry = ToolRegistry()
        specs = make_repository_tool_specs()
        handlers = make_repository_tool_handlers(repo)
        for name in specs:
            registry.register(specs[name], handlers[name])
        runtime = ToolRuntime(registry=registry, sleep=_FakeSleeper())
        invoker = FaultAwareToolInvoker(runtime, injector=None)

        ctx = ToolExecutionContext()
        result = _run(invoker.execute(
            ToolCall(tool_name="run_tests", arguments={}), ctx,
        ))
        assert result.success is True
        assert result.attempt_count == 1

    def test_invoker_clears_active_on_exception(self):
        """The invoker clears the active invocation even when an
        exception propagates (e.g. InjectedProcessInterruption)."""
        from evaluation.faults import InjectedProcessInterruption

        fixture = make_calculator_fixture()
        repo = fixture.create_repository()

        plan = FaultPlan(
            plan_id="interruption_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.PROCESS_INTERRUPTION,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        )
        injector = FaultInjector(plan)
        _, invoker = _make_runtime_and_invoker(repo, injector)

        ctx = ToolExecutionContext()
        with pytest.raises(InjectedProcessInterruption):
            _run(invoker.execute(
                ToolCall(tool_name="run_tests", arguments={}), ctx,
            ))

        # The active invocation must have been cleared by the finally block.
        assert injector.current_invocation("run_tests") == 0
        # But the counter still advanced.
        assert injector.peek_invocation("run_tests") == 2

    def test_invoker_rejects_non_runtime(self):
        """The invoker rejects a non-ToolRuntime runtime argument."""
        with pytest.raises(ValueError, match="runtime"):
            FaultAwareToolInvoker(runtime="not a runtime")  # type: ignore

    def test_invoker_rejects_non_injector(self):
        """The invoker rejects a non-FaultInjector injector argument."""
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        registry = ToolRegistry()
        runtime = ToolRuntime(registry=registry, sleep=_FakeSleeper())
        with pytest.raises(ValueError, match="injector"):
            FaultAwareToolInvoker(runtime=runtime, injector="not an injector")  # type: ignore


# ===========================================================================
# Section 16 — BenchmarkRunRecord terminology consistency
# ===========================================================================


class TestBenchmarkRecordTerminologyConsistency:
    """BenchmarkRunRecord terminology (logical_action_count,
    tool_invocation_count, tool_attempt_count, retry_count) must be
    consistent with FaultInjector terminology (logical invocation)."""

    def test_record_counts_match_invoker_semantics(self):
        """A run with one transient fault produces:
        - logical_action_count = 4 (4 scripted actions)
        - tool_invocation_count = 4 (4 execute() calls)
        - tool_attempt_count = 5 (first run_tests retried once)
        - retry_count = 1
        """
        from evaluation.records import (
            BenchmarkEvent,
            BenchmarkEventType,
            BenchmarkResult,
            BenchmarkRunRecord,
        )

        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        oracle = RepositoryOracle()

        plan = FaultPlan(
            plan_id="transient_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        )
        injector = FaultInjector(plan)
        _, invoker = _make_runtime_and_invoker(repo, injector)

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
            result = _run(invoker.execute(call, ctx))
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
            scenario_id="transient_failure",
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
        assert record.tool_attempt_count == 5  # 2 + 1 + 1 + 1
        assert record.retry_count == 1
        # The FaultInjector also shows exactly 4 logical invocations.
        assert injector.peek_invocation("run_tests") == 3  # 2 run_tests calls
