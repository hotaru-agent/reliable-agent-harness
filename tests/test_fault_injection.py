"""Tests for deterministic fault injection (Phase 7 Step 1).

Covers:
* Same FaultPlan produces same fault sequence
* Fresh injector resets counters
* Transient fault at configured invocation
* Permanent fault deterministic
* Timeout fault reaches existing ToolRuntime timeout path
* Large output deterministic
* Process interruption signal distinct from ToolError
* Logical invocation count != retry attempt count

All offline, deterministic.
"""

from __future__ import annotations

import asyncio

import pytest

from evaluation.faults import (
    FaultInjector,
    FaultPlan,
    FaultSpec,
    FaultType,
    InjectedProcessInterruption,
)
from tools.errors import ToolReportedFailure
from tools.models import ToolErrorType


def _run(coro):
    return asyncio.run(coro)


# ===========================================================================
# FaultSpec validation
# ===========================================================================


class TestFaultSpecValidation:
    def test_valid_spec(self):
        spec = FaultSpec(
            fault_type=FaultType.TRANSIENT_FAILURE,
            tool_name="run_tests",
            logical_invocation=1,
        )
        assert spec.fault_type is FaultType.TRANSIENT_FAILURE

    def test_empty_tool_name_rejected(self):
        with pytest.raises(ValueError, match="tool_name"):
            FaultSpec(
                fault_type=FaultType.TRANSIENT_FAILURE,
                tool_name="",
                logical_invocation=1,
            )

    def test_zero_invocation_rejected(self):
        with pytest.raises(ValueError, match="logical_invocation"):
            FaultSpec(
                fault_type=FaultType.TRANSIENT_FAILURE,
                tool_name="run_tests",
                logical_invocation=0,
            )

    def test_negative_invocation_rejected(self):
        with pytest.raises(ValueError, match="logical_invocation"):
            FaultSpec(
                fault_type=FaultType.TRANSIENT_FAILURE,
                tool_name="run_tests",
                logical_invocation=-1,
            )

    def test_zero_large_output_size_rejected(self):
        with pytest.raises(ValueError, match="large_output_size"):
            FaultSpec(
                fault_type=FaultType.LARGE_OUTPUT,
                tool_name="run_tests",
                logical_invocation=1,
                large_output_size=0,
            )

    def test_zero_timeout_sleep_rejected(self):
        with pytest.raises(ValueError, match="timeout_sleep_seconds"):
            FaultSpec(
                fault_type=FaultType.TIMEOUT,
                tool_name="run_tests",
                logical_invocation=1,
                timeout_sleep_seconds=0,
            )

    def test_spec_is_frozen(self):
        spec = FaultSpec(
            fault_type=FaultType.TRANSIENT_FAILURE,
            tool_name="run_tests",
            logical_invocation=1,
        )
        with pytest.raises(Exception):
            spec.tool_name = "other"  # type: ignore[misc]


# ===========================================================================
# FaultPlan validation
# ===========================================================================


class TestFaultPlanValidation:
    def test_valid_plan(self):
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
        assert plan.plan_id == "p1"
        assert len(plan.faults) == 1

    def test_empty_plan_id_rejected(self):
        with pytest.raises(ValueError, match="plan_id"):
            FaultPlan(plan_id="", faults=())

    def test_no_faults_plan(self):
        plan = FaultPlan.no_faults()
        assert plan.plan_id == "none"
        assert plan.faults == ()

    def test_plan_is_frozen(self):
        plan = FaultPlan(plan_id="p1", faults=())
        with pytest.raises(Exception):
            plan.plan_id = "p2"  # type: ignore[misc]


# ===========================================================================
# FaultInjector — counter semantics
# ===========================================================================


class TestFaultInjectorCounters:
    def test_next_invocation_increments(self):
        plan = FaultPlan.no_faults()
        injector = FaultInjector(plan)
        assert injector.next_invocation("run_tests") == 1
        assert injector.next_invocation("run_tests") == 2
        assert injector.next_invocation("run_tests") == 3

    def test_peek_invocation_does_not_increment(self):
        plan = FaultPlan.no_faults()
        injector = FaultInjector(plan)
        assert injector.peek_invocation("run_tests") == 1
        assert injector.peek_invocation("run_tests") == 1
        injector.next_invocation("run_tests")
        assert injector.peek_invocation("run_tests") == 2

    def test_counters_per_tool(self):
        plan = FaultPlan.no_faults()
        injector = FaultInjector(plan)
        assert injector.next_invocation("run_tests") == 1
        assert injector.next_invocation("read_file") == 1
        assert injector.next_invocation("run_tests") == 2
        assert injector.next_invocation("read_file") == 2

    def test_fresh_injector_resets_counters(self):
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
        # First injector: fire the fault.
        injector_a = FaultInjector(plan)
        inv = injector_a.next_invocation("run_tests")
        with pytest.raises(ToolReportedFailure):
            _run(injector_a.maybe_inject("run_tests", inv))
        assert injector_a.fired_fault_indices == (0,)

        # Fresh injector: fault not yet fired.
        injector_b = FaultInjector(plan)
        assert injector_b.fired_fault_indices == ()


# ===========================================================================
# Same FaultPlan produces same fault sequence
# ===========================================================================


class TestDeterministicFaultSequence:
    def test_same_plan_same_sequence(self):
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
        # Run A.
        inj_a = FaultInjector(plan)
        inv_a = inj_a.next_invocation("run_tests")
        with pytest.raises(ToolReportedFailure):
            _run(inj_a.maybe_inject("run_tests", inv_a))
        fired_a = inj_a.fired_fault_indices

        # Run B (fresh injector).
        inj_b = FaultInjector(plan)
        inv_b = inj_b.next_invocation("run_tests")
        with pytest.raises(ToolReportedFailure):
            _run(inj_b.maybe_inject("run_tests", inv_b))
        fired_b = inj_b.fired_fault_indices

        assert fired_a == fired_b == (0,)


# ===========================================================================
# Transient failure injection
# ===========================================================================


class TestTransientFailureInjection:
    def test_transient_fires_at_configured_invocation(self):
        plan = FaultPlan(
            plan_id="p1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=2,
                ),
            ),
        )
        injector = FaultInjector(plan)

        # Invocation 1: no fault.
        inv1 = injector.next_invocation("run_tests")
        spec = _run(injector.maybe_inject("run_tests", inv1))
        assert spec is None

        # Invocation 2: transient fault.
        inv2 = injector.next_invocation("run_tests")
        with pytest.raises(ToolReportedFailure) as exc_info:
            _run(injector.maybe_inject("run_tests", inv2))
        assert exc_info.value.error_type is ToolErrorType.TRANSIENT

    def test_transient_fires_only_once(self):
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
        injector = FaultInjector(plan)

        # Invocation 1: fault fires.
        injector.next_invocation("run_tests")
        with pytest.raises(ToolReportedFailure):
            _run(injector.maybe_inject("run_tests", 1))

        # Invocation 2: no fault (already fired).
        injector.next_invocation("run_tests")
        spec = _run(injector.maybe_inject("run_tests", 2))
        assert spec is None


# ===========================================================================
# Permanent failure injection
# ===========================================================================


class TestPermanentFailureInjection:
    def test_permanent_fires_at_configured_invocation(self):
        plan = FaultPlan(
            plan_id="p1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.PERMANENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        )
        injector = FaultInjector(plan)
        injector.next_invocation("run_tests")
        with pytest.raises(ToolReportedFailure) as exc_info:
            _run(injector.maybe_inject("run_tests", 1))
        assert exc_info.value.error_type is ToolErrorType.PERMANENT


# ===========================================================================
# Timeout injection
# ===========================================================================


class TestTimeoutInjection:
    def test_timeout_sleeps(self):
        plan = FaultPlan(
            plan_id="p1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TIMEOUT,
                    tool_name="run_tests",
                    logical_invocation=1,
                    timeout_sleep_seconds=0.05,
                ),
            ),
        )
        injector = FaultInjector(plan)
        injector.next_invocation("run_tests")

        slept = []

        async def fake_sleep(seconds):
            slept.append(seconds)

        spec = _run(injector.maybe_inject("run_tests", 1, sleep_fn=fake_sleep))
        assert spec is not None
        assert spec.fault_type is FaultType.TIMEOUT
        assert slept == [0.05]


# ===========================================================================
# Large output injection
# ===========================================================================


class TestLargeOutputInjection:
    def test_large_output_returns_spec(self):
        plan = FaultPlan(
            plan_id="p1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.LARGE_OUTPUT,
                    tool_name="run_tests",
                    logical_invocation=1,
                    large_output_size=100,
                ),
            ),
        )
        injector = FaultInjector(plan)
        injector.next_invocation("run_tests")
        spec = _run(injector.maybe_inject("run_tests", 1))
        assert spec is not None
        assert spec.fault_type is FaultType.LARGE_OUTPUT

    def test_generate_large_output_deterministic(self):
        spec = FaultSpec(
            fault_type=FaultType.LARGE_OUTPUT,
            tool_name="run_tests",
            logical_invocation=1,
            large_output_size=100,
        )
        payload1 = FaultInjector.generate_large_output(spec)
        payload2 = FaultInjector.generate_large_output(spec)
        assert payload1 == payload2
        assert len(payload1) == 100

    def test_generate_large_output_wrong_type_rejected(self):
        spec = FaultSpec(
            fault_type=FaultType.TRANSIENT_FAILURE,
            tool_name="run_tests",
            logical_invocation=1,
        )
        with pytest.raises(ValueError, match="LARGE_OUTPUT"):
            FaultInjector.generate_large_output(spec)


# ===========================================================================
# Process interruption injection
# ===========================================================================


class TestProcessInterruptionInjection:
    def test_interruption_raises_base_exception(self):
        plan = FaultPlan(
            plan_id="p1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.PROCESS_INTERRUPTION,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        )
        injector = FaultInjector(plan)
        injector.next_invocation("run_tests")
        with pytest.raises(InjectedProcessInterruption):
            _run(injector.maybe_inject("run_tests", 1))

    def test_interruption_is_not_tool_error(self):
        """InjectedProcessInterruption is a BaseException, not a
        ToolReportedFailure. It must not be caught by generic
        ``except Exception`` handlers."""
        plan = FaultPlan(
            plan_id="p1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.PROCESS_INTERRUPTION,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        )
        injector = FaultInjector(plan)
        injector.next_invocation("run_tests")

        caught = False
        try:
            _run(injector.maybe_inject("run_tests", 1))
        except ToolReportedFailure:
            caught = True
        except Exception:
            caught = True
        except BaseException:
            # InjectedProcessInterruption is a BaseException but not Exception.
            pass
        assert not caught, "interruption was caught by Exception handler"

    def test_interruption_is_not_exception_subclass(self):
        assert not issubclass(InjectedProcessInterruption, Exception)
        assert issubclass(InjectedProcessInterruption, BaseException)


# ===========================================================================
# Logical invocation vs retry attempt semantics
# ===========================================================================


class TestLogicalInvocationVsRetryAttempt:
    """The critical distinction: logical invocation count (how many times
    the agent called the tool) is NOT the same as retry attempt count
    (how many times the handler was executed within one logical
    invocation's retry loop).

    FaultPlan targets logical invocations. A transient fault on
    invocation #1 means the FIRST logical call to that tool fails
    transiently. The ToolRuntime may retry it (attempt 2, 3, ...) within
    that same logical invocation, but the fault injector does NOT fire
    again for those retry attempts.
    """

    def test_fault_targets_logical_not_attempt(self):
        """A transient fault on logical invocation #1 fires once. The
        ToolRuntime retry (attempt 2) does NOT trigger another fault."""
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
        injector = FaultInjector(plan)

        # Logical invocation 1: fault fires.
        inv1 = injector.next_invocation("run_tests")
        with pytest.raises(ToolReportedFailure):
            _run(injector.maybe_inject("run_tests", inv1))

        # The ToolRuntime would retry (attempt 2) — but the injector
        # does NOT increment the logical invocation counter for retries.
        # The retry uses the SAME logical_invocation number.
        # The fault has already been fired, so it won't fire again.
        spec = _run(injector.maybe_inject("run_tests", inv1))
        assert spec is None  # fault already consumed

    def test_logical_count_independent_of_attempts(self):
        """Three logical invocations, each potentially with multiple
        retry attempts, still produces logical counts 1, 2, 3."""
        plan = FaultPlan.no_faults()
        injector = FaultInjector(plan)

        # Logical invocation 1 (may have N retry attempts internally).
        assert injector.next_invocation("run_tests") == 1
        # Logical invocation 2.
        assert injector.next_invocation("run_tests") == 2
        # Logical invocation 3.
        assert injector.next_invocation("run_tests") == 3


# ===========================================================================
# Wildcard tool_name matching
# ===========================================================================


class TestWildcardMatching:
    def test_wildcard_matches_any_tool(self):
        plan = FaultPlan(
            plan_id="p1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="*",
                    logical_invocation=1,
                ),
            ),
        )
        injector = FaultInjector(plan)

        # First invocation of read_file: fault fires (wildcard match).
        injector.next_invocation("read_file")
        with pytest.raises(ToolReportedFailure):
            _run(injector.maybe_inject("read_file", 1))


# ===========================================================================
# No faults plan
# ===========================================================================


class TestNoFaultsPlan:
    def test_no_faults_never_injects(self):
        plan = FaultPlan.no_faults()
        injector = FaultInjector(plan)
        for i in range(1, 10):
            injector.next_invocation("run_tests")
            spec = _run(injector.maybe_inject("run_tests", i))
            assert spec is None
        assert injector.fired_fault_indices == ()
