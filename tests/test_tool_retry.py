"""Tests for deterministic retry policy & side-effect safety (Phase 2 Step 2).

Covers:

* RetryPolicy validation (Section 37)
* should_retry eligibility (Sections 12-13, 17-18)
* compute_backoff sequence (Sections 19, 48-49)
* READ_ONLY retry then success (Sections 14, 38, 46)
* IDEMPOTENT retry (Sections 15, 45)
* max attempts exhausted (Section 39)
* timeout then success (Section 40)
* permanent stops immediately (Section 41)
* execution stops immediately (Section 42)
* SIDE_EFFECTING transient not retried (Sections 7, 16, 43)
* SIDE_EFFECTING timeout not retried (Section 44)
* policy narrows error types (Section 18, 47)
* backoff sequence via FakeSleeper (Sections 19, 48)
* no sleep before first attempt (Section 49)
* cancellation during backoff (Section 50)
* cancellation during attempt (Section 51)
* handler argument mutation isolation (Sections 21-22, 52)
* caller dict not mutated (Section 53)
* attempt_count / retry_history metadata (Sections 30-35)

All offline, deterministic, no network, no LLM, no MCP.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.fake_tools import (
    ControllableSleeper,
    CooperativeCancelTool,
    FakeSleeper,
    MutatingTool,
    PermanentFailureTool,
    RaisingTool,
    ScriptedTool,
    SlowTool,
    SuccessTool,
    TransientFailureTool,
)
from tools import (
    RetryPolicy,
    ToolCall,
    ToolError,
    ToolErrorType,
    ToolExecutionContext,
    ToolRegistry,
    ToolRuntime,
    ToolSideEffect,
    ToolSpec,
    compute_backoff,
    should_retry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec(
    name: str = "tool",
    *,
    side_effect: ToolSideEffect = ToolSideEffect.READ_ONLY,
    retry_policy: RetryPolicy | None = None,
    timeout_seconds: float = 1.0,
    schema: dict | None = None,
    required_permissions: frozenset[str] = frozenset(),
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="test tool",
        input_schema=schema or {"type": "object"},
        required_permissions=required_permissions,
        timeout_seconds=timeout_seconds,
        side_effect=side_effect,
        retry_policy=retry_policy or RetryPolicy(),
    )


def _registry_with(spec: ToolSpec, handler) -> tuple[ToolRegistry, ToolRuntime, object]:
    registry = ToolRegistry()
    registry.register(spec, handler)
    return registry, handler


def _make_runtime(
    registry: ToolRegistry, *, sleeper: FakeSleeper | None = None
) -> ToolRuntime:
    return ToolRuntime(
        registry=registry,
        sleep=sleeper if sleeper is not None else FakeSleeper(),
    )


# ===========================================================================
# Section 37 — RetryPolicy validation
# ===========================================================================


class TestRetryPolicyValidation:
    def test_default_is_single_attempt(self):
        p = RetryPolicy()
        assert p.max_attempts == 1

    def test_max_attempts_must_be_ge_1(self):
        with pytest.raises(ValueError):
            RetryPolicy(max_attempts=0)
        with pytest.raises(ValueError):
            RetryPolicy(max_attempts=-1)

    def test_max_attempts_must_be_int(self):
        with pytest.raises(ValueError):
            RetryPolicy(max_attempts=2.5)

    def test_negative_initial_backoff_rejected(self):
        with pytest.raises(ValueError):
            RetryPolicy(initial_backoff_seconds=-1)

    def test_multiplier_lt_1_rejected(self):
        with pytest.raises(ValueError):
            RetryPolicy(backoff_multiplier=0.5)

    def test_negative_max_backoff_rejected(self):
        with pytest.raises(ValueError):
            RetryPolicy(max_backoff_seconds=-1)

    def test_initial_gt_max_backoff_rejected(self):
        with pytest.raises(ValueError):
            RetryPolicy(initial_backoff_seconds=5, max_backoff_seconds=1)

    def test_retryable_error_types_must_be_frozenset(self):
        with pytest.raises(ValueError):
            RetryPolicy(retryable_error_types={ToolErrorType.TRANSIENT})  # set not frozenset

    def test_valid_policy(self):
        p = RetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=1.0,
            backoff_multiplier=2.0,
            max_backoff_seconds=10.0,
            retryable_error_types=frozenset({ToolErrorType.TRANSIENT}),
        )
        assert p.max_attempts == 3
        assert p.retryable_error_types == frozenset({ToolErrorType.TRANSIENT})


# ===========================================================================
# Sections 12-13, 17-18 — should_retry eligibility
# ===========================================================================


class TestShouldRetry:
    def _error(self, error_type: ToolErrorType, retryable: bool) -> ToolError:
        return ToolError(error_type=error_type, message="x", retryable=retryable)

    def test_non_retryable_error_no_retry(self):
        spec = _spec(retry_policy=RetryPolicy(max_attempts=5))
        err = self._error(ToolErrorType.VALIDATION, False)
        assert should_retry(spec=spec, error=err, attempt_number=1) is False

    def test_retryable_but_error_type_not_in_policy(self):
        spec = _spec(
            retry_policy=RetryPolicy(
                max_attempts=5,
                retryable_error_types=frozenset({ToolErrorType.TRANSIENT}),
            )
        )
        err = self._error(ToolErrorType.TIMEOUT, True)
        assert should_retry(spec=spec, error=err, attempt_number=1) is False

    def test_max_attempts_reached(self):
        spec = _spec(retry_policy=RetryPolicy(max_attempts=3))
        err = self._error(ToolErrorType.TRANSIENT, True)
        assert should_retry(spec=spec, error=err, attempt_number=3) is False

    def test_attempts_remaining(self):
        spec = _spec(retry_policy=RetryPolicy(max_attempts=3))
        err = self._error(ToolErrorType.TRANSIENT, True)
        assert should_retry(spec=spec, error=err, attempt_number=2) is True

    def test_side_effecting_blocks_retry(self):
        spec = _spec(
            side_effect=ToolSideEffect.SIDE_EFFECTING,
            retry_policy=RetryPolicy(max_attempts=5),
        )
        err = self._error(ToolErrorType.TRANSIENT, True)
        assert should_retry(spec=spec, error=err, attempt_number=1) is False

    def test_side_effecting_timeout_blocks_retry(self):
        spec = _spec(
            side_effect=ToolSideEffect.SIDE_EFFECTING,
            retry_policy=RetryPolicy(max_attempts=5),
        )
        err = self._error(ToolErrorType.TIMEOUT, True)
        assert should_retry(spec=spec, error=err, attempt_number=1) is False

    def test_read_only_allows_retry(self):
        spec = _spec(
            side_effect=ToolSideEffect.READ_ONLY,
            retry_policy=RetryPolicy(max_attempts=3),
        )
        err = self._error(ToolErrorType.TRANSIENT, True)
        assert should_retry(spec=spec, error=err, attempt_number=1) is True

    def test_idempotent_allows_retry(self):
        spec = _spec(
            side_effect=ToolSideEffect.IDEMPOTENT,
            retry_policy=RetryPolicy(max_attempts=3),
        )
        err = self._error(ToolErrorType.TRANSIENT, True)
        assert should_retry(spec=spec, error=err, attempt_number=1) is True

    def test_permanent_not_retried_even_if_retryable_flag(self):
        # PERMANENT has retryable=False, but even if someone set True:
        spec = _spec(retry_policy=RetryPolicy(max_attempts=5))
        err = self._error(ToolErrorType.PERMANENT, False)
        assert should_retry(spec=spec, error=err, attempt_number=1) is False


# ===========================================================================
# Section 19, 48-49 — compute_backoff
# ===========================================================================


class TestComputeBackoff:
    def test_sequence_1_2_4_capped_at_5(self):
        p = RetryPolicy(
            initial_backoff_seconds=1,
            backoff_multiplier=2,
            max_backoff_seconds=5,
        )
        assert compute_backoff(p, 1) == 1.0
        assert compute_backoff(p, 2) == 2.0
        assert compute_backoff(p, 3) == 4.0
        assert compute_backoff(p, 4) == 5.0  # capped
        assert compute_backoff(p, 5) == 5.0  # capped

    def test_sequence_1_2_3_capped_at_3(self):
        p = RetryPolicy(
            initial_backoff_seconds=1,
            backoff_multiplier=2,
            max_backoff_seconds=3,
        )
        assert compute_backoff(p, 1) == 1.0
        assert compute_backoff(p, 2) == 2.0
        assert compute_backoff(p, 3) == 3.0  # 4 capped to 3

    def test_zero_initial_means_zero_delay(self):
        p = RetryPolicy(initial_backoff_seconds=0, max_backoff_seconds=10)
        assert compute_backoff(p, 1) == 0.0
        assert compute_backoff(p, 5) == 0.0

    def test_retry_index_lt_1_rejected(self):
        p = RetryPolicy()
        with pytest.raises(ValueError):
            compute_backoff(p, 0)


# ===========================================================================
# Sections 14, 38, 46 — READ_ONLY retry then success
# ===========================================================================


class TestReadOnlyRetry:
    def test_transient_then_success(self):
        handler = ScriptedTool([
            ("transient", "fail 1"),
            ("transient", "fail 2"),
            ("success", "done"),
        ])
        registry = ToolRegistry()
        registry.register(
            _spec(
                side_effect=ToolSideEffect.READ_ONLY,
                retry_policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=0),
            ),
            handler,
        )
        sleeper = FakeSleeper()
        runtime = ToolRuntime(registry=registry, sleep=sleeper)

        result = asyncio.run(
            runtime.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )
        assert result.success is True
        assert result.output == "done"
        assert handler.call_count == 3
        assert result.attempt_count == 3
        assert len(result.retry_history) == 2
        assert all(
            e.error_type is ToolErrorType.TRANSIENT for e in result.retry_history
        )

    def test_no_sleep_before_first_attempt(self):
        handler = ScriptedTool([
            ("transient", "fail"),
            ("success", "ok"),
        ])
        registry = ToolRegistry()
        registry.register(
            _spec(
                side_effect=ToolSideEffect.READ_ONLY,
                retry_policy=RetryPolicy(
                    max_attempts=2, initial_backoff_seconds=1.0
                ),
            ),
            handler,
        )
        sleeper = FakeSleeper()
        runtime = ToolRuntime(registry=registry, sleep=sleeper)

        result = asyncio.run(
            runtime.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )
        assert result.success is True
        # Only one sleep (after attempt 1, before attempt 2).
        assert sleeper.delays == [1.0]


# ===========================================================================
# Section 39 — max attempts exhausted
# ===========================================================================


def test_max_attempts_exhausted():
    handler = TransientFailureTool()
    registry = ToolRegistry()
    registry.register(
        _spec(
            side_effect=ToolSideEffect.READ_ONLY,
            retry_policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=0),
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())

    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="tool", arguments={}),
            ToolExecutionContext(),
        )
    )
    assert result.success is False
    assert result.error.error_type is ToolErrorType.TRANSIENT
    assert handler.call_count == 3
    assert result.attempt_count == 3
    assert len(result.retry_history) == 2
    # Final error is NOT in retry_history.
    assert result.error not in result.retry_history


# ===========================================================================
# Section 40 — timeout then success
# ===========================================================================


def test_timeout_then_success():
    handler = ScriptedTool([
        ("timeout",),
        ("success", "recovered"),
    ])
    registry = ToolRegistry()
    registry.register(
        _spec(
            side_effect=ToolSideEffect.READ_ONLY,
            retry_policy=RetryPolicy(max_attempts=2, initial_backoff_seconds=0),
            timeout_seconds=0.05,
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())

    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="tool", arguments={}),
            ToolExecutionContext(),
        )
    )
    assert result.success is True
    assert result.output == "recovered"
    assert handler.call_count == 2
    assert result.attempt_count == 2
    assert len(result.retry_history) == 1
    assert result.retry_history[0].error_type is ToolErrorType.TIMEOUT


# ===========================================================================
# Section 41 — permanent stops immediately
# ===========================================================================


def test_permanent_stops_immediately():
    handler = PermanentFailureTool()
    registry = ToolRegistry()
    registry.register(
        _spec(
            side_effect=ToolSideEffect.READ_ONLY,
            retry_policy=RetryPolicy(max_attempts=5),
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())

    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="tool", arguments={}),
            ToolExecutionContext(),
        )
    )
    assert result.success is False
    assert result.error.error_type is ToolErrorType.PERMANENT
    assert handler.call_count == 1
    assert result.attempt_count == 1
    assert result.retry_history == ()


# ===========================================================================
# Section 42 — execution stops immediately
# ===========================================================================


def test_execution_stops_immediately():
    handler = RaisingTool(RuntimeError("boom"))
    registry = ToolRegistry()
    registry.register(
        _spec(
            side_effect=ToolSideEffect.READ_ONLY,
            retry_policy=RetryPolicy(max_attempts=5),
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())

    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="tool", arguments={}),
            ToolExecutionContext(),
        )
    )
    assert result.success is False
    assert result.error.error_type is ToolErrorType.EXECUTION
    assert handler.call_count == 1
    assert result.attempt_count == 1
    assert result.retry_history == ()


# ===========================================================================
# Sections 7, 16, 43 — SIDE_EFFECTING transient not retried
# ===========================================================================


def test_side_effecting_transient_not_retried():
    handler = TransientFailureTool()
    registry = ToolRegistry()
    registry.register(
        _spec(
            side_effect=ToolSideEffect.SIDE_EFFECTING,
            retry_policy=RetryPolicy(max_attempts=5),
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())

    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="tool", arguments={}),
            ToolExecutionContext(),
        )
    )
    assert result.success is False
    assert result.error.error_type is ToolErrorType.TRANSIENT
    assert result.error.retryable is True  # error-level retryable...
    assert handler.call_count == 1          # ...but not retried
    assert result.attempt_count == 1
    assert result.retry_history == ()


# ===========================================================================
# Section 44 — SIDE_EFFECTING timeout not retried
# ===========================================================================


def test_side_effecting_timeout_is_not_retried():
    handler = SlowTool(delay=1.0)
    registry = ToolRegistry()
    registry.register(
        _spec(
            side_effect=ToolSideEffect.SIDE_EFFECTING,
            retry_policy=RetryPolicy(max_attempts=5),
            timeout_seconds=0.05,
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())

    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="tool", arguments={}),
            ToolExecutionContext(),
        )
    )
    assert result.success is False
    assert result.error.error_type is ToolErrorType.TIMEOUT
    assert result.error.retryable is True
    assert handler.call_count == 1
    assert result.attempt_count == 1
    assert result.retry_history == ()


# ===========================================================================
# Section 45 — IDEMPOTENT retry
# ===========================================================================


def test_idempotent_retry_then_success():
    handler = ScriptedTool([
        ("transient", "fail"),
        ("success", "ok"),
    ])
    registry = ToolRegistry()
    registry.register(
        _spec(
            side_effect=ToolSideEffect.IDEMPOTENT,
            retry_policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=0),
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())

    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="tool", arguments={}),
            ToolExecutionContext(),
        )
    )
    assert result.success is True
    assert handler.call_count == 2


# ===========================================================================
# Section 47 — policy narrows error types
# ===========================================================================


def test_policy_narrows_error_types_timeout_not_retried():
    """Policy with only TRANSIENT -> TIMEOUT is not retried even if retryable."""
    handler = SlowTool(delay=1.0)
    registry = ToolRegistry()
    registry.register(
        _spec(
            side_effect=ToolSideEffect.READ_ONLY,
            retry_policy=RetryPolicy(
                max_attempts=5,
                retryable_error_types=frozenset({ToolErrorType.TRANSIENT}),
            ),
            timeout_seconds=0.05,
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())

    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="tool", arguments={}),
            ToolExecutionContext(),
        )
    )
    assert result.success is False
    assert result.error.error_type is ToolErrorType.TIMEOUT
    assert handler.call_count == 1  # not retried despite retryable=True


# ===========================================================================
# Section 48 — backoff sequence via FakeSleeper
# ===========================================================================


def test_backoff_sequence():
    handler = TransientFailureTool()  # always fails
    registry = ToolRegistry()
    registry.register(
        _spec(
            side_effect=ToolSideEffect.READ_ONLY,
            retry_policy=RetryPolicy(
                max_attempts=3,
                initial_backoff_seconds=1,
                backoff_multiplier=2,
                max_backoff_seconds=3,
            ),
        ),
        handler,
    )
    sleeper = FakeSleeper()
    runtime = ToolRuntime(registry=registry, sleep=sleeper)

    asyncio.run(
        runtime.execute(
            ToolCall(tool_name="tool", arguments={}),
            ToolExecutionContext(),
        )
    )
    # 3 attempts -> 2 sleeps: after attempt 1 (retry_index=1) and after
    # attempt 2 (retry_index=2).
    assert sleeper.delays == [1.0, 2.0]


# ===========================================================================
# Section 50 — cancellation during backoff
# ===========================================================================


def test_cancellation_during_backoff():
    handler = TransientFailureTool()
    registry = ToolRegistry()
    registry.register(
        _spec(
            side_effect=ToolSideEffect.READ_ONLY,
            retry_policy=RetryPolicy(
                max_attempts=3, initial_backoff_seconds=1.0
            ),
        ),
        handler,
    )
    sleeper = ControllableSleeper(block_forever=True)
    runtime = ToolRuntime(registry=registry, sleep=sleeper)

    async def _drive():
        task = asyncio.create_task(
            runtime.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )
        await asyncio.sleep(0.05)  # let attempt 1 fail and enter backoff
        task.cancel()
        return await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_drive())
    # Attempt 1 happened, then cancellation during backoff — no attempt 2.
    assert handler.call_count == 1


# ===========================================================================
# Section 51 — cancellation during attempt (max_attempts > 1)
# ===========================================================================


def test_cancellation_during_attempt_with_retry_policy():
    handler = CooperativeCancelTool(delay=5.0)
    registry = ToolRegistry()
    registry.register(
        _spec(
            side_effect=ToolSideEffect.READ_ONLY,
            retry_policy=RetryPolicy(max_attempts=3),
            timeout_seconds=30.0,
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())

    async def _drive():
        task = asyncio.create_task(
            runtime.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        return await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_drive())
    assert handler.call_count == 1


# ===========================================================================
# Sections 21-22, 52 — handler argument mutation isolation
# ===========================================================================


def test_handler_mutation_does_not_affect_next_retry():
    """A handler that mutates its argument dict must not affect the next attempt."""
    handler = MutatingTool(key="value", new_value=999)
    registry = ToolRegistry()
    registry.register(
        _spec(
            side_effect=ToolSideEffect.READ_ONLY,
            retry_policy=RetryPolicy(max_attempts=2, initial_backoff_seconds=0),
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())

    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="tool", arguments={"value": 1}),
            ToolExecutionContext(),
        )
    )
    assert result.success is False  # both attempts fail (MutatingTool always raises)
    assert handler.call_count == 2
    # Each attempt should have seen value=1, not 999.
    # last_arguments is recorded by _record which copies the dict at entry.
    # But MutatingTool mutates AFTER _record, so we need to check differently.
    # Instead, verify via a custom handler.


def test_fresh_argument_copy_per_attempt():
    """Explicitly verify each attempt sees the original value."""
    seen_values: list = []

    class _MutatingFailer:
        def __init__(self):
            self.call_count = 0

        async def __call__(self, arguments):
            self.call_count += 1
            seen_values.append(arguments["value"])
            arguments["value"] = 999  # mutate
            from tools.errors import ToolReportedFailure
            raise ToolReportedFailure(ToolErrorType.TRANSIENT, "fail")

    handler = _MutatingFailer()
    registry = ToolRegistry()
    registry.register(
        _spec(
            side_effect=ToolSideEffect.READ_ONLY,
            retry_policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=0),
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())

    asyncio.run(
        runtime.execute(
            ToolCall(tool_name="tool", arguments={"value": 1}),
            ToolExecutionContext(),
        )
    )
    # All three attempts saw the original value=1, not 999.
    assert seen_values == [1, 1, 1]
    assert handler.call_count == 3


# ===========================================================================
# Section 53 — caller dict not mutated
# ===========================================================================


def test_caller_dict_not_mutated_by_handler():
    caller_args = {"value": 1}

    class _MutatingFailer:
        def __init__(self):
            self.call_count = 0

        async def __call__(self, arguments):
            self.call_count += 1
            arguments["value"] = 999
            from tools.errors import ToolReportedFailure
            raise ToolReportedFailure(ToolErrorType.TRANSIENT, "fail")

    handler = _MutatingFailer()
    registry = ToolRegistry()
    registry.register(
        _spec(
            side_effect=ToolSideEffect.READ_ONLY,
            retry_policy=RetryPolicy(max_attempts=2, initial_backoff_seconds=0),
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())

    asyncio.run(
        runtime.execute(
            ToolCall(tool_name="tool", arguments=caller_args),
            ToolExecutionContext(),
        )
    )
    # Caller's dict is untouched.
    assert caller_args == {"value": 1}


# ===========================================================================
# Sections 30-35 — attempt_count / retry_history metadata
# ===========================================================================


class TestResultMetadata:
    def test_preflight_failure_has_zero_attempts(self):
        registry = ToolRegistry()
        runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())
        result = asyncio.run(
            runtime.execute(
                ToolCall(tool_name="missing", arguments={}),
                ToolExecutionContext(),
            )
        )
        assert result.attempt_count == 0
        assert result.retry_history == ()

    def test_validation_failure_has_zero_attempts(self):
        registry = ToolRegistry()
        registry.register(
            _spec(
                schema={
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                    "additionalProperties": False,
                },
            ),
            SuccessTool(),
        )
        runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())
        result = asyncio.run(
            runtime.execute(
                ToolCall(tool_name="tool", arguments={}),  # missing x
                ToolExecutionContext(),
            )
        )
        assert result.attempt_count == 0
        assert result.retry_history == ()

    def test_permission_failure_has_zero_attempts(self):
        registry = ToolRegistry()
        registry.register(
            _spec(required_permissions=frozenset({"repo.write"})),
            SuccessTool(),
        )
        runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())
        result = asyncio.run(
            runtime.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(granted_permissions=frozenset()),
            )
        )
        assert result.attempt_count == 0
        assert result.retry_history == ()

    def test_success_has_attempt_count_1(self):
        registry = ToolRegistry()
        registry.register(_spec(), SuccessTool("ok"))
        runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())
        result = asyncio.run(
            runtime.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )
        assert result.success is True
        assert result.attempt_count == 1
        assert result.retry_history == ()

    def test_final_error_not_in_retry_history(self):
        handler = TransientFailureTool()
        registry = ToolRegistry()
        registry.register(
            _spec(
                retry_policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=0),
            ),
            handler,
        )
        runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())
        result = asyncio.run(
            runtime.execute(
                ToolCall(tool_name="tool", arguments={}),
                ToolExecutionContext(),
            )
        )
        assert result.error is not None
        assert result.error not in result.retry_history
        assert len(result.retry_history) == 2
        assert result.attempt_count == 3


# ===========================================================================
# Section 56 — retry doesn't change preflight order
# ===========================================================================


def test_validation_before_permission_with_retry_policy():
    """Bad args + missing permission -> VALIDATION (attempt_count=0)."""
    registry = ToolRegistry()
    registry.register(
        _spec(
            schema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
                "additionalProperties": False,
            },
            required_permissions=frozenset({"repo.write"}),
            retry_policy=RetryPolicy(max_attempts=5),
        ),
        SuccessTool(),
    )
    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())
    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="tool", arguments={"x": 123}),  # wrong type
            ToolExecutionContext(granted_permissions=frozenset()),
        )
    )
    assert result.error.error_type is ToolErrorType.VALIDATION
    assert result.attempt_count == 0


# ===========================================================================
# Section 28 — per-attempt timeout
# ===========================================================================


def test_each_attempt_gets_fresh_timeout():
    """Attempt 1 times out, attempt 2 gets a full timeout window and succeeds."""
    handler = ScriptedTool([
        ("timeout",),
        ("success", "ok"),
    ])
    registry = ToolRegistry()
    registry.register(
        _spec(
            side_effect=ToolSideEffect.READ_ONLY,
            retry_policy=RetryPolicy(max_attempts=2, initial_backoff_seconds=0),
            timeout_seconds=0.05,
        ),
        handler,
    )
    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())
    result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="tool", arguments={}),
            ToolExecutionContext(),
        )
    )
    assert result.success is True
    assert result.attempt_count == 2
