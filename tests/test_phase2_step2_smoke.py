"""Smoke test for Phase 2 Step 2 — deterministic retry & side-effect safety.

Proves the end-to-end retry guarantee:

    READ_ONLY tool, max_attempts=3
      -> attempt 1 TRANSIENT  -> retry (backoff)
      -> attempt 2 TRANSIENT  -> retry (backoff)
      -> attempt 3 SUCCESS

    SIDE_EFFECTING tool, max_attempts=3
      -> attempt 1 TRANSIENT  -> NO retry (safety gate)
      -> attempt 1 TIMEOUT    -> NO retry (safety gate)

Fully offline, deterministic, no LLM, no network, no MCP.
"""

from __future__ import annotations

import asyncio

from tests.fake_tools import FakeSleeper, ScriptedTool, SlowTool
from tools import (
    RetryPolicy,
    ToolCall,
    ToolErrorType,
    ToolExecutionContext,
    ToolRegistry,
    ToolRuntime,
    ToolSideEffect,
    ToolSpec,
)


def test_phase2_step2_smoke():
    registry = ToolRegistry()
    sleeper = FakeSleeper()

    # --- READ_ONLY tool that fails twice then succeeds. ---
    read_only_spec = ToolSpec(
        name="read_repo",
        description="read repo",
        input_schema={"type": "object"},
        required_permissions=frozenset(),
        timeout_seconds=1.0,
        side_effect=ToolSideEffect.READ_ONLY,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=1.0,
            backoff_multiplier=2.0,
            max_backoff_seconds=10.0,
        ),
    )
    read_handler = ScriptedTool([
        ("transient", "fail 1"),
        ("transient", "fail 2"),
        ("success", "repo-content"),
    ])
    registry.register(read_only_spec, read_handler)

    # --- SIDE_EFFECTING tool that fails with TRANSIENT. ---
    side_effect_spec = ToolSpec(
        name="send_message",
        description="send message",
        input_schema={"type": "object"},
        required_permissions=frozenset(),
        timeout_seconds=1.0,
        side_effect=ToolSideEffect.SIDE_EFFECTING,
        retry_policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=0),
    )
    side_handler = ScriptedTool([("transient", "send failed")])
    registry.register(side_effect_spec, side_handler)

    # --- SIDE_EFFECTING tool that times out. ---
    timeout_spec = ToolSpec(
        name="charge_card",
        description="charge card",
        input_schema={"type": "object"},
        required_permissions=frozenset(),
        timeout_seconds=0.05,
        side_effect=ToolSideEffect.SIDE_EFFECTING,
        retry_policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=0),
    )
    timeout_handler = SlowTool(delay=1.0)
    registry.register(timeout_spec, timeout_handler)

    runtime = ToolRuntime(registry=registry, sleep=sleeper)

    # 1. READ_ONLY retry then success.
    ok = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="read_repo", arguments={}),
            ToolExecutionContext(),
        )
    )
    assert ok.success is True
    assert ok.output == "repo-content"
    assert read_handler.call_count == 3
    assert ok.attempt_count == 3
    assert len(ok.retry_history) == 2
    assert all(
        e.error_type is ToolErrorType.TRANSIENT for e in ok.retry_history
    )
    # Backoff sequence: 1.0 after attempt 1, 2.0 after attempt 2.
    assert sleeper.delays == [1.0, 2.0]

    # 2. SIDE_EFFECTING transient -> NO retry.
    sleeper.delays.clear()
    fail = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="send_message", arguments={}),
            ToolExecutionContext(),
        )
    )
    assert fail.success is False
    assert fail.error.error_type is ToolErrorType.TRANSIENT
    assert fail.error.retryable is True  # error-level retryable...
    assert side_handler.call_count == 1  # ...but not retried
    assert fail.attempt_count == 1
    assert fail.retry_history == ()
    assert sleeper.delays == []  # no backoff sleep happened

    # 3. SIDE_EFFECTING timeout -> NO retry.
    timeout_result = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="charge_card", arguments={}),
            ToolExecutionContext(),
        )
    )
    assert timeout_result.success is False
    assert timeout_result.error.error_type is ToolErrorType.TIMEOUT
    assert timeout_result.error.retryable is True
    assert timeout_handler.call_count == 1  # not retried
    assert timeout_result.attempt_count == 1
    assert timeout_result.retry_history == ()
