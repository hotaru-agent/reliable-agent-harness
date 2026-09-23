"""Smoke test for Phase 3 Step 2 — agent action controller & replan gate.

Proves the end-to-end governance guarantee:

    A state-1  → READY
    A state-1  → READY
    A state-1  → DUPLICATE_CALL → ReplanSignal → REPLAN_REQUIRED
    A (4th)    → blocked (ReplanRequiredError, handler not called)
    acknowledge_replan() → READY, history cleared
    B state-2  → READY (new plan, not contaminated)
    C state-3  → READY

Fully offline, deterministic, no LLM, no network, no MCP.
"""

from __future__ import annotations

import asyncio

import pytest

from harness.action_controller import (
    ActionControllerState,
    AgentActionController,
    ReplanRequiredError,
)
from harness.loop_detection import LoopDetector, LoopDetectorConfig
from tests.fake_tools import FakeSleeper, SuccessTool
from tools import (
    ToolCall,
    ToolExecutionContext,
    ToolRegistry,
    ToolRuntime,
    ToolSpec,
)


class _ScriptedProgress:
    """Returns tokens from a list, repeating the last."""

    def __init__(self, tokens: list[str | None]) -> None:
        self.tokens = tokens
        self.call_count = 0

    def get_progress_token(self, call, result) -> str | None:
        idx = min(self.call_count, len(self.tokens) - 1)
        self.call_count += 1
        return self.tokens[idx]


def test_phase3_step2_smoke():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="A",
            description="tool A",
            input_schema={"type": "object"},
            required_permissions=frozenset(),
            timeout_seconds=1.0,
            side_effect=__import__("tools").ToolSideEffect.READ_ONLY,
        ),
        SuccessTool("a-result"),
    )
    registry.register(
        ToolSpec(
            name="B",
            description="tool B",
            input_schema={"type": "object"},
            required_permissions=frozenset(),
            timeout_seconds=1.0,
            side_effect=__import__("tools").ToolSideEffect.READ_ONLY,
        ),
        SuccessTool("b-result"),
    )
    registry.register(
        ToolSpec(
            name="C",
            description="tool C",
            input_schema={"type": "object"},
            required_permissions=frozenset(),
            timeout_seconds=1.0,
            side_effect=__import__("tools").ToolSideEffect.READ_ONLY,
        ),
        SuccessTool("c-result"),
    )

    runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())
    detector = LoopDetector(
        config=LoopDetectorConfig(duplicate_threshold=3),
    )
    progress = _ScriptedProgress([
        "state-1", "state-1", "state-1",  # first 3 A's
        "state-2", "state-3",              # B, C after replan
    ])
    controller = AgentActionController(
        tool_runtime=runtime,
        loop_detector=detector,
        progress_provider=progress,
    )

    ctx = ToolExecutionContext()
    call_a = ToolCall(tool_name="A", arguments={})

    # 1. A state-1 → READY
    r0 = asyncio.run(controller.execute(call_a, ctx))
    assert r0.replan_signal is None
    assert controller.state is ActionControllerState.READY

    # 2. A state-1 → READY
    r1 = asyncio.run(controller.execute(call_a, ctx))
    assert r1.replan_signal is None
    assert controller.state is ActionControllerState.READY

    # 3. A state-1 → DUPLICATE_CALL → REPLAN_REQUIRED
    r2 = asyncio.run(controller.execute(call_a, ctx))
    assert r2.replan_signal is not None
    assert r2.replan_signal.reason is __import__(
        "harness.loop_detection", fromlist=["LoopDetectionReason"]
    ).LoopDetectionReason.DUPLICATE_CALL
    assert controller.state is ActionControllerState.REPLAN_REQUIRED
    assert controller.next_action_index == 3

    # 4. A again → blocked
    with pytest.raises(ReplanRequiredError):
        asyncio.run(controller.execute(call_a, ctx))
    assert controller.next_action_index == 3  # unchanged

    # 5. acknowledge_replan → READY, history cleared
    controller.acknowledge_replan()
    assert controller.state is ActionControllerState.READY
    assert controller.pending_replan_signal is None
    assert len(controller.loop_detector.history) == 0

    # 6. B state-2 → READY (new plan, not contaminated)
    call_b = ToolCall(tool_name="B", arguments={})
    r3 = asyncio.run(controller.execute(call_b, ctx))
    assert r3.replan_signal is None
    assert controller.state is ActionControllerState.READY
    assert r3.action_record.action_index == 3  # continues from 3

    # 7. C state-3 → READY
    call_c = ToolCall(tool_name="C", arguments={})
    r4 = asyncio.run(controller.execute(call_c, ctx))
    assert r4.replan_signal is None
    assert controller.state is ActionControllerState.READY
    assert r4.action_record.action_index == 4
    assert controller.next_action_index == 5
