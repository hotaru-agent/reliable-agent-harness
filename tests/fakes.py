"""Deterministic fakes and helpers for Phase 1 Step 2 runtime tests.

Everything here is fully offline: no network, no LLM, no tools, no real
wall-clock, no random IDs. The ``FakeStepExecutor`` is an execution
boundary used to prove the run loop / checkpoint / resume semantics — it
is explicitly NOT a Tool Runtime.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

from harness.execution import ExecutionState, StepOutcome, StepResult


class FakeClock:
    """Injectable clock returning strictly increasing UTC datetimes.

    Each call advances the internal time by one second so every
    timestamp produced during a run is distinct and deterministic.
    """

    def __init__(self, start: datetime | None = None) -> None:
        self._t = start or datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        result = self._t
        self._t = self._t + timedelta(seconds=1)
        return result


def make_id_factory(prefix: str, start: int = 1):
    """Return a callable producing ids like ``R-001``, ``R-002``, ..."""
    counter = start - 1

    def _factory() -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}-{counter:03d}"

    return _factory


class FakeStepExecutor:
    """Deterministic single-step executor for runtime tests.

    Configuration:

    * ``complete_at_step`` — the step index that returns ``COMPLETE``.
    * ``cancel_at_step`` — the step index that raises
      ``asyncio.CancelledError`` (takes precedence over fail/complete).
    * ``fail_at_step`` — the step index that raises an ordinary exception.
    * ``fail_exc`` — the exception instance to raise (default
      ``RuntimeError("boom")``).

    Every successful step appends ``"step-{N}"`` to ``completed_actions``
    so resume can prove execution state (not just the step counter) is
    restored.

    ``executed_steps`` records the step index seen on each invocation,
    in call order — used to assert that resume does not re-run earlier
    steps.
    """

    def __init__(
        self,
        *,
        complete_at_step: int,
        cancel_at_step: int | None = None,
        fail_at_step: int | None = None,
        fail_exc: BaseException | None = None,
    ) -> None:
        self.complete_at_step = complete_at_step
        self.cancel_at_step = cancel_at_step
        self.fail_at_step = fail_at_step
        self.fail_exc: BaseException = fail_exc or RuntimeError("boom")
        self.executed_steps: list[int] = []

    async def execute_step(self, task, run, state: ExecutionState) -> StepResult:
        n = run.current_step
        self.executed_steps.append(n)

        if self.cancel_at_step is not None and n == self.cancel_at_step:
            raise asyncio.CancelledError()

        if self.fail_at_step is not None and n == self.fail_at_step:
            raise self.fail_exc

        # Successful step: grow completed_actions with a stable label.
        new_actions = state.completed_actions + (f"step-{n}",)
        new_state = ExecutionState(
            agent_state=dict(state.agent_state),
            critical_context=dict(state.critical_context),
            tool_state=dict(state.tool_state),
            completed_actions=new_actions,
            pending_action=state.pending_action,
        )
        outcome = (
            StepOutcome.COMPLETE
            if n == self.complete_at_step
            else StepOutcome.CONTINUE
        )
        return StepResult(outcome=outcome, state=new_state)
