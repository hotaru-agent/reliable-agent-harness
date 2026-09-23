"""Execution-boundary types for the harness run loop (Phase 1 Step 2).

This module defines the lightweight, explicit state that flows between
steps and is restored on resume, plus the minimal ``StepExecutor``
protocol that the ``HarnessRuntime`` depends on.

It is deliberately NOT a Tool Runtime: the executor here is an injectable
fake boundary used to prove the run loop, checkpoint boundary and resume
semantics. Real tool execution belongs to a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol

from harness.state import Checkpoint


@dataclass
class ExecutionState:
    """State passed between steps and restored on resume.

    This is not the final Context Manager. It only captures the fields
    that must survive an interruption + resume so that already-completed
    work is not repeated.
    """

    agent_state: dict[str, Any] = field(default_factory=dict)
    critical_context: dict[str, Any] = field(default_factory=dict)
    tool_state: dict[str, Any] = field(default_factory=dict)
    completed_actions: tuple[str, ...] = ()
    pending_action: Optional[dict[str, Any]] = None

    @classmethod
    def from_checkpoint(cls, checkpoint: Checkpoint) -> "ExecutionState":
        """Build an ``ExecutionState`` from a checkpoint snapshot.

        Uses the checkpoint's deep-copied fields (the caller is expected
        to have obtained the checkpoint via the store, which already
        returns an isolated copy). A fresh deepcopy is performed here so
        that the returned state never shares mutable references with the
        source checkpoint object.
        """
        import copy

        return cls(
            agent_state=copy.deepcopy(checkpoint.agent_state),
            critical_context=copy.deepcopy(checkpoint.critical_context),
            tool_state=copy.deepcopy(checkpoint.tool_state),
            completed_actions=checkpoint.completed_actions,
            pending_action=copy.deepcopy(checkpoint.pending_action),
        )

    def to_checkpoint_fields(self) -> dict[str, Any]:
        """Return a dict of the checkpoint-shaped fields for snapshotting."""
        import copy

        return {
            "agent_state": copy.deepcopy(self.agent_state),
            "critical_context": copy.deepcopy(self.critical_context),
            "tool_state": copy.deepcopy(self.tool_state),
            "completed_actions": self.completed_actions,
            "pending_action": copy.deepcopy(self.pending_action),
        }


class StepOutcome(Enum):
    """Minimal outcome of a single step.

    Only two values in this phase. Retry / replan / timeout / tool-error
    outcomes belong to later phases.
    """

    CONTINUE = "continue"
    COMPLETE = "complete"


@dataclass
class StepResult:
    """The result of executing one logical step.

    The executor is responsible for a single step only: it must NOT
    modify Run / Task lifecycle, save checkpoints or perform resume.
    All of those are the ``HarnessRuntime``'s responsibility.
    """

    outcome: StepOutcome
    state: ExecutionState


class StepExecutor(Protocol):
    """Protocol for a single-step execution boundary.

    The executor learns which step it is running from ``run.current_step``
    (the index of the next step to execute). It returns a ``StepResult``
    describing the outcome and the updated execution state.
    """

    async def execute_step(
        self,
        task: "Task",  # noqa: F821 — forward ref to harness.state.Task
        run: "Run",  # noqa: F821 — forward ref to harness.state.Run
        state: ExecutionState,
    ) -> StepResult:
        ...
