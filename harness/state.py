"""Explicit runtime state models for the agent harness.

This module is the single source of truth for the Task / Run /
Checkpoint lifecycle in Phase 1 Step 1. It deliberately keeps to the
Python standard library so the state layer has no external runtime
dependencies and stays easy to test deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional

from harness.errors import InvalidStateTransitionError


def _utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _ensure_aware(dt: Optional[datetime], field_name: str) -> Optional[datetime]:
    """Validate that a datetime, when present, is timezone-aware."""
    if dt is not None and dt.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return dt


# ---------------------------------------------------------------------------
# Status enums
# ---------------------------------------------------------------------------


class TaskStatus(Enum):
    """Lifecycle status for a long-horizon user Task.

    A Task spans multiple Runs, so it intentionally has no
    ``INTERRUPTED`` state: a single Run being interrupted does not end
    the user's long-term goal.
    """

    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


class RunStatus(Enum):
    """Lifecycle status for a single execution attempt of a Task."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


# Terminal statuses: once reached the entity cannot move forward again.
RUN_TERMINAL_STATUSES = frozenset(
    {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.INTERRUPTED}
)
TASK_TERMINAL_STATUSES = frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED})

# Explicit, auditable transition tables.
TASK_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset({TaskStatus.RUNNING}),
    TaskStatus.RUNNING: frozenset(
        {TaskStatus.WAITING, TaskStatus.COMPLETED, TaskStatus.FAILED}
    ),
    TaskStatus.WAITING: frozenset({TaskStatus.RUNNING, TaskStatus.FAILED}),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset(),
}

RUN_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PENDING: frozenset({RunStatus.RUNNING}),
    RunStatus.RUNNING: frozenset(
        {RunStatus.WAITING, RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.INTERRUPTED}
    ),
    RunStatus.WAITING: frozenset(
        {RunStatus.RUNNING, RunStatus.FAILED, RunStatus.INTERRUPTED}
    ),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.INTERRUPTED: frozenset(),
}


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """A long-horizon user goal that may be attempted by multiple Runs."""

    task_id: str
    goal: str
    created_at: datetime
    status: TaskStatus = TaskStatus.PENDING
    constraints: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if not self.goal or not self.goal.strip():
            raise ValueError("goal must be non-empty")
        self.created_at = _ensure_aware(self.created_at, "created_at")
        if not isinstance(self.status, TaskStatus):
            raise ValueError("status must be a TaskStatus")
        if not isinstance(self.constraints, tuple):
            raise ValueError("constraints must be a tuple")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dict")

    def transition_to(self, new_status: TaskStatus) -> None:
        """Transition this Task to ``new_status`` or raise."""
        transition_task_status(self, new_status)


def transition_task_status(task: Task, new_status: TaskStatus) -> None:
    """Apply a legal Task status transition, rejecting illegal ones.

    Task has no ``started_at`` / ``ended_at`` fields in this phase (only
    ``created_at``), so there is no timestamp to maintain and therefore no
    ``now`` parameter. If Task-level lifecycle timestamps become necessary
    in a later phase, an explicit parameter can be reintroduced then.
    """
    if not isinstance(new_status, TaskStatus):
        raise ValueError("new_status must be a TaskStatus")
    allowed = TASK_TRANSITIONS.get(task.status, frozenset())
    if new_status not in allowed:
        raise InvalidStateTransitionError(
            entity="Task",
            current=task.status.value,
            target=new_status.value,
        )
    task.status = new_status


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass
class Run:
    """A single execution attempt of a Task."""

    run_id: str
    task_id: str
    status: RunStatus = RunStatus.PENDING
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    current_step: int = 0
    retry_count: int = 0
    context_state: dict[str, Any] = field(default_factory=dict)
    last_checkpoint_id: Optional[str] = None
    # Lineage: which historical checkpoint this Run was resumed from.
    # ``None`` for a fresh Run; set to the source checkpoint_id on resume.
    # Distinct from ``last_checkpoint_id`` which tracks this Run's own
    # most recently produced checkpoint.
    resumed_from_checkpoint_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if not isinstance(self.status, RunStatus):
            raise ValueError("status must be a RunStatus")
        self.started_at = _ensure_aware(self.started_at, "started_at")
        self.ended_at = _ensure_aware(self.ended_at, "ended_at")
        if self.current_step < 0:
            raise ValueError("current_step must be >= 0")
        if self.retry_count < 0:
            raise ValueError("retry_count must be >= 0")
        if not isinstance(self.context_state, dict):
            raise ValueError("context_state must be a dict")

    @property
    def is_terminal(self) -> bool:
        """True when this Run has reached a terminal status."""
        return self.status in RUN_TERMINAL_STATUSES

    def transition_to(
        self, new_status: RunStatus, now: Optional[datetime] = None
    ) -> None:
        """Transition this Run to ``new_status`` or raise."""
        transition_run_status(self, new_status, now=now)


def transition_run_status(
    run: Run, new_status: RunStatus, now: Optional[datetime] = None
) -> None:
    """Apply a legal Run status transition, rejecting illegal ones.

    Lifecycle timestamps are maintained here:

    * ``PENDING -> RUNNING`` sets ``started_at`` if it is still ``None``.
    * entering a terminal status (COMPLETED / FAILED / INTERRUPTED) sets
      ``ended_at``.

    ``now`` may be injected for deterministic tests.
    """
    if not isinstance(new_status, RunStatus):
        raise ValueError("new_status must be a RunStatus")
    allowed = RUN_TRANSITIONS.get(run.status, frozenset())
    if new_status not in allowed:
        raise InvalidStateTransitionError(
            entity="Run",
            current=run.status.value,
            target=new_status.value,
        )

    moment = now if now is not None else _utc_now()
    if moment.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    if run.status == RunStatus.PENDING and new_status == RunStatus.RUNNING:
        if run.started_at is None:
            run.started_at = moment

    if new_status in RUN_TERMINAL_STATUSES:
        run.ended_at = moment

    run.status = new_status


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


@dataclass
class Checkpoint:
    """A restorable snapshot of a Run's execution state."""

    checkpoint_id: str
    run_id: str
    step_index: int
    created_at: datetime
    agent_state: dict[str, Any] = field(default_factory=dict)
    critical_context: dict[str, Any] = field(default_factory=dict)
    tool_state: dict[str, Any] = field(default_factory=dict)
    completed_actions: tuple[str, ...] = ()
    pending_action: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id must be non-empty")
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.step_index < 0:
            raise ValueError("step_index must be >= 0")
        self.created_at = _ensure_aware(self.created_at, "created_at")
        if not isinstance(self.agent_state, Mapping):
            raise ValueError("agent_state must be a dict")
        if not isinstance(self.critical_context, Mapping):
            raise ValueError("critical_context must be a dict")
        if not isinstance(self.tool_state, Mapping):
            raise ValueError("tool_state must be a dict")
        if not isinstance(self.completed_actions, tuple):
            raise ValueError("completed_actions must be a tuple")
        if self.pending_action is not None and not isinstance(
            self.pending_action, Mapping
        ):
            raise ValueError("pending_action must be a dict or None")
