"""Checkpoint storage abstraction and an in-memory implementation.

The ``CheckpointStore`` protocol defines the async surface that the
future Harness Runtime will depend on. ``InMemoryCheckpointStore`` is a
deterministic, offline implementation suitable for tests and local
development.

Snapshot semantics
--------------------
Both ``save`` and ``get`` perform a deep copy so that callers can never
mutate a stored checkpoint (or a returned one) and silently corrupt the
store's history. This is the core correctness guarantee tested in
``tests/test_checkpoint_store.py``.
"""

from __future__ import annotations

import copy
from typing import Protocol, runtime_checkable

from harness.state import Checkpoint


@runtime_checkable
class CheckpointStore(Protocol):
    """Async storage contract for restorable Run checkpoints."""

    async def save(self, checkpoint: Checkpoint) -> None:
        """Persist a checkpoint snapshot."""
        ...

    async def get(self, checkpoint_id: str) -> Checkpoint | None:
        """Return a snapshot for ``checkpoint_id`` or ``None`` if missing."""
        ...

    async def latest_for_run(self, run_id: str) -> Checkpoint | None:
        """Return the latest checkpoint for ``run_id`` or ``None``."""
        ...


class InMemoryCheckpointStore:
    """Deterministic in-memory ``CheckpointStore`` with snapshot semantics.

    "Latest" is defined as the checkpoint with the largest ``step_index``;
    ties are broken by ``created_at`` and finally by ``checkpoint_id`` so
    that even identical ``(step_index, created_at)`` pairs produce a fully
    deterministic result without relying on dict insertion order.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, Checkpoint] = {}

    async def save(self, checkpoint: Checkpoint) -> None:
        """Store a deep-copy snapshot of ``checkpoint``."""
        self._by_id[checkpoint.checkpoint_id] = copy.deepcopy(checkpoint)

    async def get(self, checkpoint_id: str) -> Checkpoint | None:
        """Return a deep-copy snapshot for ``checkpoint_id`` or ``None``."""
        stored = self._by_id.get(checkpoint_id)
        if stored is None:
            return None
        return copy.deepcopy(stored)

    async def latest_for_run(self, run_id: str) -> Checkpoint | None:
        """Return the latest checkpoint for ``run_id`` or ``None``.

        Ordering key: ``(step_index, created_at, checkpoint_id)`` descending.
        ``created_at`` makes the result stable for equal ``step_index``;
        ``checkpoint_id`` is a final deterministic tiebreaker so that even
        checkpoints sharing both ``step_index`` and ``created_at`` produce
        a fully deterministic result without relying on dict insertion order.
        """
        candidates = [c for c in self._by_id.values() if c.run_id == run_id]
        if not candidates:
            return None
        latest = max(
            candidates,
            key=lambda c: (c.step_index, c.created_at, c.checkpoint_id),
        )
        return copy.deepcopy(latest)
