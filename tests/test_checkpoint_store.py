"""Tests for the checkpoint store (Phase 1 Step 1).

Async store methods are exercised through ``asyncio.run`` so the test
suite needs no extra async pytest plugin and stays fully offline.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from harness import Checkpoint
from storage import CheckpointStore, InMemoryCheckpointStore

T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
T2 = datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)


def _checkpoint(checkpoint_id: str, run_id: str, step: int, created_at: datetime = T0):
    return Checkpoint(
        checkpoint_id=checkpoint_id,
        run_id=run_id,
        step_index=step,
        created_at=created_at,
        agent_state={"step": step},
        critical_context={"plan": ["s0", "s1"][: step + 1]},
        tool_state={},
        completed_actions=tuple(f"action-{i}" for i in range(step)),
        pending_action={"name": f"pending-{step}"} if step % 2 == 0 else None,
    )


# ---------------------------------------------------------------------------
# Test 1 — save / get round trip
# ---------------------------------------------------------------------------


def test_save_and_get():
    store = InMemoryCheckpointStore()
    cp = _checkpoint("C-001", "R-001", 0)
    asyncio.run(store.save(cp))
    got = asyncio.run(store.get("C-001"))
    assert got is not None
    assert got.checkpoint_id == "C-001"
    assert got.run_id == "R-001"
    assert got.step_index == 0
    assert got.critical_context == {"plan": ["s0"]}


# ---------------------------------------------------------------------------
# Test 2 — missing checkpoint returns None
# ---------------------------------------------------------------------------


def test_get_missing_returns_none():
    store = InMemoryCheckpointStore()
    assert asyncio.run(store.get("missing")) is None


# ---------------------------------------------------------------------------
# Test 3 — latest_for_run returns highest step
# ---------------------------------------------------------------------------


def test_latest_for_run_returns_highest_step():
    store = InMemoryCheckpointStore()
    asyncio.run(store.save(_checkpoint("C-0", "R-001", 0)))
    asyncio.run(store.save(_checkpoint("C-1", "R-001", 1)))
    asyncio.run(store.save(_checkpoint("C-2", "R-001", 2)))

    latest = asyncio.run(store.latest_for_run("R-001"))
    assert latest is not None
    assert latest.checkpoint_id == "C-2"
    assert latest.step_index == 2


# ---------------------------------------------------------------------------
# Test 4 — checkpoints do not leak across runs
# ---------------------------------------------------------------------------


def test_latest_for_run_isolated_per_run():
    store = InMemoryCheckpointStore()
    asyncio.run(store.save(_checkpoint("C-A1", "R-A", 0)))
    asyncio.run(store.save(_checkpoint("C-A2", "R-A", 1)))
    asyncio.run(store.save(_checkpoint("C-B1", "R-B", 5)))

    latest_a = asyncio.run(store.latest_for_run("R-A"))
    assert latest_a is not None
    assert latest_a.checkpoint_id == "C-A2"
    assert latest_a.run_id == "R-A"

    latest_b = asyncio.run(store.latest_for_run("R-B"))
    assert latest_b is not None
    assert latest_b.checkpoint_id == "C-B1"

    assert asyncio.run(store.latest_for_run("R-Z")) is None


# ---------------------------------------------------------------------------
# Test 5 — snapshot semantics on save (caller mutation does not leak in)
# ---------------------------------------------------------------------------


def test_save_snapshot_isolates_caller_mutation():
    store = InMemoryCheckpointStore()
    cp = _checkpoint("C-001", "R-001", 0)
    asyncio.run(store.save(cp))

    # Mutate the original object's nested mutable data.
    cp.critical_context["plan"].append("s2")
    cp.completed_actions = cp.completed_actions + ("extra",)
    cp.agent_state["step"] = 999

    got = asyncio.run(store.get("C-001"))
    assert got is not None
    assert got.critical_context == {"plan": ["s0"]}
    assert got.completed_actions == ()
    assert got.agent_state == {"step": 0}


# ---------------------------------------------------------------------------
# Test 6 — snapshot semantics on get (returned mutation does not leak back)
# ---------------------------------------------------------------------------


def test_get_returns_copy_protecting_store():
    store = InMemoryCheckpointStore()
    asyncio.run(store.save(_checkpoint("C-001", "R-001", 0)))

    first = asyncio.run(store.get("C-001"))
    assert first is not None
    first.critical_context["plan"].append("polluted")
    first.completed_actions = first.completed_actions + ("sneaky",)

    second = asyncio.run(store.get("C-001"))
    assert second is not None
    assert second.critical_context == {"plan": ["s0"]}
    assert second.completed_actions == ()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_in_memory_store_satisfies_protocol():
    store = InMemoryCheckpointStore()
    assert isinstance(store, CheckpointStore)


# ---------------------------------------------------------------------------
# Tie-break stability for equal step_index
# ---------------------------------------------------------------------------


def test_latest_for_run_tiebreak_by_created_at():
    store = InMemoryCheckpointStore()
    asyncio.run(store.save(_checkpoint("C-early", "R-001", 1, created_at=T0)))
    asyncio.run(store.save(_checkpoint("C-late", "R-001", 1, created_at=T2)))

    latest = asyncio.run(store.latest_for_run("R-001"))
    assert latest is not None
    assert latest.checkpoint_id == "C-late"


# ---------------------------------------------------------------------------
# Final tie-break by checkpoint_id when step_index AND created_at are equal
# ---------------------------------------------------------------------------


def test_latest_for_run_tiebreak_by_checkpoint_id():
    """When step_index and created_at are identical, checkpoint_id decides.

    This guarantees full determinism even for pathological inputs where two
    checkpoints share both ``step_index`` and ``created_at`` — the result
    must never depend on dict insertion order.
    """
    store = InMemoryCheckpointStore()
    # Insert in reverse id order to ensure the result is not just "last inserted".
    asyncio.run(store.save(_checkpoint("C-zeta", "R-001", 1, created_at=T1)))
    asyncio.run(store.save(_checkpoint("C-alpha", "R-001", 1, created_at=T1)))

    latest = asyncio.run(store.latest_for_run("R-001"))
    assert latest is not None
    # max() on checkpoint_id lexicographically: "C-zeta" > "C-alpha"
    assert latest.checkpoint_id == "C-zeta"
