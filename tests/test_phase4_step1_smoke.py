"""Smoke test for Phase 4 Step 1 — context budget & deterministic assembly.

Simulates a long-horizon context:

    Task goal              CRITICAL   20  must_keep
    Task constraints       CRITICAL   10  must_keep
    Old pytest output      HISTORICAL LOW  50
    Old reasoning          HISTORICAL LOW  20
    Recent Tool obs A      RECENT     NORMAL 20
    Recent Tool obs B      RECENT     NORMAL 20
    Important decision     HISTORICAL HIGH  20

budget = 70

Expected:
    goal + constraints always kept (30 tokens)
    remaining = 40
    optional ranked:
      decision (HIGH, seq=5)   20 → fits, remaining=20
      obs B (NORMAL, seq=4)    20 → fits, remaining=0
      obs A (NORMAL, seq=3)    20 → omitted (0 remaining)
      old reasoning (LOW, seq=2) 20 → omitted
      old pytest (LOW, seq=1)  50 → omitted

Final included (render order by seq ASC):
    goal(0), constraints(1), obs B(4)... wait, decision is seq=5.

Let me recompute with explicit sequence indices:
    goal           seq=0  CRITICAL  20  must_keep
    constraints    seq=1  CRITICAL  10  must_keep
    old_pytest     seq=2  HISTORICAL LOW  50
    old_reasoning  seq=3  HISTORICAL LOW  20
    obs_A          seq=4  RECENT    NORMAL 20
    obs_B          seq=5  RECENT    NORMAL 20
    decision       seq=6  HISTORICAL HIGH  20

must_keep: goal(20) + constraints(10) = 30
remaining: 70 - 30 = 40

optional ranked (priority DESC, seq DESC, id ASC):
  decision   HIGH  seq=6  20 → fits (40), remaining=20
  obs_B      NORMAL seq=5 20 → fits (20), remaining=0
  obs_A      NORMAL seq=4 20 → omitted (0 remaining)
  old_reasoning LOW seq=3 20 → omitted
  old_pytest    LOW seq=2 50 → omitted

included (render order seq ASC): goal, constraints, obs_B, decision
used_tokens = 20 + 10 + 20 + 20 = 70

Fully offline, deterministic, no LLM, no network, no MCP.
"""

from __future__ import annotations

from harness.context import (
    ContextAssembler,
    ContextBudget,
    ContextItem,
    ContextKind,
    ContextPriority,
    ContextSelectionResult,
)


def test_phase4_step1_smoke():
    items = [
        ContextItem(
            item_id="goal",
            kind=ContextKind.CRITICAL_STATE,
            content="Fix the failing tests in module X.",
            estimated_tokens=20,
            sequence_index=0,
            must_keep=True,
        ),
        ContextItem(
            item_id="constraints",
            kind=ContextKind.CRITICAL_STATE,
            content="Do not modify public API.",
            estimated_tokens=10,
            sequence_index=1,
            must_keep=True,
        ),
        ContextItem(
            item_id="old_pytest",
            kind=ContextKind.HISTORICAL_EVIDENCE,
            content="... 5000 lines of old pytest output ...",
            estimated_tokens=50,
            sequence_index=2,
            priority=ContextPriority.LOW,
        ),
        ContextItem(
            item_id="old_reasoning",
            kind=ContextKind.HISTORICAL_EVIDENCE,
            content="Earlier I thought the bug was in Y.",
            estimated_tokens=20,
            sequence_index=3,
            priority=ContextPriority.LOW,
        ),
        ContextItem(
            item_id="obs_A",
            kind=ContextKind.RECENT_INTERACTION,
            content="Tool observed state A.",
            estimated_tokens=20,
            sequence_index=4,
            priority=ContextPriority.NORMAL,
        ),
        ContextItem(
            item_id="obs_B",
            kind=ContextKind.RECENT_INTERACTION,
            content="Tool observed state B.",
            estimated_tokens=20,
            sequence_index=5,
            priority=ContextPriority.NORMAL,
        ),
        ContextItem(
            item_id="decision",
            kind=ContextKind.HISTORICAL_EVIDENCE,
            content="Decided to refactor module Z first.",
            estimated_tokens=20,
            sequence_index=6,
            priority=ContextPriority.HIGH,
        ),
    ]

    budget = ContextBudget(max_tokens=70)
    result = ContextAssembler().assemble(items, budget)

    assert isinstance(result, ContextSelectionResult)

    # Critical state always kept.
    included_ids = [it.item_id for it in result.included_items]
    omitted_ids = [it.item_id for it in result.omitted_items]
    assert "goal" in included_ids
    assert "constraints" in included_ids

    # Important decision (HIGH) beats low historical output.
    assert "decision" in included_ids
    assert "old_pytest" in omitted_ids
    assert "old_reasoning" in omitted_ids

    # Same priority (NORMAL): more recent obs_B beats obs_A.
    assert "obs_B" in included_ids
    assert "obs_A" in omitted_ids

    # Budget respected.
    assert result.used_tokens <= 70
    assert result.used_tokens == 70  # exact fit: 20+10+20+20

    # Render order is chronological (sequence_index ASC).
    seqs = [it.sequence_index for it in result.included_items]
    assert seqs == sorted(seqs)
    assert seqs == [0, 1, 5, 6]  # goal, constraints, obs_B, decision

    # Omitted also chronological.
    om_seqs = [it.sequence_index for it in result.omitted_items]
    assert om_seqs == sorted(om_seqs)

    # All items accounted for.
    assert set(included_ids) | set(omitted_ids) == {
        it.item_id for it in items
    }
    assert set(included_ids).isdisjoint(set(omitted_ids))
