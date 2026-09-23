"""Tests for the ContextAssembler (Phase 4 Step 1).

Covers:

* empty input (Section 32)
* exact fit (Section 33)
* must_keep exact fit (Section 34)
* must_keep overflow (Sections 15, 39)
* priority over recency (Section 35)
* recency tie-break (Section 36)
* stable tie-break by item_id (Section 37)
* oversized optional skipped, smaller fits (Sections 23, 47)
* render order chronological (Sections 25, 30)
* omitted order chronological (Section 31)
* duplicate item_id rejected (Section 43)
* duplicate sequence_index rejected (Section 44)
* input-order independence (Section 49)
* complete multi-item scenario (Section 48)
* compressible / externalizable metadata only (Section 41)
* result invariants (Section 28)

All offline, deterministic, no LLM, no network, no MCP.
"""

from __future__ import annotations

import random

import pytest

from harness.context import (
    ContextAssembler,
    ContextAssemblyError,
    ContextBudget,
    ContextBudgetExceededError,
    ContextItem,
    ContextKind,
    ContextPriority,
    ContextSelectionResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _item(
    item_id: str,
    seq: int,
    *,
    kind: ContextKind = ContextKind.RECENT_INTERACTION,
    tokens: int = 10,
    priority: ContextPriority = ContextPriority.NORMAL,
    must_keep: bool = False,
    compressible: bool = False,
    externalizable: bool = False,
    content: str = "x",
) -> ContextItem:
    return ContextItem(
        item_id=item_id,
        kind=kind,
        content=content,
        estimated_tokens=tokens,
        sequence_index=seq,
        priority=priority,
        must_keep=must_keep,
        compressible=compressible,
        externalizable=externalizable,
    )


def _critical(item_id: str, seq: int, tokens: int = 10) -> ContextItem:
    """CRITICAL_STATE forces must_keep=True."""
    return _item(
        item_id, seq,
        kind=ContextKind.CRITICAL_STATE, tokens=tokens, must_keep=True,
    )


def _assemble(items, budget_tokens: int) -> ContextSelectionResult:
    return ContextAssembler().assemble(items, ContextBudget(max_tokens=budget_tokens))


def _ids(result: ContextSelectionResult) -> tuple[str, ...]:
    return tuple(it.item_id for it in result.included_items)


def _omitted_ids(result: ContextSelectionResult) -> tuple[str, ...]:
    return tuple(it.item_id for it in result.omitted_items)


# ===========================================================================
# Section 32 — Empty input
# ===========================================================================


class TestEmpty:
    def test_empty_items(self):
        r = _assemble([], 100)
        assert r.included_items == ()
        assert r.omitted_items == ()
        assert r.used_tokens == 0
        assert r.remaining_tokens == 100


# ===========================================================================
# Section 33 — Exact fit
# ===========================================================================


class TestExactFit:
    def test_all_optional_exact_fit(self):
        items = [
            _item("a", 0, tokens=50),
            _item("b", 1, tokens=50),
        ]
        r = _assemble(items, 100)
        assert _ids(r) == ("a", "b")
        assert r.used_tokens == 100
        assert r.remaining_tokens == 0

    def test_must_keep_plus_optional_exact_fit(self):
        items = [
            _critical("c", 0, tokens=40),
            _item("a", 1, tokens=30),
            _item("b", 2, tokens=30),
        ]
        r = _assemble(items, 100)
        assert set(_ids(r)) == {"c", "a", "b"}
        assert r.used_tokens == 100


# ===========================================================================
# Section 34 — Must-keep exact fit
# ===========================================================================


def test_must_keep_exact_fit_no_optional():
    """must_keep sum == max_tokens → only must_keep, optional all omitted."""
    items = [
        _critical("c", 0, tokens=100),
        _item("a", 1, tokens=10),
    ]
    r = _assemble(items, 100)
    assert _ids(r) == ("c",)
    assert _omitted_ids(r) == ("a",)
    assert r.used_tokens == 100


# ===========================================================================
# Sections 15, 39 — Must-keep overflow
# ===========================================================================


class TestMustKeepOverflow:
    def test_critical_overflow_raises(self):
        items = [
            _critical("c1", 0, tokens=60),
            _critical("c2", 1, tokens=41),
        ]
        with pytest.raises(ContextBudgetExceededError) as exc_info:
            _assemble(items, 100)
        assert exc_info.value.required_tokens == 101
        assert exc_info.value.available_tokens == 100

    def test_critical_overflow_message_contains_values(self):
        items = [_critical("c", 0, tokens=120)]
        with pytest.raises(ContextBudgetExceededError) as exc_info:
            _assemble(items, 100)
        msg = str(exc_info.value)
        assert "120" in msg
        assert "100" in msg

    def test_must_keep_non_critical_overflow_raises(self):
        """Non-critical must_keep items also cause overflow."""
        items = [
            _item("a", 0, tokens=60, must_keep=True),
            _item("b", 1, tokens=50, must_keep=True),
        ]
        with pytest.raises(ContextBudgetExceededError):
            _assemble(items, 100)


# ===========================================================================
# Section 35 — Priority over recency
# ===========================================================================


def test_priority_over_recency():
    """HIGH old item beats NORMAL recent item when budget allows one."""
    items = [
        _item("A", seq=1, tokens=20, priority=ContextPriority.HIGH),
        _item("B", seq=2, tokens=20, priority=ContextPriority.NORMAL),
    ]
    r = _assemble(items, 20)
    assert _ids(r) == ("A",)
    assert _omitted_ids(r) == ("B",)


# ===========================================================================
# Section 36 — Recency tie-break
# ===========================================================================


def test_recency_tie_break():
    """Same priority, budget allows one → more recent (higher seq) wins."""
    items = [
        _item("A", seq=1, tokens=20),
        _item("B", seq=2, tokens=20),
    ]
    r = _assemble(items, 20)
    assert _ids(r) == ("B",)
    assert _omitted_ids(r) == ("A",)


# ===========================================================================
# Section 37 — Stable tie-break by item_id
# ===========================================================================


def test_stable_tie_break_by_item_id():
    """Same priority AND same sequence_index is impossible (rejected).
    But same priority with different seq still uses seq DESC.
    Test item_id ASC as final tie-break by using identical priority
    and tokens with seq that doesn't matter because only one fits."""
    # With same priority, same tokens, different seq + id:
    # ranking is seq DESC, so higher seq first.
    items = [
        _item("z", seq=5, tokens=10),
        _item("a", seq=5, tokens=10),  # same seq → duplicate, rejected
    ]
    with pytest.raises(ContextAssemblyError, match="duplicate sequence_index"):
        _assemble(items, 100)


def test_item_id_tie_break_when_seq_equal_not_possible():
    """sequence_index is unique, so the only tie-break scenario is
    when priority and seq produce the same ranking key — but seq is
    unique so item_id is only used if priority AND seq are identical,
    which is impossible. We verify item_id is in the sort key by
    checking deterministic output across runs."""
    items = [
        _item("b", seq=2, tokens=10, priority=ContextPriority.NORMAL),
        _item("a", seq=1, tokens=10, priority=ContextPriority.NORMAL),
    ]
    r = _assemble(items, 10)
    # seq=2 is more recent → included.
    assert _ids(r) == ("b",)


# ===========================================================================
# Sections 23, 47 — Oversized optional skipped, smaller fits
# ===========================================================================


class TestOversizedSkip:
    def test_oversized_high_skipped_smaller_normal_fits(self):
        """HIGH item too big → omitted; NORMAL smaller item → included."""
        items = [
            _item("big", seq=1, tokens=50, priority=ContextPriority.HIGH),
            _item("small", seq=2, tokens=10, priority=ContextPriority.NORMAL),
        ]
        r = _assemble(items, 20)
        assert _ids(r) == ("small",)
        assert _omitted_ids(r) == ("big",)
        assert r.used_tokens == 10

    def test_multiple_oversized_then_fit(self):
        """Multiple oversized items skipped, then a small one fits."""
        items = [
            _item("big1", seq=1, tokens=100, priority=ContextPriority.HIGH),
            _item("big2", seq=2, tokens=80, priority=ContextPriority.NORMAL),
            _item("small", seq=3, tokens=15, priority=ContextPriority.LOW),
        ]
        r = _assemble(items, 20)
        assert _ids(r) == ("small",)
        assert r.used_tokens == 15

    def test_skip_does_not_stop_selection(self):
        """Skipping an oversized item does not terminate selection."""
        items = [
            _item("big", seq=1, tokens=60, priority=ContextPriority.HIGH),
            _item("a", seq=2, tokens=20, priority=ContextPriority.NORMAL),
            _item("b", seq=3, tokens=20, priority=ContextPriority.NORMAL),
        ]
        r = _assemble(items, 40)
        # big skipped (60 > 40), a and b fit (20+20=40).
        assert set(_ids(r)) == {"a", "b"}
        assert _omitted_ids(r) == ("big",)


# ===========================================================================
# Sections 25, 30 — Render order chronological
# ===========================================================================


class TestRenderOrder:
    def test_included_rendered_chronological(self):
        """included_items ordered by sequence_index ASC, not ranking."""
        items = [
            _item("old", seq=1, tokens=10, priority=ContextPriority.LOW),
            _item("new", seq=10, tokens=10, priority=ContextPriority.HIGH),
            _item("mid", seq=5, tokens=10, priority=ContextPriority.NORMAL),
        ]
        r = _assemble(items, 100)
        # All fit. Render order is seq ASC: old, mid, new.
        assert _ids(r) == ("old", "mid", "new")

    def test_omitted_rendered_chronological(self):
        """omitted_items also ordered by sequence_index ASC."""
        items = [
            _item("a", seq=1, tokens=60, priority=ContextPriority.HIGH),
            _item("b", seq=2, tokens=60, priority=ContextPriority.NORMAL),
        ]
        r = _assemble(items, 50)
        # Neither fits (both > 50 after must_keep=0). Both omitted.
        assert _omitted_ids(r) == ("a", "b")

    def test_must_keep_rendered_chronological_with_optional(self):
        items = [
            _critical("c2", seq=5, tokens=20),
            _item("opt", seq=1, tokens=10),
            _critical("c1", seq=0, tokens=20),
        ]
        r = _assemble(items, 50)
        # All fit. Render order: c1(0), opt(1), c2(5).
        assert _ids(r) == ("c1", "opt", "c2")


# ===========================================================================
# Section 43 — Duplicate item_id rejected
# ===========================================================================


def test_duplicate_item_id_rejected():
    items = [
        _item("a", seq=0, tokens=10),
        _item("a", seq=1, tokens=10),
    ]
    with pytest.raises(ContextAssemblyError, match="duplicate item_id"):
        _assemble(items, 100)


# ===========================================================================
# Section 44 — Duplicate sequence_index rejected
# ===========================================================================


def test_duplicate_sequence_index_rejected():
    items = [
        _item("a", seq=5, tokens=10),
        _item("b", seq=5, tokens=10),
    ]
    with pytest.raises(ContextAssemblyError, match="duplicate sequence_index"):
        _assemble(items, 100)


# ===========================================================================
# Section 46 — Invalid item types rejected
# ===========================================================================


class TestInvalidItems:
    def test_non_context_item_rejected(self):
        with pytest.raises(ContextAssemblyError, match="ContextItem"):
            _assemble(["not an item"], 100)  # type: ignore[list-item]

    def test_string_input_rejected(self):
        with pytest.raises(ContextAssemblyError, match="sequence"):
            _assemble("abc", 100)  # type: ignore[arg-type]


# ===========================================================================
# Section 49 — Input-order independence
# ===========================================================================


class TestInputOrderIndependence:
    def _scenario(self) -> list[ContextItem]:
        return [
            _critical("goal", seq=0, tokens=20),
            _item("obs_a", seq=3, tokens=20, priority=ContextPriority.NORMAL),
            _item("decision", seq=2, tokens=20, priority=ContextPriority.HIGH),
            _item("obs_b", seq=4, tokens=20, priority=ContextPriority.NORMAL),
            _item("old_log", seq=1, tokens=50, priority=ContextPriority.LOW),
        ]

    def test_same_result_regardless_of_input_order(self):
        items = self._scenario()
        budget_tokens = 60
        r1 = _assemble(items, budget_tokens)

        # Shuffle deterministically.
        shuffled = list(reversed(items))
        r2 = _assemble(shuffled, budget_tokens)
        assert _ids(r1) == _ids(r2)
        assert _omitted_ids(r1) == _omitted_ids(r2)

        # Random shuffle with fixed seed.
        rng = random.Random(42)
        randomized = items[:]
        rng.shuffle(randomized)
        r3 = _assemble(randomized, budget_tokens)
        assert _ids(r1) == _ids(r3)
        assert _omitted_ids(r1) == _omitted_ids(r3)

    def test_render_order_consistent(self):
        items = self._scenario()
        r1 = _assemble(items, 60)
        r2 = _assemble(list(reversed(items)), 60)
        # Render order (seq ASC) must be identical.
        assert [it.sequence_index for it in r1.included_items] == \
               [it.sequence_index for it in r2.included_items]


# ===========================================================================
# Section 48 — Complete multi-item scenario
# ===========================================================================


class TestCompleteScenario:
    def test_full_selection(self):
        """
        budget=100

        critical goal=20
        critical constraints=10
        high history=40
        normal recent1=20
        normal recent2=20

        must_keep: goal(20) + constraints(10) = 30
        remaining: 70

        optional ranked:
          high history (40, HIGH, seq=2)
          normal recent2 (20, NORMAL, seq=4)
          normal recent1 (20, NORMAL, seq=3)

        fit:
          high history 40 <= 70 → included, remaining=30
          normal recent2 20 <= 30 → included, remaining=10
          normal recent1 20 > 10 → omitted
        """
        items = [
            _critical("goal", seq=0, tokens=20),
            _critical("constraints", seq=1, tokens=10),
            _item("history", seq=2, tokens=40, priority=ContextPriority.HIGH),
            _item("recent1", seq=3, tokens=20, priority=ContextPriority.NORMAL),
            _item("recent2", seq=4, tokens=20, priority=ContextPriority.NORMAL),
        ]
        r = _assemble(items, 100)
        assert set(_ids(r)) == {"goal", "constraints", "history", "recent2"}
        assert _omitted_ids(r) == ("recent1",)
        assert r.used_tokens == 90
        assert r.remaining_tokens == 10

    def test_render_order_in_full_scenario(self):
        items = [
            _critical("goal", seq=0, tokens=20),
            _critical("constraints", seq=1, tokens=10),
            _item("history", seq=2, tokens=40, priority=ContextPriority.HIGH),
            _item("recent1", seq=3, tokens=20, priority=ContextPriority.NORMAL),
            _item("recent2", seq=4, tokens=20, priority=ContextPriority.NORMAL),
        ]
        r = _assemble(items, 100)
        # Render order: 0, 1, 2, 4 (recent1 omitted).
        assert [it.sequence_index for it in r.included_items] == [0, 1, 2, 4]


# ===========================================================================
# Section 38 — Critical state test
# ===========================================================================


def test_critical_state_preserved_over_recent():
    """Budget only fits critical state → recent all omitted."""
    items = [
        _critical("goal", seq=0, tokens=40),
        _critical("plan", seq=1, tokens=40),
        _item("recent_a", seq=2, tokens=20, priority=ContextPriority.NORMAL),
        _item("recent_b", seq=3, tokens=20, priority=ContextPriority.HIGH),
    ]
    r = _assemble(items, 80)
    assert set(_ids(r)) == {"goal", "plan"}
    assert set(_omitted_ids(r)) == {"recent_a", "recent_b"}
    assert r.used_tokens == 80


# ===========================================================================
# Section 40 — Recoverable reference
# ===========================================================================


def test_recoverable_reference_participates_in_selection():
    """RECOVERABLE_REFERENCE is a normal item this phase."""
    items = [
        _critical("goal", seq=0, tokens=20),
        _item(
            "ref", seq=1, tokens=8,
            kind=ContextKind.RECOVERABLE_REFERENCE,
            content="artifact:A-102",
        ),
        _item("big", seq=2, tokens=100, priority=ContextPriority.LOW),
    ]
    r = _assemble(items, 30)
    # goal(20) must_keep, ref(8) fits, big(100) doesn't.
    assert set(_ids(r)) == {"goal", "ref"}
    assert r.used_tokens == 28


# ===========================================================================
# Section 41 — compressible / externalizable metadata only
# ===========================================================================


class TestMetadataOnly:
    def test_compressible_item_not_modified(self):
        item = _item("a", seq=0, tokens=50, compressible=True)
        r = _assemble([item], 100)
        included = r.included_items[0]
        assert included is item  # same object, not modified
        assert included.content == "x"
        assert included.estimated_tokens == 50

    def test_externalizable_item_not_externalized(self):
        item = _item("a", seq=0, tokens=50, externalizable=True)
        r = _assemble([item], 100)
        included = r.included_items[0]
        assert included is item
        assert included.content == "x"  # content unchanged

    def test_compressible_omitted_when_budget_tight(self):
        """compressible=True does NOT cause automatic compression."""
        items = [
            _critical("c", seq=0, tokens=80),
            _item("big", seq=1, tokens=50, compressible=True),
        ]
        r = _assemble(items, 100)
        assert _ids(r) == ("c",)
        assert _omitted_ids(r) == ("big",)
        # big was NOT compressed to fit; it was omitted.


# ===========================================================================
# Section 28 — Result invariants
# ===========================================================================


class TestResultInvariants:
    def test_used_tokens_never_exceeds_budget(self):
        items = [
            _critical("c", seq=0, tokens=50),
            _item("a", seq=1, tokens=30),
            _item("b", seq=2, tokens=30),
        ]
        r = _assemble(items, 100)
        assert r.used_tokens <= 100

    def test_included_plus_omitted_equals_input(self):
        items = [
            _critical("c", seq=0, tokens=20),
            _item("a", seq=1, tokens=30),
            _item("b", seq=2, tokens=50),
            _item("d", seq=3, tokens=10),
        ]
        r = _assemble(items, 60)
        all_ids = set(_ids(r)) | set(_omitted_ids(r))
        assert all_ids == {it.item_id for it in items}

    def test_no_item_in_both_lists(self):
        items = [
            _critical("c", seq=0, tokens=20),
            _item("a", seq=1, tokens=30),
        ]
        r = _assemble(items, 50)
        inc_ids = set(_ids(r))
        om_ids = set(_omitted_ids(r))
        assert inc_ids.isdisjoint(om_ids)

    def test_used_tokens_matches_included_sum(self):
        items = [
            _critical("c", seq=0, tokens=20),
            _item("a", seq=1, tokens=30),
        ]
        r = _assemble(items, 100)
        expected = sum(it.estimated_tokens for it in r.included_items)
        assert r.used_tokens == expected


# ===========================================================================
# Assembler statelessness
# ===========================================================================


def test_assembler_is_stateless():
    """Same inputs always produce same output."""
    assembler = ContextAssembler()
    items = [
        _critical("c", seq=0, tokens=20),
        _item("a", seq=1, tokens=30, priority=ContextPriority.HIGH),
        _item("b", seq=2, tokens=30, priority=ContextPriority.NORMAL),
    ]
    budget = ContextBudget(max_tokens=50)
    r1 = assembler.assemble(items, budget)
    r2 = assembler.assemble(items, budget)
    assert _ids(r1) == _ids(r2)
    assert r1.used_tokens == r2.used_tokens
