"""Tests for context model invariants (Phase 4 Step 1).

Covers:

* ContextKind enum values
* ContextPriority ordering
* ContextItem field validation (Section 10-11)
* CRITICAL_STATE must_keep invariant (Section 12)
* ContextBudget validation (Section 13)
* ContextSelectionResult invariants (Sections 27-31)

All offline, deterministic, no LLM, no network, no MCP.
"""

from __future__ import annotations

import pytest

from harness.context import (
    ContextAssemblyError,
    ContextBudget,
    ContextBudgetExceededError,
    ContextItem,
    ContextKind,
    ContextPriority,
    ContextSelectionResult,
)


# ===========================================================================
# ContextKind
# ===========================================================================


class TestContextKind:
    def test_enum_values(self):
        assert ContextKind.CRITICAL_STATE.value == "critical_state"
        assert ContextKind.RECENT_INTERACTION.value == "recent_interaction"
        assert ContextKind.HISTORICAL_EVIDENCE.value == "historical_evidence"
        assert ContextKind.RECOVERABLE_REFERENCE.value == "recoverable_reference"

    def test_all_kinds_distinct(self):
        kinds = {k for k in ContextKind}
        assert len(kinds) == 4


# ===========================================================================
# ContextPriority
# ===========================================================================


class TestContextPriority:
    def test_ordering(self):
        assert ContextPriority.LOW < ContextPriority.NORMAL
        assert ContextPriority.NORMAL < ContextPriority.HIGH
        assert ContextPriority.HIGH < ContextPriority.CRITICAL

    def test_int_values(self):
        assert int(ContextPriority.LOW) == 10
        assert int(ContextPriority.NORMAL) == 20
        assert int(ContextPriority.HIGH) == 30
        assert int(ContextPriority.CRITICAL) == 40

    def test_sortable(self):
        priorities = [
            ContextPriority.NORMAL,
            ContextPriority.LOW,
            ContextPriority.CRITICAL,
            ContextPriority.HIGH,
        ]
        ordered = sorted(priorities)
        assert ordered == [
            ContextPriority.LOW,
            ContextPriority.NORMAL,
            ContextPriority.HIGH,
            ContextPriority.CRITICAL,
        ]


# ===========================================================================
# ContextItem invariants
# ===========================================================================


def _valid_item(**overrides) -> ContextItem:
    defaults = dict(
        item_id="i1",
        kind=ContextKind.RECENT_INTERACTION,
        content="hello",
        estimated_tokens=10,
        sequence_index=0,
    )
    defaults.update(overrides)
    return ContextItem(**defaults)


class TestContextItem:
    def test_valid_construction(self):
        item = _valid_item()
        assert item.item_id == "i1"
        assert item.kind is ContextKind.RECENT_INTERACTION
        assert item.content == "hello"
        assert item.estimated_tokens == 10
        assert item.sequence_index == 0
        assert item.priority is ContextPriority.NORMAL
        assert item.must_keep is False
        assert item.compressible is False
        assert item.externalizable is False

    def test_empty_item_id_rejected(self):
        with pytest.raises(ValueError, match="item_id"):
            _valid_item(item_id="")

    def test_non_str_item_id_rejected(self):
        with pytest.raises(ValueError, match="item_id"):
            _valid_item(item_id=123)  # type: ignore[arg-type]

    def test_non_str_content_rejected(self):
        with pytest.raises(ValueError, match="content"):
            _valid_item(content=123)  # type: ignore[arg-type]

    def test_negative_estimated_tokens_rejected(self):
        with pytest.raises(ValueError, match="estimated_tokens"):
            _valid_item(estimated_tokens=-1)

    def test_zero_estimated_tokens_accepted(self):
        """Zero-token items (empty content markers) are allowed."""
        item = _valid_item(estimated_tokens=0, content="")
        assert item.estimated_tokens == 0
        assert item.content == ""

    def test_zero_tokens_with_non_empty_content_rejected(self):
        """Phase 4 Step 2: non-empty content must have estimated_tokens >= 1."""
        with pytest.raises(ValueError, match="estimated_tokens=0"):
            _valid_item(estimated_tokens=0, content="hello")

    def test_zero_tokens_empty_content_accepted(self):
        """Empty content with 0 tokens is a valid empty marker."""
        item = _valid_item(estimated_tokens=0, content="")
        assert item.estimated_tokens == 0
        assert item.content == ""

    def test_bool_estimated_tokens_rejected(self):
        """bool is a subclass of int but should not be accepted."""
        with pytest.raises(ValueError, match="estimated_tokens"):
            _valid_item(estimated_tokens=True)  # type: ignore[arg-type]

    def test_negative_sequence_index_rejected(self):
        with pytest.raises(ValueError, match="sequence_index"):
            _valid_item(sequence_index=-1)

    def test_zero_sequence_index_accepted(self):
        item = _valid_item(sequence_index=0)
        assert item.sequence_index == 0

    def test_bool_sequence_index_rejected(self):
        with pytest.raises(ValueError, match="sequence_index"):
            _valid_item(sequence_index=True)  # type: ignore[arg-type]

    def test_non_enum_kind_rejected(self):
        with pytest.raises(ValueError, match="kind"):
            _valid_item(kind="critical_state")  # type: ignore[arg-type]

    def test_non_enum_priority_rejected(self):
        with pytest.raises(ValueError, match="priority"):
            _valid_item(priority=20)  # type: ignore[arg-type]

    def test_non_bool_must_keep_rejected(self):
        with pytest.raises(ValueError, match="must_keep"):
            _valid_item(must_keep=1)  # type: ignore[arg-type]

    def test_non_bool_compressible_rejected(self):
        with pytest.raises(ValueError, match="compressible"):
            _valid_item(compressible=1)  # type: ignore[arg-type]

    def test_non_bool_externalizable_rejected(self):
        with pytest.raises(ValueError, match="externalizable"):
            _valid_item(externalizable=1)  # type: ignore[arg-type]

    def test_frozen(self):
        item = _valid_item()
        with pytest.raises(Exception):
            item.item_id = "other"  # type: ignore[misc]

    # --- CRITICAL_STATE must_keep invariant (Section 12) ---

    def test_critical_state_with_must_keep_accepted(self):
        item = _valid_item(
            kind=ContextKind.CRITICAL_STATE, must_keep=True,
        )
        assert item.must_keep is True

    def test_critical_state_without_must_keep_rejected(self):
        with pytest.raises(ValueError, match="CRITICAL_STATE"):
            _valid_item(
                kind=ContextKind.CRITICAL_STATE, must_keep=False,
            )

    def test_critical_state_default_must_keep_still_rejected(self):
        """Default must_keep=False is rejected for CRITICAL_STATE."""
        with pytest.raises(ValueError, match="CRITICAL_STATE"):
            _valid_item(kind=ContextKind.CRITICAL_STATE)

    def test_non_critical_state_with_must_keep_accepted(self):
        """Non-critical kinds can optionally be must_keep."""
        item = _valid_item(
            kind=ContextKind.RECENT_INTERACTION, must_keep=True,
        )
        assert item.must_keep is True

    def test_non_critical_state_without_must_keep_accepted(self):
        item = _valid_item(
            kind=ContextKind.HISTORICAL_EVIDENCE, must_keep=False,
        )
        assert item.must_keep is False

    # --- compressible / externalizable metadata (Section 41) ---

    def test_compressible_metadata_only(self):
        item = _valid_item(compressible=True)
        assert item.compressible is True
        # The field is just metadata; no compression happens.

    def test_externalizable_metadata_only(self):
        item = _valid_item(externalizable=True)
        assert item.externalizable is True
        # The field is just metadata; no externalization happens.

    def test_recoverable_reference_kind(self):
        """RECOVERABLE_REFERENCE is a normal kind this phase."""
        item = _valid_item(
            kind=ContextKind.RECOVERABLE_REFERENCE,
            content="artifact:A-102",
            estimated_tokens=8,
        )
        assert item.kind is ContextKind.RECOVERABLE_REFERENCE


# ===========================================================================
# ContextBudget
# ===========================================================================


class TestContextBudget:
    def test_valid_budget(self):
        b = ContextBudget(max_tokens=100)
        assert b.max_tokens == 100

    def test_zero_budget_rejected(self):
        with pytest.raises(ValueError, match="max_tokens"):
            ContextBudget(max_tokens=0)

    def test_negative_budget_rejected(self):
        with pytest.raises(ValueError, match="max_tokens"):
            ContextBudget(max_tokens=-1)

    def test_bool_budget_rejected(self):
        with pytest.raises(ValueError, match="max_tokens"):
            ContextBudget(max_tokens=True)  # type: ignore[arg-type]

    def test_frozen(self):
        b = ContextBudget(max_tokens=100)
        with pytest.raises(Exception):
            b.max_tokens = 200  # type: ignore[misc]


# ===========================================================================
# ContextSelectionResult
# ===========================================================================


class TestContextSelectionResult:
    def _item(self, item_id: str, seq: int, tokens: int = 10) -> ContextItem:
        return ContextItem(
            item_id=item_id,
            kind=ContextKind.RECENT_INTERACTION,
            content="x",
            estimated_tokens=tokens,
            sequence_index=seq,
        )

    def test_valid_result(self):
        inc = (self._item("a", 0), self._item("b", 1))
        om = (self._item("c", 2),)
        r = ContextSelectionResult(
            included_items=inc,
            omitted_items=om,
            used_tokens=20,
            max_tokens=100,
        )
        assert r.remaining_tokens == 80

    def test_used_tokens_exceeding_max_rejected(self):
        with pytest.raises(ValueError, match="used_tokens"):
            ContextSelectionResult(
                included_items=(),
                omitted_items=(),
                used_tokens=101,
                max_tokens=100,
            )

    def test_negative_used_tokens_rejected(self):
        with pytest.raises(ValueError, match="used_tokens"):
            ContextSelectionResult(
                included_items=(),
                omitted_items=(),
                used_tokens=-1,
                max_tokens=100,
            )

    def test_zero_max_tokens_rejected(self):
        with pytest.raises(ValueError, match="max_tokens"):
            ContextSelectionResult(
                included_items=(),
                omitted_items=(),
                used_tokens=0,
                max_tokens=0,
            )

    def test_duplicate_included_ids_rejected(self):
        with pytest.raises(ValueError, match="duplicate item_id"):
            ContextSelectionResult(
                included_items=(self._item("a", 0), self._item("a", 1)),
                omitted_items=(),
                used_tokens=20,
                max_tokens=100,
            )

    def test_non_tuple_included_rejected(self):
        with pytest.raises(ValueError, match="included_items"):
            ContextSelectionResult(
                included_items=[self._item("a", 0)],  # type: ignore[arg-type]
                omitted_items=(),
                used_tokens=10,
                max_tokens=100,
            )

    def test_remaining_tokens_property(self):
        r = ContextSelectionResult(
            included_items=(self._item("a", 0, tokens=30),),
            omitted_items=(),
            used_tokens=30,
            max_tokens=100,
        )
        assert r.remaining_tokens == 70


# ===========================================================================
# ContextBudgetExceededError
# ===========================================================================


class TestContextBudgetExceededError:
    def test_carries_required_and_available(self):
        err = ContextBudgetExceededError(
            required_tokens=120, available_tokens=100,
        )
        assert err.required_tokens == 120
        assert err.available_tokens == 100
        assert "120" in str(err)
        assert "100" in str(err)


# ===========================================================================
# ContextAssemblyError
# ===========================================================================


def test_context_assembly_error_is_exception():
    assert issubclass(ContextAssemblyError, Exception)
