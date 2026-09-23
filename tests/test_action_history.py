"""Tests for ActionFingerprint, ActionRecord, and ActionHistory (Phase 3 Step 1).

Covers:

* ActionFingerprint stability (Section 39)
* canonical args deterministic / dict ordering irrelevant (Section 7)
* True vs 1 distinct (Section 10)
* unsupported args fail explicitly (Section 9)
* ActionRecord invariants + from_tool_result (Sections 11-13)
* ActionHistory bounded / eviction / snapshot (Section 40)
* invalid max size rejected (Section 18)

All offline, deterministic, no LLM, no network, no MCP.
"""

from __future__ import annotations

import pytest

from harness.loop_detection import (
    ActionFingerprint,
    ActionHistory,
    ActionNormalizationError,
    ActionRecord,
    canonicalize_arguments,
)
from tools.models import ToolCall, ToolError, ToolErrorType, ToolExecutionResult


# ---------------------------------------------------------------------------
# canonicalize_arguments
# ---------------------------------------------------------------------------


class TestCanonicalizeArguments:
    def test_dict_key_ordering_irrelevant(self):
        a = canonicalize_arguments({"path": "a.py", "line": 10})
        b = canonicalize_arguments({"line": 10, "path": "a.py"})
        assert a == b

    def test_nested_dict_key_ordering_irrelevant(self):
        a = canonicalize_arguments({"outer": {"b": 2, "a": 1}})
        b = canonicalize_arguments({"outer": {"a": 1, "b": 2}})
        assert a == b

    def test_list_order_matters(self):
        assert canonicalize_arguments([1, 2]) != canonicalize_arguments([2, 1])

    def test_true_vs_one_distinct(self):
        assert canonicalize_arguments(True) != canonicalize_arguments(1)

    def test_false_vs_zero_distinct(self):
        assert canonicalize_arguments(False) != canonicalize_arguments(0)

    def test_none_distinct(self):
        assert canonicalize_arguments(None) == "null"

    def test_str_int_distinct(self):
        assert canonicalize_arguments("1") != canonicalize_arguments(1)

    def test_float_int_distinct(self):
        assert canonicalize_arguments(1.0) != canonicalize_arguments(1)

    def test_nested_list_and_dict(self):
        a = canonicalize_arguments({"items": [{"k": 1}, {"k": 2}]})
        b = canonicalize_arguments({"items": [{"k": 1}, {"k": 2}]})
        assert a == b

    def test_unsupported_object_rejected(self):
        class Custom:
            pass
        with pytest.raises(ActionNormalizationError):
            canonicalize_arguments(Custom())

    def test_unsupported_nested_object_rejected(self):
        with pytest.raises(ActionNormalizationError):
            canonicalize_arguments({"ok": 1, "bad": object()})

    def test_unsupported_dict_key_type_rejected(self):
        with pytest.raises(ActionNormalizationError):
            canonicalize_arguments({1: "a"})

    def test_nan_rejected(self):
        with pytest.raises(ActionNormalizationError):
            # json.dumps(allow_nan=False) raises ValueError, but our
            # pre-check should also catch it via the json path.
            canonicalize_arguments(float("nan"))


# ---------------------------------------------------------------------------
# ActionFingerprint
# ---------------------------------------------------------------------------


class TestActionFingerprint:
    def _fp(self, tool: str, args: dict) -> ActionFingerprint:
        return ActionFingerprint.from_call(ToolCall(tool_name=tool, arguments=args))

    def test_same_tool_same_args_same_fingerprint(self):
        a = self._fp("read_file", {"path": "a.py"})
        b = self._fp("read_file", {"path": "a.py"})
        assert a == b
        assert hash(a) == hash(b)

    def test_different_dict_ordering_same_fingerprint(self):
        a = self._fp("read_file", {"path": "a.py", "line": 10})
        b = self._fp("read_file", {"line": 10, "path": "a.py"})
        assert a == b

    def test_different_args_different_fingerprint(self):
        a = self._fp("read_file", {"path": "a.py"})
        b = self._fp("read_file", {"path": "b.py"})
        assert a != b

    def test_different_tool_same_args_different_fingerprint(self):
        a = self._fp("read_file", {"path": "a.py"})
        b = self._fp("write_file", {"path": "a.py"})
        assert a != b

    def test_true_vs_one_different_fingerprint(self):
        a = self._fp("tool", {"flag": True})
        b = self._fp("tool", {"flag": 1})
        assert a != b

    def test_empty_tool_name_rejected(self):
        with pytest.raises(ValueError):
            ActionFingerprint(tool_name="", canonical_arguments="{}")

    def test_unsupported_args_rejected_via_from_call(self):
        with pytest.raises(ActionNormalizationError):
            ActionFingerprint.from_call(
                ToolCall(tool_name="t", arguments={"bad": object()})
            )


# ---------------------------------------------------------------------------
# ActionRecord
# ---------------------------------------------------------------------------


class TestActionRecord:
    def _fp(self) -> ActionFingerprint:
        return ActionFingerprint(tool_name="t", canonical_arguments="{}")

    def test_success_record(self):
        r = ActionRecord(
            action_index=0,
            fingerprint=self._fp(),
            success=True,
            error_type=None,
            progress_token="v1",
        )
        assert r.success is True
        assert r.error_type is None
        assert r.progress_token == "v1"

    def test_failure_record(self):
        r = ActionRecord(
            action_index=1,
            fingerprint=self._fp(),
            success=False,
            error_type=ToolErrorType.PERMANENT,
            progress_token=None,
        )
        assert r.success is False
        assert r.error_type is ToolErrorType.PERMANENT

    def test_negative_action_index_rejected(self):
        with pytest.raises(ValueError):
            ActionRecord(action_index=-1, fingerprint=self._fp(), success=True)

    def test_success_with_error_type_rejected(self):
        with pytest.raises(ValueError):
            ActionRecord(
                action_index=0,
                fingerprint=self._fp(),
                success=True,
                error_type=ToolErrorType.EXECUTION,
            )

    def test_failure_without_error_type_rejected(self):
        """Phase 3 Step 2: success=False must carry an error_type."""
        with pytest.raises(ValueError):
            ActionRecord(
                action_index=0,
                fingerprint=self._fp(),
                success=False,
                error_type=None,
            )

    def test_non_int_action_index_rejected(self):
        with pytest.raises(ValueError):
            ActionRecord(action_index=1.0, fingerprint=self._fp(), success=True)

    def test_non_str_progress_token_rejected(self):
        with pytest.raises(ValueError):
            ActionRecord(
                action_index=0, fingerprint=self._fp(), success=True,
                progress_token=123,
            )

    def test_from_tool_result_success(self):
        call = ToolCall(tool_name="read_file", arguments={"path": "a.py"})
        result = ToolExecutionResult(
            tool_name="read_file", success=True, output="content", error=None,
        )
        record = ActionRecord.from_tool_result(
            action_index=5, call=call, result=result, progress_token="v1",
        )
        assert record.action_index == 5
        assert record.success is True
        assert record.error_type is None
        assert record.progress_token == "v1"
        assert record.fingerprint.tool_name == "read_file"

    def test_from_tool_result_failure(self):
        call = ToolCall(tool_name="run_tests", arguments={"suite": "x"})
        result = ToolExecutionResult(
            tool_name="run_tests",
            success=False,
            error=ToolError(
                error_type=ToolErrorType.PERMANENT,
                message="fail",
                retryable=False,
            ),
        )
        record = ActionRecord.from_tool_result(
            action_index=3, call=call, result=result,
        )
        assert record.success is False
        assert record.error_type is ToolErrorType.PERMANENT
        assert record.progress_token is None

    def test_from_tool_result_does_not_carry_output(self):
        """Loop detection should not copy tool output / retry history."""
        call = ToolCall(tool_name="t", arguments={})
        result = ToolExecutionResult(
            tool_name="t", success=True, output="huge output blob",
            error=None, attempt_count=3,
            retry_history=(
                ToolError(ToolErrorType.TRANSIENT, "x", True),
                ToolError(ToolErrorType.TRANSIENT, "y", True),
            ),
        )
        record = ActionRecord.from_tool_result(
            action_index=0, call=call, result=result,
        )
        # ActionRecord has no output / retry_history fields.
        assert not hasattr(record, "output")
        assert not hasattr(record, "retry_history")


# ---------------------------------------------------------------------------
# ActionHistory
# ---------------------------------------------------------------------------


class TestActionHistory:
    def _fp(self, i: int = 0) -> ActionFingerprint:
        return ActionFingerprint(tool_name="t", canonical_arguments=str(i))

    def _rec(self, i: int = 0) -> ActionRecord:
        return ActionRecord(
            action_index=i, fingerprint=self._fp(i), success=True,
        )

    def test_append_and_len(self):
        h = ActionHistory(max_size=4)
        assert len(h) == 0
        h.append(self._rec(0))
        h.append(self._rec(1))
        assert len(h) == 2

    def test_bounded_max_size_evicts_oldest(self):
        h = ActionHistory(max_size=3)
        for i in range(5):
            h.append(self._rec(i))
        assert len(h) == 3
        recs = h.records()
        # Oldest two (0, 1) evicted; (2, 3, 4) remain.
        assert [r.action_index for r in recs] == [2, 3, 4]

    def test_records_returns_snapshot_tuple(self):
        h = ActionHistory(max_size=4)
        h.append(self._rec(0))
        h.append(self._rec(1))
        snap = h.records()
        assert isinstance(snap, tuple)
        # Mutating the snapshot does not affect the history.
        snap  # noqa: B018 — tuple is immutable anyway
        assert len(h) == 2

    def test_recent_n(self):
        h = ActionHistory(max_size=10)
        for i in range(5):
            h.append(self._rec(i))
        recent = h.recent(3)
        assert [r.action_index for r in recent] == [2, 3, 4]

    def test_recent_fewer_than_n(self):
        h = ActionHistory(max_size=10)
        h.append(self._rec(0))
        h.append(self._rec(1))
        assert len(h.recent(5)) == 2

    def test_recent_zero(self):
        h = ActionHistory(max_size=10)
        h.append(self._rec(0))
        assert h.recent(0) == ()

    def test_recent_negative_rejected(self):
        h = ActionHistory(max_size=10)
        with pytest.raises(ValueError):
            h.recent(-1)

    def test_invalid_max_size_zero_rejected(self):
        with pytest.raises(ValueError):
            ActionHistory(max_size=0)

    def test_invalid_max_size_negative_rejected(self):
        with pytest.raises(ValueError):
            ActionHistory(max_size=-1)

    def test_append_non_record_rejected(self):
        h = ActionHistory(max_size=4)
        with pytest.raises(ValueError):
            h.append("not a record")  # type: ignore[arg-type]

    def test_records_snapshot_isolated_from_internal_deque(self):
        h = ActionHistory(max_size=4)
        h.append(self._rec(0))
        snap = h.records()
        # Even though tuple is immutable, ensure the record objects
        # inside are the same frozen dataclass instances.
        assert snap[0].action_index == 0
        # Appending more does not change the prior snapshot.
        h.append(self._rec(1))
        assert len(snap) == 1
        assert len(h) == 2

    # --- Phase 3 Step 2: monotonic index enforcement ---

    def test_monotonic_index_accepted(self):
        """Strictly increasing indices are accepted."""
        h = ActionHistory(max_size=10)
        h.append(self._rec(5))
        h.append(self._rec(6))
        h.append(self._rec(7))
        assert [r.action_index for r in h.records()] == [5, 6, 7]

    def test_non_monotonic_index_rejected(self):
        """An index <= the last appended index is rejected."""
        h = ActionHistory(max_size=10)
        h.append(self._rec(5))
        h.append(self._rec(6))
        h.append(self._rec(7))
        with pytest.raises(ValueError, match="strictly greater"):
            h.append(self._rec(6))  # less than 7

    def test_equal_index_rejected(self):
        """Same index as last appended is rejected."""
        h = ActionHistory(max_size=10)
        h.append(self._rec(7))
        with pytest.raises(ValueError, match="strictly greater"):
            h.append(self._rec(7))

    def test_first_append_with_index_zero_accepted(self):
        """First append with index 0 is accepted (last defaults to -1)."""
        h = ActionHistory(max_size=10)
        h.append(self._rec(0))
        assert len(h) == 1

    def test_first_append_with_negative_index_rejected_by_record(self):
        """ActionRecord itself rejects negative index before history."""
        with pytest.raises(ValueError):
            self._rec(-1)
