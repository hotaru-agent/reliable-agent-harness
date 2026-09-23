"""Tests for benchmark execution records (Phase 7 Step 1).

Covers:
* BenchmarkEvent validation
* BenchmarkRunRecord validation and field semantics
* BenchmarkResult derivation from record
* Record snapshots not externally mutable
* Logical action count vs tool attempt count vs retry count

All offline, deterministic.
"""

from __future__ import annotations

import pytest

from evaluation.records import (
    BenchmarkEvent,
    BenchmarkEventType,
    BenchmarkResult,
    BenchmarkRunRecord,
)


# ===========================================================================
# BenchmarkEvent validation
# ===========================================================================


class TestBenchmarkEventValidation:
    def test_valid_event(self):
        ev = BenchmarkEvent(
            event_type=BenchmarkEventType.LOGICAL_ACTION,
            sequence=0,
            tool_name="run_tests",
            success=True,
        )
        assert ev.event_type is BenchmarkEventType.LOGICAL_ACTION
        assert ev.sequence == 0

    def test_negative_sequence_rejected(self):
        with pytest.raises(ValueError, match="sequence"):
            BenchmarkEvent(
                event_type=BenchmarkEventType.LOGICAL_ACTION,
                sequence=-1,
            )

    def test_empty_tool_name_rejected(self):
        with pytest.raises(ValueError, match="tool_name"):
            BenchmarkEvent(
                event_type=BenchmarkEventType.TOOL_INVOCATION,
                sequence=0,
                tool_name="",
            )

    def test_none_tool_name_allowed(self):
        ev = BenchmarkEvent(
            event_type=BenchmarkEventType.CHECKPOINT,
            sequence=0,
            tool_name=None,
        )
        assert ev.tool_name is None

    def test_non_bool_success_rejected(self):
        with pytest.raises(ValueError, match="success"):
            BenchmarkEvent(
                event_type=BenchmarkEventType.LOGICAL_ACTION,
                sequence=0,
                success="yes",  # type: ignore[arg-type]
            )

    def test_event_is_frozen(self):
        ev = BenchmarkEvent(
            event_type=BenchmarkEventType.LOGICAL_ACTION,
            sequence=0,
        )
        with pytest.raises(Exception):
            ev.sequence = 1  # type: ignore[misc]


# ===========================================================================
# BenchmarkRunRecord validation
# ===========================================================================


class TestBenchmarkRunRecordValidation:
    def test_valid_record(self):
        record = BenchmarkRunRecord(
            scenario_id="clean_success",
            task_id="fix_calculator_add",
            completed=True,
            logical_action_count=4,
        )
        assert record.scenario_id == "clean_success"
        assert record.completed is True
        assert record.logical_action_count == 4

    def test_empty_scenario_id_rejected(self):
        with pytest.raises(ValueError, match="scenario_id"):
            BenchmarkRunRecord(scenario_id="", task_id="t1")

    def test_empty_task_id_rejected(self):
        with pytest.raises(ValueError, match="task_id"):
            BenchmarkRunRecord(scenario_id="s1", task_id="")

    def test_negative_count_rejected(self):
        with pytest.raises(ValueError, match="logical_action_count"):
            BenchmarkRunRecord(
                scenario_id="s1", task_id="t1", logical_action_count=-1
            )

    def test_negative_retry_count_rejected(self):
        with pytest.raises(ValueError, match="retry_count"):
            BenchmarkRunRecord(
                scenario_id="s1", task_id="t1", retry_count=-1
            )

    def test_negative_wall_clock_rejected(self):
        with pytest.raises(ValueError, match="wall_clock_seconds"):
            BenchmarkRunRecord(
                scenario_id="s1", task_id="t1", wall_clock_seconds=-1.0
            )

    def test_bool_count_rejected(self):
        with pytest.raises(ValueError, match="logical_action_count"):
            BenchmarkRunRecord(
                scenario_id="s1", task_id="t1", logical_action_count=True
            )


# ===========================================================================
# Logical action count vs tool attempt count vs retry count
# ===========================================================================


class TestCountSemantics:
    def test_one_logical_action_two_attempts_one_retry(self):
        """One logical action with a transient failure:
        - logical_action_count = 1
        - tool_attempt_count = 2
        - retry_count = 1
        """
        record = BenchmarkRunRecord(
            scenario_id="s1",
            task_id="t1",
            completed=True,
            logical_action_count=1,
            tool_invocation_count=1,
            tool_attempt_count=2,
            retry_count=1,
        )
        assert record.logical_action_count == 1
        assert record.tool_attempt_count == 2
        assert record.retry_count == 1
        # retry_count = attempt_count - invocation_count
        assert record.retry_count == record.tool_attempt_count - record.tool_invocation_count

    def test_duplicate_action_count_semantics(self):
        """Duplicate logical actions (same tool + same args) are counted
        separately from retry attempts."""
        record = BenchmarkRunRecord(
            scenario_id="s1",
            task_id="t1",
            completed=True,
            logical_action_count=5,
            tool_invocation_count=5,
            tool_attempt_count=5,
            retry_count=0,
            duplicate_action_count=2,
        )
        assert record.duplicate_action_count == 2
        assert record.retry_count == 0


# ===========================================================================
# BenchmarkResult derivation
# ===========================================================================


class TestBenchmarkResult:
    def test_from_completed_record(self):
        record = BenchmarkRunRecord(
            scenario_id="s1",
            task_id="t1",
            completed=True,
            logical_action_count=4,
        )
        result = BenchmarkResult.from_record(record)
        assert result.task_completed is True
        assert result.steps_to_completion == 4
        assert result.failure_reason is None

    def test_from_incomplete_record(self):
        record = BenchmarkRunRecord(
            scenario_id="s1",
            task_id="t1",
            completed=False,
            logical_action_count=3,
            failure_reason="max actions exceeded",
        )
        result = BenchmarkResult.from_record(record)
        assert result.task_completed is False
        assert result.steps_to_completion is None
        assert result.failure_reason == "max actions exceeded"

    def test_result_is_frozen(self):
        result = BenchmarkResult(
            scenario_id="s1",
            task_completed=True,
            steps_to_completion=4,
        )
        with pytest.raises(Exception):
            result.task_completed = False  # type: ignore[misc]

    def test_empty_scenario_id_rejected(self):
        with pytest.raises(ValueError, match="scenario_id"):
            BenchmarkResult(scenario_id="", task_completed=True)

    def test_negative_steps_rejected(self):
        with pytest.raises(ValueError, match="steps_to_completion"):
            BenchmarkResult(
                scenario_id="s1",
                task_completed=True,
                steps_to_completion=-1,
            )


# ===========================================================================
# Record snapshots not externally mutable
# ===========================================================================


class TestRecordImmutability:
    def test_to_dict_does_not_mutate_record(self):
        record = BenchmarkRunRecord(
            scenario_id="s1",
            task_id="t1",
            completed=True,
            logical_action_count=4,
        )
        snapshot = record.to_dict()
        snapshot["logical_action_count"] = 999
        assert record.logical_action_count == 4

    def test_events_tuple_immutable(self):
        events = (
            BenchmarkEvent(event_type=BenchmarkEventType.LOGICAL_ACTION, sequence=0),
            BenchmarkEvent(event_type=BenchmarkEventType.TOOL_INVOCATION, sequence=1),
        )
        record = BenchmarkRunRecord(
            scenario_id="s1",
            task_id="t1",
            events=events,
        )
        assert record.events == events
        # Cannot assign to events (dataclass is not frozen, but tuple is immutable).
        with pytest.raises(TypeError):
            record.events[0] = events[1]  # type: ignore[index]


# ===========================================================================
# Completion based on oracle, not self-declaration
# ===========================================================================


class TestCompletionSemantics:
    def test_completion_is_oracle_based(self):
        """The 'completed' field represents oracle verdict, not agent
        self-declaration. A run where the agent claims completion but
        the oracle disagrees must have completed=False."""
        record = BenchmarkRunRecord(
            scenario_id="s1",
            task_id="t1",
            completed=False,  # oracle says incomplete
            logical_action_count=10,
            failure_reason="oracle: tests still failing",
        )
        assert record.completed is False
        assert record.failure_reason is not None


# ===========================================================================
# Wall-clock field exists but is not asserted
# ===========================================================================


class TestWallClockField:
    def test_wall_clock_field_exists(self):
        record = BenchmarkRunRecord(
            scenario_id="s1",
            task_id="t1",
            wall_clock_seconds=1.23,
        )
        assert record.wall_clock_seconds == 1.23

    def test_wall_clock_default_zero(self):
        record = BenchmarkRunRecord(scenario_id="s1", task_id="t1")
        assert record.wall_clock_seconds == 0.0

    def test_wall_clock_not_used_for_assertions(self):
        """Wall-clock is recorded but NOT used for deterministic
        unit-test assertions. This test just verifies the field exists
        and is non-negative."""
        record = BenchmarkRunRecord(
            scenario_id="s1",
            task_id="t1",
            wall_clock_seconds=0.0,
        )
        assert record.wall_clock_seconds >= 0
