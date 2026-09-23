"""Tests for the deterministic LoopDetector (Phase 3 Step 1).

Covers:

* LoopDetectorConfig validation (Section 23)
* LoopDetectionResult invariants (Section 22)
* Duplicate detection (Sections 25, 41-44)
* Duplicate with progress does NOT trigger (Sections 26, 43)
* Unknown progress does NOT trigger duplicate (Section 27)
* Repeating sequence AB / ABC (Sections 28-29, 45-46)
* Partial cycle does NOT trigger (Section 30, 47)
* Sequence with progress does NOT trigger (Section 31, 48)
* No-progress detection (Sections 32-33, 49)
* No-progress below window (Section 50)
* No-progress reset on token change (Section 51)
* Unknown progress does NOT trigger no-progress (Section 52)
* Detection priority DUPLICATE > SEQUENCE > NO_PROGRESS (Section 53)
* Failed actions participate (Section 34, 54)
* Retry attempts do NOT inflate history (Sections 35, 55)
* Detector does not change Task/Run state (Section 37)
* False result is structured, not None (Section 38)

All offline, deterministic, no LLM, no network, no MCP.
"""

from __future__ import annotations

import asyncio

import pytest

from harness.loop_detection import (
    ActionFingerprint,
    ActionHistory,
    ActionRecord,
    LoopDetectionReason,
    LoopDetectionResult,
    LoopDetector,
    LoopDetectorConfig,
)
from tests.fake_tools import ScriptedTool
from tools import (
    RetryPolicy,
    ToolCall,
    ToolErrorType,
    ToolExecutionContext,
    ToolRegistry,
    ToolRuntime,
    ToolSideEffect,
    ToolSpec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fp(tool: str = "t", args: dict | None = None) -> ActionFingerprint:
    return ActionFingerprint.from_call(
        ToolCall(tool_name=tool, arguments=args or {})
    )


def _rec(
    idx: int,
    fp: ActionFingerprint | None = None,
    *,
    success: bool = True,
    error_type: ToolErrorType | None = None,
    progress: str | None = None,
) -> ActionRecord:
    return ActionRecord(
        action_index=idx,
        fingerprint=fp or _fp(),
        success=success,
        error_type=error_type,
        progress_token=progress,
    )


def _observe_all(
    detector: LoopDetector, records: list[ActionRecord]
) -> LoopDetectionResult:
    """Observe all records, return the final result."""
    result = LoopDetectionResult(detected=False)
    for r in records:
        result = detector.observe(r)
    return result


# ===========================================================================
# Section 23 — LoopDetectorConfig validation
# ===========================================================================


class TestLoopDetectorConfig:
    def test_defaults(self):
        c = LoopDetectorConfig()
        assert c.duplicate_threshold == 3
        assert c.max_cycle_length == 4
        assert c.cycle_repetitions == 3
        assert c.no_progress_window == 5

    def test_duplicate_threshold_lt_2_rejected(self):
        with pytest.raises(ValueError):
            LoopDetectorConfig(duplicate_threshold=1)
        with pytest.raises(ValueError):
            LoopDetectorConfig(duplicate_threshold=0)

    def test_max_cycle_length_lt_1_rejected(self):
        with pytest.raises(ValueError):
            LoopDetectorConfig(max_cycle_length=0)

    def test_cycle_repetitions_lt_2_rejected(self):
        with pytest.raises(ValueError):
            LoopDetectorConfig(cycle_repetitions=1)

    def test_no_progress_window_lt_2_rejected(self):
        with pytest.raises(ValueError):
            LoopDetectorConfig(no_progress_window=1)

    def test_valid_custom_config(self):
        c = LoopDetectorConfig(
            duplicate_threshold=2,
            max_cycle_length=3,
            cycle_repetitions=2,
            no_progress_window=4,
        )
        assert c.duplicate_threshold == 2


# ===========================================================================
# Section 22 — LoopDetectionResult invariants
# ===========================================================================


class TestLoopDetectionResult:
    def test_detected_false_no_reason(self):
        r = LoopDetectionResult(detected=False)
        assert r.reason is None
        assert r.evidence_action_indices == ()

    def test_detected_true_requires_reason(self):
        with pytest.raises(ValueError):
            LoopDetectionResult(detected=True, reason=None)

    def test_detected_false_rejects_reason(self):
        with pytest.raises(ValueError):
            LoopDetectionResult(
                detected=False, reason=LoopDetectionReason.NO_PROGRESS
            )


# ===========================================================================
# Sections 25, 41 — Duplicate detection
# ===========================================================================


class TestDuplicateDetection:
    def test_duplicate_at_threshold(self):
        """A A A with same progress -> DUPLICATE_CALL on 3rd."""
        detector = LoopDetector(
            config=LoopDetectorConfig(duplicate_threshold=3)
        )
        fp_a = _fp("read_file", {"path": "a.py"})
        r0 = detector.observe(_rec(0, fp_a, progress="s1"))
        r1 = detector.observe(_rec(1, fp_a, progress="s1"))
        r2 = detector.observe(_rec(2, fp_a, progress="s1"))
        assert r0.detected is False
        assert r1.detected is False
        assert r2.detected is True
        assert r2.reason is LoopDetectionReason.DUPLICATE_CALL
        assert r2.evidence_action_indices == (0, 1, 2)

    def test_duplicate_below_threshold(self):
        """A A with threshold 3 -> not detected."""
        detector = LoopDetector(
            config=LoopDetectorConfig(duplicate_threshold=3)
        )
        fp_a = _fp("read_file", {"path": "a.py"})
        r0 = detector.observe(_rec(0, fp_a, progress="s1"))
        r1 = detector.observe(_rec(1, fp_a, progress="s1"))
        assert r0.detected is False
        assert r1.detected is False

    def test_duplicate_with_progress_does_not_trigger(self):
        """A A A but progress changes each time -> not a loop."""
        detector = LoopDetector(
            config=LoopDetectorConfig(duplicate_threshold=3)
        )
        fp_a = _fp("poll_status")
        r = _observe_all(
            detector,
            [
                _rec(0, fp_a, progress="v1"),
                _rec(1, fp_a, progress="v2"),
                _rec(2, fp_a, progress="v3"),
            ],
        )
        assert r.detected is False

    def test_unknown_progress_does_not_trigger_duplicate(self):
        """A A A with progress=None -> unknown, not confirmed no-progress."""
        detector = LoopDetector(
            config=LoopDetectorConfig(duplicate_threshold=3)
        )
        fp_a = _fp("read_file", {"path": "a.py"})
        r = _observe_all(
            detector,
            [
                _rec(0, fp_a, progress=None),
                _rec(1, fp_a, progress=None),
                _rec(2, fp_a, progress=None),
            ],
        )
        assert r.detected is False

    def test_non_consecutive_duplicate_does_not_trigger(self):
        """A B A -> not consecutive duplicate."""
        detector = LoopDetector(
            config=LoopDetectorConfig(duplicate_threshold=3)
        )
        fp_a = _fp("read_file", {"path": "a.py"})
        fp_b = _fp("read_file", {"path": "b.py"})
        r = _observe_all(
            detector,
            [
                _rec(0, fp_a, progress="s1"),
                _rec(1, fp_b, progress="s1"),
                _rec(2, fp_a, progress="s1"),
            ],
        )
        assert r.detected is False


# ===========================================================================
# Sections 28-29, 45-46 — Repeating sequence detection
# ===========================================================================


class TestRepeatingSequence:
    def test_ab_cycle_3_repetitions(self):
        """A B A B A B -> REPEATING_SEQUENCE, cycle_length=2."""
        detector = LoopDetector(
            config=LoopDetectorConfig(
                max_cycle_length=4, cycle_repetitions=3,
                duplicate_threshold=10,  # disable duplicate for this test
            )
        )
        fp_a = _fp("A")
        fp_b = _fp("B")
        records = []
        for rep in range(3):
            records.append(_rec(rep * 2, fp_a, progress="s1"))
            records.append(_rec(rep * 2 + 1, fp_b, progress="s1"))
        result = _observe_all(detector, records)
        assert result.detected is True
        assert result.reason is LoopDetectionReason.REPEATING_SEQUENCE
        assert "length 2" in result.message

    def test_abc_cycle_3_repetitions(self):
        """A B C A B C A B C -> REPEATING_SEQUENCE, cycle_length=3."""
        detector = LoopDetector(
            config=LoopDetectorConfig(
                max_cycle_length=4, cycle_repetitions=3,
                duplicate_threshold=10,
            )
        )
        fps = [_fp("A"), _fp("B"), _fp("C")]
        records = []
        for rep in range(3):
            for i, fp in enumerate(fps):
                records.append(_rec(rep * 3 + i, fp, progress="s1"))
        result = _observe_all(detector, records)
        assert result.detected is True
        assert result.reason is LoopDetectionReason.REPEATING_SEQUENCE
        assert "length 3" in result.message

    def test_partial_cycle_does_not_trigger(self):
        """A B A B A -> only 2.5 repetitions of length-2 cycle -> not detected."""
        detector = LoopDetector(
            config=LoopDetectorConfig(
                max_cycle_length=4, cycle_repetitions=3,
                duplicate_threshold=10,
                no_progress_window=10,  # disable no-progress for this test
            )
        )
        fp_a = _fp("A")
        fp_b = _fp("B")
        # A B A B A = 5 records, not 6 (3 * 2)
        records = [
            _rec(0, fp_a, progress="s1"),
            _rec(1, fp_b, progress="s1"),
            _rec(2, fp_a, progress="s1"),
            _rec(3, fp_b, progress="s1"),
            _rec(4, fp_a, progress="s1"),
        ]
        result = _observe_all(detector, records)
        assert result.detected is False

    def test_sequence_with_progress_does_not_trigger(self):
        """A B A B A B but progress changes -> legitimate iteration."""
        detector = LoopDetector(
            config=LoopDetectorConfig(
                max_cycle_length=4, cycle_repetitions=3,
                duplicate_threshold=10,
            )
        )
        fp_a = _fp("A")
        fp_b = _fp("B")
        records = [
            _rec(0, fp_a, progress="p1"),
            _rec(1, fp_b, progress="p2"),
            _rec(2, fp_a, progress="p3"),
            _rec(3, fp_b, progress="p4"),
            _rec(4, fp_a, progress="p5"),
            _rec(5, fp_b, progress="p6"),
        ]
        result = _observe_all(detector, records)
        assert result.detected is False


# ===========================================================================
# Sections 32-33, 49-52 — No-progress detection
# ===========================================================================


class TestNoProgress:
    def test_no_progress_at_window(self):
        """5 actions, same non-None token -> NO_PROGRESS."""
        detector = LoopDetector(
            config=LoopDetectorConfig(
                no_progress_window=5,
                duplicate_threshold=10,  # disable duplicate
                max_cycle_length=2, cycle_repetitions=10,  # disable sequence
            )
        )
        fps = [_fp("A"), _fp("B"), _fp("C"), _fp("D"), _fp("E")]
        records = [
            _rec(i, fps[i], progress="state-X") for i in range(5)
        ]
        result = _observe_all(detector, records)
        assert result.detected is True
        assert result.reason is LoopDetectionReason.NO_PROGRESS
        assert result.evidence_action_indices == (0, 1, 2, 3, 4)

    def test_no_progress_below_window(self):
        """Only 4 actions -> not detected."""
        detector = LoopDetector(
            config=LoopDetectorConfig(
                no_progress_window=5,
                duplicate_threshold=10,
                max_cycle_length=2, cycle_repetitions=10,
            )
        )
        fps = [_fp("A"), _fp("B"), _fp("C"), _fp("D")]
        records = [_rec(i, fps[i], progress="state-X") for i in range(4)]
        result = _observe_all(detector, records)
        assert result.detected is False

    def test_no_progress_reset_on_token_change(self):
        """Token change breaks the no-progress streak."""
        detector = LoopDetector(
            config=LoopDetectorConfig(
                no_progress_window=5,
                duplicate_threshold=10,
                max_cycle_length=2, cycle_repetitions=10,
            )
        )
        # s1 s1 s1 s2 s2 s2 — after s1->s2, the streak resets.
        records = [
            _rec(0, _fp("A"), progress="s1"),
            _rec(1, _fp("B"), progress="s1"),
            _rec(2, _fp("C"), progress="s1"),
            _rec(3, _fp("D"), progress="s2"),
            _rec(4, _fp("E"), progress="s2"),
            _rec(5, _fp("F"), progress="s2"),
        ]
        result = _observe_all(detector, records)
        # Only 3 consecutive s2 at the end, window=5 -> not detected.
        assert result.detected is False

    def test_no_progress_resets_then_reaches_window(self):
        """After reset, a new streak reaching window triggers."""
        detector = LoopDetector(
            config=LoopDetectorConfig(
                no_progress_window=3,
                duplicate_threshold=10,
                max_cycle_length=2, cycle_repetitions=10,
            )
        )
        records = [
            _rec(0, _fp("A"), progress="s1"),
            _rec(1, _fp("B"), progress="s1"),
            _rec(2, _fp("C"), progress="s2"),  # reset
            _rec(3, _fp("D"), progress="s3"),  # reset
            _rec(4, _fp("E"), progress="s3"),
            _rec(5, _fp("F"), progress="s3"),
            _rec(6, _fp("G"), progress="s3"),
        ]
        result = _observe_all(detector, records)
        assert result.detected is True
        assert result.reason is LoopDetectionReason.NO_PROGRESS
        assert result.evidence_action_indices == (4, 5, 6)

    def test_unknown_progress_does_not_trigger_no_progress(self):
        """All None tokens -> unknown, not confirmed no-progress."""
        detector = LoopDetector(
            config=LoopDetectorConfig(
                no_progress_window=5,
                duplicate_threshold=10,
                max_cycle_length=2, cycle_repetitions=10,
            )
        )
        fps = [_fp("A"), _fp("B"), _fp("C"), _fp("D"), _fp("E")]
        records = [_rec(i, fps[i], progress=None) for i in range(5)]
        result = _observe_all(detector, records)
        assert result.detected is False

    def test_no_progress_with_successful_actions(self):
        """Tool success != task progress; same token still triggers."""
        detector = LoopDetector(
            config=LoopDetectorConfig(
                no_progress_window=5,
                duplicate_threshold=10,
                max_cycle_length=2, cycle_repetitions=10,
            )
        )
        fps = [_fp("A"), _fp("B"), _fp("C"), _fp("D"), _fp("E")]
        records = [
            _rec(i, fps[i], success=True, progress="state-X")
            for i in range(5)
        ]
        result = _observe_all(detector, records)
        assert result.detected is True
        assert result.reason is LoopDetectionReason.NO_PROGRESS


# ===========================================================================
# Section 53 — Detection priority
# ===========================================================================


class TestDetectionPriority:
    def test_duplicate_over_no_progress(self):
        """A A A with same token satisfies both DUPLICATE and NO_PROGRESS.
        DUPLICATE must win."""
        detector = LoopDetector(
            config=LoopDetectorConfig(
                duplicate_threshold=3,
                no_progress_window=3,
            )
        )
        fp_a = _fp("A")
        records = [
            _rec(0, fp_a, progress="s1"),
            _rec(1, fp_a, progress="s1"),
            _rec(2, fp_a, progress="s1"),
        ]
        result = _observe_all(detector, records)
        assert result.detected is True
        assert result.reason is LoopDetectionReason.DUPLICATE_CALL

    def test_sequence_over_no_progress(self):
        """A B A B A B with same token satisfies SEQUENCE and NO_PROGRESS.
        SEQUENCE must win."""
        detector = LoopDetector(
            config=LoopDetectorConfig(
                duplicate_threshold=10,  # disable duplicate
                max_cycle_length=4, cycle_repetitions=3,
                no_progress_window=3,
            )
        )
        fp_a = _fp("A")
        fp_b = _fp("B")
        records = []
        for rep in range(3):
            records.append(_rec(rep * 2, fp_a, progress="s1"))
            records.append(_rec(rep * 2 + 1, fp_b, progress="s1"))
        result = _observe_all(detector, records)
        assert result.detected is True
        assert result.reason is LoopDetectionReason.REPEATING_SEQUENCE

    def test_duplicate_over_sequence(self):
        """A A A A A A — satisfies both DUPLICATE (threshold=3) and
        SEQUENCE (cycle_len=1? no, min cycle_len=2). Actually A A A A A A
        with cycle_len=2 would be (A A)(A A)(A A) = 3 reps, but also
        duplicate threshold=3. DUPLICATE should win."""
        detector = LoopDetector(
            config=LoopDetectorConfig(
                duplicate_threshold=3,
                max_cycle_length=4, cycle_repetitions=3,
            )
        )
        fp_a = _fp("A")
        records = [_rec(i, fp_a, progress="s1") for i in range(6)]
        result = _observe_all(detector, records)
        assert result.detected is True
        assert result.reason is LoopDetectionReason.DUPLICATE_CALL


# ===========================================================================
# Section 34, 54 — Failed actions participate
# ===========================================================================


class TestFailedActions:
    def test_failed_duplicate_detection(self):
        """Same fingerprint, same progress, all PERMANENT failures -> DUPLICATE."""
        detector = LoopDetector(
            config=LoopDetectorConfig(duplicate_threshold=3)
        )
        fp = _fp("run_tests", {"suite": "x"})
        records = [
            _rec(i, fp, success=False, error_type=ToolErrorType.PERMANENT,
                 progress="s1")
            for i in range(3)
        ]
        result = _observe_all(detector, records)
        assert result.detected is True
        assert result.reason is LoopDetectionReason.DUPLICATE_CALL

    def test_failed_no_progress(self):
        """Failed actions with same token -> NO_PROGRESS."""
        detector = LoopDetector(
            config=LoopDetectorConfig(
                no_progress_window=3,
                duplicate_threshold=10,
                max_cycle_length=2, cycle_repetitions=10,
            )
        )
        fps = [_fp("A"), _fp("B"), _fp("C")]
        records = [
            _rec(i, fps[i], success=False,
                 error_type=ToolErrorType.EXECUTION, progress="s1")
            for i in range(3)
        ]
        result = _observe_all(detector, records)
        assert result.detected is True
        assert result.reason is LoopDetectionReason.NO_PROGRESS


# ===========================================================================
# Sections 35, 55 — Retry attempts do NOT inflate history
# ===========================================================================


class TestRetryDoesNotInflateHistory:
    def test_retry_produces_single_action_record(self):
        """A ToolRuntime call with 3 internal attempts -> 1 ActionRecord."""
        registry = ToolRegistry()
        handler = ScriptedTool([
            ("transient", "fail 1"),
            ("transient", "fail 2"),
            ("success", "done"),
        ])
        registry.register(
            ToolSpec(
                name="read_repo",
                description="read",
                input_schema={"type": "object"},
                required_permissions=frozenset(),
                timeout_seconds=1.0,
                side_effect=ToolSideEffect.READ_ONLY,
                retry_policy=RetryPolicy(
                    max_attempts=3, initial_backoff_seconds=0,
                ),
            ),
            handler,
        )
        from tests.fake_tools import FakeSleeper
        runtime = ToolRuntime(registry=registry, sleep=FakeSleeper())

        call = ToolCall(tool_name="read_repo", arguments={})
        result = asyncio.run(
            runtime.execute(call, ToolExecutionContext())
        )
        # The ToolRuntime did 3 internal attempts.
        assert result.attempt_count == 3

        # But the caller converts this to ONE ActionRecord.
        record = ActionRecord.from_tool_result(
            action_index=0, call=call, result=result, progress_token="v1",
        )
        history = ActionHistory(max_size=64)
        history.append(record)
        assert len(history) == 1
        assert history.records()[0].success is True
        assert history.records()[0].error_type is None


# ===========================================================================
# Section 37 — Detector does not change Task/Run state
# ===========================================================================


class TestDetectorDoesNotChangeState:
    def test_detector_only_returns_result(self):
        """The detector observe() returns a result; it does not raise or
        mutate any Task/Run/Checkpoint."""
        detector = LoopDetector()
        fp = _fp("A")
        for i in range(5):
            result = detector.observe(_rec(i, fp, progress="s1"))
            assert isinstance(result, LoopDetectionResult)
        # The detector's history is the only state it holds.
        assert len(detector.history) == 5


# ===========================================================================
# Section 38 — False result is structured, not None
# ===========================================================================


def test_false_result_is_structured_not_none():
    detector = LoopDetector()
    result = detector.observe(_rec(0, _fp("A"), progress="v1"))
    assert result is not None
    assert isinstance(result, LoopDetectionResult)
    assert result.detected is False
    assert result.reason is None


# ===========================================================================
# Section 56 — ToolRuntime does not auto-call detector
# ===========================================================================


def test_tool_runtime_does_not_call_detector():
    """ToolRuntime.execute() has no LoopDetector dependency."""
    import inspect
    from tools.runtime import ToolRuntime
    sig = inspect.signature(ToolRuntime.__init__)
    params = set(sig.parameters)
    assert "loop_detector" not in params
    assert "detector" not in params


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_history_detects_nothing(self):
        detector = LoopDetector()
        # No observations yet — nothing to detect.
        # We can't call _detect directly, but observe on first record
        # with a unique fingerprint should be clean.
        result = detector.observe(_rec(0, _fp("unique"), progress="v1"))
        assert result.detected is False

    def test_single_record_never_triggers(self):
        detector = LoopDetector(
            config=LoopDetectorConfig(duplicate_threshold=2)
        )
        result = detector.observe(_rec(0, _fp("A"), progress="s1"))
        assert result.detected is False

    def test_two_distinct_actions_no_loop(self):
        detector = LoopDetector()
        result = _observe_all(
            detector,
            [
                _rec(0, _fp("A"), progress="v1"),
                _rec(1, _fp("B"), progress="v2"),
            ],
        )
        assert result.detected is False

    def test_history_bound_does_not_break_detection(self):
        """Detection works even when history is bounded and old records
        are evicted."""
        detector = LoopDetector(
            config=LoopDetectorConfig(
                duplicate_threshold=3,
            ),
            history=ActionHistory(max_size=3),
        )
        fp_a = _fp("A")
        # Fill with unrelated actions that get evicted.
        detector.observe(_rec(0, _fp("X"), progress="s1"))
        detector.observe(_rec(1, _fp("Y"), progress="s1"))
        # Now 3 consecutive A's — but X and Y get evicted, leaving
        # only the last 3 (which are all A).
        r2 = detector.observe(_rec(2, fp_a, progress="s1"))
        r3 = detector.observe(_rec(3, fp_a, progress="s1"))
        r4 = detector.observe(_rec(4, fp_a, progress="s1"))
        assert r4.detected is True
        assert r4.reason is LoopDetectionReason.DUPLICATE_CALL
        assert r4.evidence_action_indices == (2, 3, 4)


# ===========================================================================
# Phase 3 Step 2 — LoopDetector.reset()
# ===========================================================================


class TestLoopDetectorReset:
    def test_reset_clears_history(self):
        """After reset, history is empty and detection starts fresh."""
        detector = LoopDetector(
            config=LoopDetectorConfig(duplicate_threshold=3)
        )
        fp_a = _fp("A")
        detector.observe(_rec(0, fp_a, progress="s1"))
        detector.observe(_rec(1, fp_a, progress="s1"))
        detector.observe(_rec(2, fp_a, progress="s1"))
        assert len(detector.history) == 3

        detector.reset()
        assert len(detector.history) == 0
        assert detector.history.records() == ()

    def test_reset_preserves_config(self):
        """Reset does not modify the config."""
        config = LoopDetectorConfig(
            duplicate_threshold=5, no_progress_window=10
        )
        detector = LoopDetector(config=config)
        detector.observe(_rec(0, _fp("A"), progress="s1"))
        detector.reset()
        assert detector.config is config
        assert detector.config.duplicate_threshold == 5

    def test_reset_allows_new_detection(self):
        """After reset, a new loop can be detected from fresh actions."""
        detector = LoopDetector(
            config=LoopDetectorConfig(duplicate_threshold=3)
        )
        fp_a = _fp("A")
        # First loop episode.
        for i in range(3):
            detector.observe(_rec(i, fp_a, progress="s1"))
        assert detector.history.records()[-1].action_index == 2

        detector.reset()

        # New loop episode with different tool.
        fp_b = _fp("B")
        r3 = detector.observe(_rec(3, fp_b, progress="s2"))
        r4 = detector.observe(_rec(4, fp_b, progress="s2"))
        r5 = detector.observe(_rec(5, fp_b, progress="s2"))
        assert r5.detected is True
        assert r5.reason is LoopDetectionReason.DUPLICATE_CALL
        assert r5.evidence_action_indices == (3, 4, 5)

    def test_reset_then_single_action_no_loop(self):
        """After reset, a single action does not trigger old evidence."""
        detector = LoopDetector(
            config=LoopDetectorConfig(duplicate_threshold=3)
        )
        fp_a = _fp("A")
        for i in range(3):
            detector.observe(_rec(i, fp_a, progress="s1"))

        detector.reset()

        # Single A after reset — should NOT trigger (history is empty).
        result = detector.observe(_rec(3, fp_a, progress="s1"))
        assert result.detected is False
