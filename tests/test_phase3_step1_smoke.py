"""Smoke test for Phase 3 Step 1 — loop detection.

Proves the end-to-end loop detection guarantee:

    Repeating AB sequence with no progress -> REPEATING_SEQUENCE
    Same AB pattern with changing progress -> detected=False (legitimate)
    Consecutive duplicate with no progress -> DUPLICATE_CALL
    No-progress window -> NO_PROGRESS

Fully offline, deterministic, no LLM, no network, no MCP.
"""

from __future__ import annotations

from harness.loop_detection import (
    ActionFingerprint,
    ActionRecord,
    LoopDetectionReason,
    LoopDetector,
    LoopDetectorConfig,
)
from tools.models import ToolCall


def _fp(tool: str) -> ActionFingerprint:
    return ActionFingerprint.from_call(ToolCall(tool_name=tool, arguments={}))


def _rec(idx: int, fp: ActionFingerprint, progress: str | None) -> ActionRecord:
    return ActionRecord(
        action_index=idx, fingerprint=fp, success=True, progress_token=progress
    )


def test_phase3_step1_smoke():
    fp_a = _fp("read_file")
    fp_b = _fp("run_tests")

    # --- Case 1: AB cycle with no progress -> REPEATING_SEQUENCE ---
    detector = LoopDetector(
        config=LoopDetectorConfig(
            duplicate_threshold=10,  # disable duplicate for sequence test
            max_cycle_length=4,
            cycle_repetitions=3,
            no_progress_window=5,
        )
    )
    no_progress_seq = []
    for rep in range(3):
        no_progress_seq.append(_rec(rep * 2, fp_a, progress="state-1"))
        no_progress_seq.append(_rec(rep * 2 + 1, fp_b, progress="state-1"))
    result = LoopDetectionResult(detected=False)
    for r in no_progress_seq:
        result = detector.observe(r)
    assert result.detected is True
    assert result.reason is LoopDetectionReason.REPEATING_SEQUENCE
    assert "length 2" in result.message
    assert len(result.evidence_action_indices) == 6

    # --- Case 2: same AB pattern but progress changes -> legitimate ---
    detector2 = LoopDetector(
        config=LoopDetectorConfig(
            duplicate_threshold=10,
            max_cycle_length=4,
            cycle_repetitions=3,
            no_progress_window=5,
        )
    )
    progress_seq = [
        _rec(0, fp_a, progress="state-1"),
        _rec(1, fp_b, progress="state-2"),
        _rec(2, fp_a, progress="state-3"),
        _rec(3, fp_b, progress="state-4"),
        _rec(4, fp_a, progress="state-5"),
        _rec(5, fp_b, progress="state-6"),
    ]
    result2 = LoopDetectionResult(detected=False)
    for r in progress_seq:
        result2 = detector2.observe(r)
    assert result2.detected is False

    # --- Case 3: consecutive duplicate with no progress -> DUPLICATE_CALL ---
    detector3 = LoopDetector(
        config=LoopDetectorConfig(duplicate_threshold=3)
    )
    fp_c = _fp("read_file")
    result3 = LoopDetectionResult(detected=False)
    for i in range(3):
        result3 = detector3.observe(_rec(i, fp_c, progress="stuck"))
    assert result3.detected is True
    assert result3.reason is LoopDetectionReason.DUPLICATE_CALL
    assert result3.evidence_action_indices == (0, 1, 2)

    # --- Case 4: no-progress window ---
    detector4 = LoopDetector(
        config=LoopDetectorConfig(
            no_progress_window=5,
            duplicate_threshold=10,
            max_cycle_length=2, cycle_repetitions=10,
        )
    )
    fps = [_fp("A"), _fp("B"), _fp("C"), _fp("D"), _fp("E")]
    result4 = LoopDetectionResult(detected=False)
    for i, fp in enumerate(fps):
        result4 = detector4.observe(_rec(i, fp, progress="frozen"))
    assert result4.detected is True
    assert result4.reason is LoopDetectionReason.NO_PROGRESS
    assert result4.evidence_action_indices == (0, 1, 2, 3, 4)


# Import here to satisfy the type checker for the LoopDetectionResult
# used as a sentinel above.
from harness.loop_detection import LoopDetectionResult  # noqa: E402
