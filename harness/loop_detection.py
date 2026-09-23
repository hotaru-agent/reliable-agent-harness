"""Action history, duplicate & no-progress loop detection (Phase 3 Step 1).

Deterministic, rule-based detection of Agent-level logical-action loops.

This module observes *logical Agent actions* (one ``ActionRecord`` per
``ToolRuntime.execute()`` invocation, regardless of internal retry
attempts) and detects three structural patterns:

* ``DUPLICATE_CALL``        — same fingerprint repeated consecutively
                               without observable progress.
* ``REPEATING_SEQUENCE``     — a short action cycle repeated in full
                               without observable progress.
* ``NO_PROGRESS``            — the same non-None ``progress_token`` held
                               across the last ``no_progress_window``
                               actions.

It does NOT:

* call an LLM;
* compute embeddings / semantic similarity;
* modify Task / Run / Checkpoint state;
* trigger replan or abort;
* integrate with ``HarnessRuntime`` or ``ToolRuntime``.

Loop detection is a *detection* boundary only. Policy integration is
Phase 3 Step 2.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Tuple

from tools.models import ToolCall, ToolErrorType, ToolExecutionResult


# ---------------------------------------------------------------------------
# Canonical argument normalization
# ---------------------------------------------------------------------------


class ActionNormalizationError(Exception):
    """Raised when an argument value cannot be stably canonicalized.

    The fingerprint only supports JSON-like data (dict / list / str /
    int / float / bool / None). Anything else is rejected explicitly
    rather than silently ``repr()``-ed (which could leak memory addresses
    or non-deterministic state).
    """


def canonicalize_arguments(arguments: Any) -> str:
    """Return a deterministic canonical JSON string for ``arguments``.

    Dict key ordering is irrelevant (``sort_keys=True``). ``True`` and
    ``1`` remain distinct because JSON distinguishes ``true`` from ``1``.

    Raises ``ActionNormalizationError`` for unsupported types or values
    that cannot be stably serialized (e.g. NaN / Infinity).
    """
    _check_jsonable(arguments)
    try:
        return json.dumps(
            arguments,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (ValueError, TypeError) as exc:
        raise ActionNormalizationError(
            f"arguments cannot be canonicalized: {exc}"
        ) from exc


def _check_jsonable(value: Any, path: str = "$") -> None:
    """Recursively verify ``value`` is JSON-like and stably serializable."""
    if value is None or isinstance(value, (str, bool)):
        return
    # bool is a subclass of int — handled above. int / float are fine.
    if isinstance(value, (int, float)):
        # Reject float NaN / Inf which json would reject with allow_nan=False
        # but isinstance check is cheaper; json.dumps(allow_nan=False) raises.
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ActionNormalizationError(
                    f"dict keys must be str at {path}, got {type(k).__name__}"
                )
            _check_jsonable(v, f"{path}.{k}")
        return
    if isinstance(value, list):
        for i, v in enumerate(value):
            _check_jsonable(v, f"{path}[{i}]")
        return
    raise ActionNormalizationError(
        f"unsupported type {type(value).__name__} at {path}; "
        "only JSON-like data (dict/list/str/int/float/bool/None) is allowed"
    )


# ---------------------------------------------------------------------------
# Action fingerprint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionFingerprint:
    """Stable identity for a logical Agent action.

    Two actions have the same fingerprint iff they use the same tool
    and logically-equal arguments (dict key ordering irrelevant).
    """

    tool_name: str
    canonical_arguments: str

    def __post_init__(self) -> None:
        if not self.tool_name or not self.tool_name.strip():
            raise ValueError("tool_name must be non-empty")
        if not isinstance(self.canonical_arguments, str):
            raise ValueError("canonical_arguments must be a str")

    @classmethod
    def from_call(cls, call: ToolCall) -> "ActionFingerprint":
        return cls(
            tool_name=call.tool_name,
            canonical_arguments=canonicalize_arguments(call.arguments),
        )


# ---------------------------------------------------------------------------
# Action record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionRecord:
    """One logical Agent action observation.

    ``action_index`` is the logical Agent action sequence number (0,
    1, 2, ...), NOT the ToolRuntime internal attempt number. A single
    ``ToolRuntime.execute()`` that retried 3 times internally produces
    exactly one ``ActionRecord``.

    ``progress_token`` is caller-supplied evidence of task progress
    (e.g. ``"test-failure-count:5"``). ``None`` means *unknown*, not
    *confirmed no progress*.
    """

    action_index: int
    fingerprint: ActionFingerprint
    success: bool
    error_type: Optional[ToolErrorType] = None
    progress_token: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.action_index, int) or self.action_index < 0:
            raise ValueError("action_index must be an int >= 0")
        if not isinstance(self.fingerprint, ActionFingerprint):
            raise ValueError("fingerprint must be an ActionFingerprint")
        if not isinstance(self.success, bool):
            raise ValueError("success must be a bool")
        if self.error_type is not None and not isinstance(
            self.error_type, ToolErrorType
        ):
            raise ValueError("error_type must be a ToolErrorType or None")
        if self.progress_token is not None and not isinstance(
            self.progress_token, str
        ):
            raise ValueError("progress_token must be a str or None")
        if self.success and self.error_type is not None:
            raise ValueError("a successful action must not carry an error_type")
        if not self.success and self.error_type is None:
            raise ValueError("a failed action must carry an error_type")

    @classmethod
    def from_tool_result(
        cls,
        *,
        action_index: int,
        call: ToolCall,
        result: ToolExecutionResult,
        progress_token: Optional[str] = None,
    ) -> "ActionRecord":
        """Build an ``ActionRecord`` from a logical ``ToolCall`` + result.

        Does NOT copy tool output, retry history or error details — loop
        detection only needs the fingerprint, success flag and error
        type.
        """
        return cls(
            action_index=action_index,
            fingerprint=ActionFingerprint.from_call(call),
            success=result.success,
            error_type=None if result.success else result.error.error_type,
            progress_token=progress_token,
        )


# ---------------------------------------------------------------------------
# Action history
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionHistory:
    """Bounded history of ``ActionRecord`` observations.

    Uses ``collections.deque(maxlen=...)`` so the oldest records are
    evicted once the bound is reached. ``records()`` returns a snapshot
    tuple so callers cannot mutate the internal deque.

    Append enforces strictly-monotonic ``action_index``: each new
    record's index must be greater than the last appended record's
    index. This prevents accidental index reuse or out-of-order
    appends. ``AgentActionController`` guarantees consecutive +1
    indexing; ``ActionHistory`` only guarantees strict increase.
    """

    max_size: int
    _records: deque = field(default_factory=lambda: deque(maxlen=64), repr=False)
    _last_action_index: int = field(default=-1, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.max_size, int) or self.max_size < 1:
            raise ValueError("max_size must be an int >= 1")
        # Rebind the deque with the correct maxlen. We can't mutate
        # ``maxlen`` on an existing deque, so rebuild it.
        if self._records.maxlen != self.max_size:
            object.__setattr__(
                self,
                "_records",
                deque(self._records, maxlen=self.max_size),
            )
        # Recompute _last_action_index from existing records (if any
        # were passed in via default_factory or rebuild).
        if self._records and self._last_action_index < 0:
            object.__setattr__(
                self, "_last_action_index", self._records[-1].action_index
            )

    def append(self, record: ActionRecord) -> None:
        if not isinstance(record, ActionRecord):
            raise ValueError("record must be an ActionRecord")
        if record.action_index <= self._last_action_index:
            raise ValueError(
                f"action_index must be strictly greater than the last "
                f"appended index ({self._last_action_index}); got "
                f"{record.action_index}"
            )
        self._records.append(record)
        object.__setattr__(self, "_last_action_index", record.action_index)

    def records(self) -> Tuple[ActionRecord, ...]:
        """Return a snapshot tuple of all records (insertion order)."""
        return tuple(self._records)

    def recent(self, n: int) -> Tuple[ActionRecord, ...]:
        """Return the last ``n`` records (or fewer if history is shorter)."""
        if n < 0:
            raise ValueError("n must be >= 0")
        items = list(self._records)
        return tuple(items[-n:]) if n else ()

    def __len__(self) -> int:
        return len(self._records)


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------


class LoopDetectionReason(Enum):
    """Why a loop was detected."""

    DUPLICATE_CALL = "duplicate_call"
    REPEATING_SEQUENCE = "repeating_sequence"
    NO_PROGRESS = "no_progress"


@dataclass(frozen=True)
class LoopDetectionResult:
    """Structured loop-detection outcome.

    Invariants:

    * ``detected=False`` -> ``reason is None``
    * ``detected=True``  -> ``reason is not None``
    * ``evidence_action_indices`` references only the logical actions
      that triggered the detection (for future replan / telemetry).
    """

    detected: bool
    reason: Optional[LoopDetectionReason] = None
    message: Optional[str] = None
    evidence_action_indices: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.detected, bool):
            raise ValueError("detected must be a bool")
        if self.detected and self.reason is None:
            raise ValueError("a detected result must carry a reason")
        if not self.detected and self.reason is not None:
            raise ValueError("a non-detected result must not carry a reason")
        if not isinstance(self.evidence_action_indices, tuple):
            raise ValueError("evidence_action_indices must be a tuple")


@dataclass(frozen=True)
class LoopDetectorConfig:
    """Deterministic loop-detector configuration.

    * ``duplicate_threshold`` — number of *consecutive* identical
      fingerprints (with no observable progress) that triggers
      ``DUPLICATE_CALL``.
    * ``max_cycle_length`` — largest cycle length considered for
      ``REPEATING_SEQUENCE``.
    * ``cycle_repetitions`` — number of full cycle repetitions required.
    * ``no_progress_window`` — number of recent actions that must share
      the same non-None ``progress_token`` to trigger ``NO_PROGRESS``.
    """

    duplicate_threshold: int = 3
    max_cycle_length: int = 4
    cycle_repetitions: int = 3
    no_progress_window: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.duplicate_threshold, int) or self.duplicate_threshold < 2:
            raise ValueError("duplicate_threshold must be an int >= 2")
        if not isinstance(self.max_cycle_length, int) or self.max_cycle_length < 1:
            raise ValueError("max_cycle_length must be an int >= 1")
        if not isinstance(self.cycle_repetitions, int) or self.cycle_repetitions < 2:
            raise ValueError("cycle_repetitions must be an int >= 2")
        if not isinstance(self.no_progress_window, int) or self.no_progress_window < 2:
            raise ValueError("no_progress_window must be an int >= 2")


# ---------------------------------------------------------------------------
# Loop detector
# ---------------------------------------------------------------------------


class LoopDetector:
    """Deterministic loop detector over a bounded ``ActionHistory``.

    ``observe(record)`` appends the record and runs the three detectors
    in priority order:

        1. DUPLICATE_CALL
        2. REPEATING_SEQUENCE
        3. NO_PROGRESS

    The detector never modifies Task / Run / Checkpoint state and never
    calls an LLM. It only returns a ``LoopDetectionResult``.
    """

    def __init__(
        self,
        *,
        config: LoopDetectorConfig | None = None,
        history: ActionHistory | None = None,
    ) -> None:
        self._config = config or LoopDetectorConfig()
        self._history = history or ActionHistory(max_size=64)

    @property
    def config(self) -> LoopDetectorConfig:
        return self._config

    @property
    def history(self) -> ActionHistory:
        return self._history

    def observe(self, record: ActionRecord) -> LoopDetectionResult:
        """Append ``record`` and return the detection result for it."""
        self._history.append(record)
        return self._detect()

    def reset(self) -> None:
        """Clear the action history, keeping the config.

        After reset, the detector can be reused for a fresh loop
        episode. The config is NOT modified.
        """
        self._history = ActionHistory(max_size=self._history.max_size)

    # ------------------------------------------------------------------
    # Detectors — run in priority order.
    # ------------------------------------------------------------------

    def _detect(self) -> LoopDetectionResult:
        dup = self._detect_duplicate()
        if dup.detected:
            return dup

        seq = self._detect_sequence()
        if seq.detected:
            return seq

        nop = self._detect_no_progress()
        if nop.detected:
            return nop

        return LoopDetectionResult(detected=False)

    # --- DUPLICATE_CALL -----------------------------------------------

    def _detect_duplicate(self) -> LoopDetectionResult:
        threshold = self._config.duplicate_threshold
        recs = self._history.recent(threshold)
        if len(recs) < threshold:
            return LoopDetectionResult(detected=False)

        # All recent `threshold` records must share the same fingerprint.
        first_fp = recs[0].fingerprint
        if any(r.fingerprint != first_fp for r in recs):
            return LoopDetectionResult(detected=False)

        # Progress must NOT have changed across the window. ``None``
        # tokens mean "unknown" — they do not confirm no-progress, so
        # a window of all-None tokens does not trigger duplicate loop.
        tokens = {r.progress_token for r in recs}
        if len(tokens) != 1 or next(iter(tokens)) is None:
            return LoopDetectionResult(detected=False)

        return LoopDetectionResult(
            detected=True,
            reason=LoopDetectionReason.DUPLICATE_CALL,
            message=(
                f"fingerprint {first_fp.tool_name!r} repeated "
                f"{threshold} times consecutively with no progress"
            ),
            evidence_action_indices=tuple(r.action_index for r in recs),
        )

    # --- REPEATING_SEQUENCE -------------------------------------------

    def _detect_sequence(self) -> LoopDetectionResult:
        max_len = self._config.max_cycle_length
        reps = self._config.cycle_repetitions

        for cycle_len in range(2, max_len + 1):
            needed = cycle_len * reps
            recs = self._history.recent(needed)
            if len(recs) < needed:
                continue

            fps = [r.fingerprint for r in recs]
            cycle = fps[:cycle_len]
            if all(
                fps[i] == cycle[i % cycle_len]
                for i in range(needed)
            ):
                # Progress must NOT have changed across the window.
                tokens = {r.progress_token for r in recs}
                if len(tokens) == 1 and next(iter(tokens)) is not None:
                    return LoopDetectionResult(
                        detected=True,
                        reason=LoopDetectionReason.REPEATING_SEQUENCE,
                        message=(
                            f"action cycle of length {cycle_len} "
                            f"repeated {reps} times with no progress"
                        ),
                        evidence_action_indices=tuple(
                            r.action_index for r in recs
                        ),
                    )
        return LoopDetectionResult(detected=False)

    # --- NO_PROGRESS --------------------------------------------------

    def _detect_no_progress(self) -> LoopDetectionResult:
        window = self._config.no_progress_window
        recs = self._history.recent(window)
        if len(recs) < window:
            return LoopDetectionResult(detected=False)

        tokens = {r.progress_token for r in recs}
        # Exactly one distinct token, and it must be non-None.
        if len(tokens) != 1:
            return LoopDetectionResult(detected=False)
        token = next(iter(tokens))
        if token is None:
            return LoopDetectionResult(detected=False)

        return LoopDetectionResult(
            detected=True,
            reason=LoopDetectionReason.NO_PROGRESS,
            message=(
                f"progress token {token!r} unchanged across last "
                f"{window} actions"
            ),
            evidence_action_indices=tuple(r.action_index for r in recs),
        )
