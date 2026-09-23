"""Deterministic retry eligibility and backoff computation (Phase 2 Step 2).

Pure functions — no I/O, no side effects, no LLM. The runtime calls
``should_retry`` to decide whether to attempt again, and
``compute_backoff`` to compute the deterministic sleep delay.
"""

from __future__ import annotations

from tools.models import RetryPolicy, ToolError, ToolSideEffect, ToolSpec


def should_retry(
    *,
    spec: ToolSpec,
    error: ToolError,
    attempt_number: int,
) -> bool:
    """Decide whether the runtime should retry after a failed attempt.

    All five conditions must hold:

    1. ``error.retryable`` is True (error-level eligibility).
    2. ``error.error_type`` is in ``policy.retryable_error_types``
       (policy narrows which error types are retried).
    3. ``attempt_number < policy.max_attempts`` (attempts remaining).
    4. ``spec.side_effect`` is not ``SIDE_EFFECTING`` (side-effect safety).
    5. (implied) the previous attempt actually failed — the caller only
       calls this on a failure path.

    ``attempt_number`` is the 1-based number of the attempt that just
    failed (1 for the first attempt). When ``attempt_number >=
    max_attempts`` there are no attempts remaining.
    """
    policy = spec.retry_policy

    if not error.retryable:
        return False

    if error.error_type not in policy.retryable_error_types:
        return False

    if attempt_number >= policy.max_attempts:
        return False

    if spec.side_effect is ToolSideEffect.SIDE_EFFECTING:
        return False

    return True


def compute_backoff(policy: RetryPolicy, retry_index: int) -> float:
    """Compute the deterministic backoff delay for a retry.

    ``retry_index`` is 1-based (1 for the first retry sleep, i.e. after
    attempt 1 fails and before attempt 2 begins).

    Formula:

        delay = min(
            initial_backoff_seconds * backoff_multiplier ** (retry_index - 1),
            max_backoff_seconds,
        )

    No jitter. When ``initial_backoff_seconds`` is 0 the delay is always
    0 (useful for tests that don't want to sleep).
    """
    if retry_index < 1:
        raise ValueError("retry_index must be >= 1")

    raw = policy.initial_backoff_seconds * (
        policy.backoff_multiplier ** (retry_index - 1)
    )
    return min(raw, policy.max_backoff_seconds)
