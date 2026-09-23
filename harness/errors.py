"""Errors for the harness runtime state layer.

Phase 1 Step 1 introduced ``InvalidStateTransitionError`` for illegal
lifecycle transitions. Construction-time invariant violations keep
using ``ValueError`` to stay distinguishable from lifecycle errors.

Phase 1 Step 2 introduced ``ResumeError`` for invalid resume attempts.

Phase 2 Step 1 restructures the runtime error hierarchy so that
``ResumeError`` no longer carries unrelated runtime validation (e.g. a
``start()`` on a non-PENDING task). The hierarchy is now:

    HarnessRuntimeError
        ├── InvalidRuntimeStateError   (bad runtime precondition)
        └── ResumeError                (bad resume request)

``InvalidStateTransitionError`` stays a standalone ``Exception`` because
it predates the runtime and is raised by the pure state layer, which has
no dependency on the runtime concepts.
"""

from __future__ import annotations


class InvalidStateTransitionError(Exception):
    """Raised when a Task / Run lifecycle transition is not allowed.

    The message always carries the entity type, the current status and
    the target status so failures are easy to diagnose without a
    debugger.
    """

    def __init__(self, entity: str, current: str, target: str) -> None:
        self.entity = entity
        self.current = current
        self.target = target
        super().__init__(f"Invalid {entity} transition: {current} -> {target}")


class HarnessRuntimeError(Exception):
    """Base class for harness runtime errors.

    Distinct from ``InvalidStateTransitionError`` (pure state layer) and
    from ``ValueError`` (construction-time invariant violations).
    """


class InvalidRuntimeStateError(HarnessRuntimeError):
    """Raised when a runtime precondition on Task / Run state is violated.

    For example, calling ``start()`` on a Task that is not ``PENDING``.
    This is a runtime-usage error, not a resume error, so it stays
    separate from ``ResumeError``.
    """


class ResumeError(HarnessRuntimeError):
    """Raised when a resume attempt is invalid.

    Covers at least:

    * the referenced checkpoint does not exist,
    * the checkpoint belongs to a different Run,
    * the source Run belongs to a different Task,
    * the source Run is not in a resumable (INTERRUPTED) status.

    Keeping this distinct from ``InvalidRuntimeStateError`` lets callers
    tell "bad resume request" apart from "bad runtime precondition".
    """
