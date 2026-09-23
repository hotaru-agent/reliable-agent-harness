"""Tool contract data models (Phase 2 Step 1 + Step 2).

Defines the unified metadata, invocation input, execution context,
structured error data and normalized result for every tool call that
flows through the ``ToolRuntime``.

Everything here is frozen data — no runtime mutable state lives on
``ToolSpec`` or ``RegisteredTool``. Side-effect metadata and retry
policy are modelled as explicit data so the runtime can make
deterministic, LLM-free reliability decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, FrozenSet, Optional, Protocol


# ---------------------------------------------------------------------------
# Side-effect metadata
# ---------------------------------------------------------------------------


class ToolSideEffect(Enum):
    """Side-effect classification for a tool.

    Used by the retry policy for side-effect safety:

    * ``READ_ONLY`` — no external side effects; automatic retry allowed.
    * ``IDEMPOTENT`` — may produce external changes, but repeating the
      same call has idempotent semantics; automatic retry allowed.
    * ``SIDE_EFFECTING`` — may produce side effects that are not safe to
      repeat; automatic retry NOT allowed.
    """

    READ_ONLY = "read_only"
    IDEMPOTENT = "idempotent"
    SIDE_EFFECTING = "side_effecting"


# ---------------------------------------------------------------------------
# Tool source metadata (Phase 5 Step 1)
# ---------------------------------------------------------------------------


class ToolSource(Enum):
    """Where a tool's handler comes from.

    * ``LOCAL`` — a Python handler registered directly by the harness
      owner. ``provider_id`` must be ``None``.
    * ``MCP`` — a tool discovered from an MCP server via
      ``MCPToolProvider``. ``provider_id`` must be a non-empty string
      identifying the provider.
    """

    LOCAL = "local"
    MCP = "mcp"


# ---------------------------------------------------------------------------
# Structured error types (needed by RetryPolicy)
# ---------------------------------------------------------------------------


class ToolErrorType(Enum):
    """Structured classification of a tool failure."""

    NOT_FOUND = "not_found"
    VALIDATION = "validation"
    PERMISSION = "permission"
    TIMEOUT = "timeout"
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    EXECUTION = "execution"


# Default retryability table — error-level eligibility only. The runtime
# also applies policy + side-effect safety before actually retrying.
DEFAULT_RETRYABLE: dict[ToolErrorType, bool] = {
    ToolErrorType.NOT_FOUND: False,
    ToolErrorType.VALIDATION: False,
    ToolErrorType.PERMISSION: False,
    ToolErrorType.TIMEOUT: True,
    ToolErrorType.TRANSIENT: True,
    ToolErrorType.PERMANENT: False,
    ToolErrorType.EXECUTION: False,
}


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """Deterministic per-tool retry policy.

    ``max_attempts`` is the maximum number of handler executions
    INCLUDING the first attempt:

        max_attempts = 3  ->  attempt 1, attempt 2, attempt 3  (3 total)

    NOT "first attempt + 3 retries = 4 attempts".

    Backoff is deterministic (no jitter):

        delay = min(
            initial_backoff_seconds * backoff_multiplier ** (retry_index - 1),
            max_backoff_seconds,
        )

    where ``retry_index`` is 1-based (1 for the first retry sleep).
    No sleep occurs before the first attempt.

    ``retryable_error_types`` narrows which error types are eligible for
    retry, independent of the per-error ``retryable`` flag. For example,
    a policy with ``retryable_error_types = {TRANSIENT}`` will not retry
    a ``TIMEOUT`` even though ``TIMEOUT`` is error-level retryable.
    """

    max_attempts: int = 1
    initial_backoff_seconds: float = 0.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 30.0
    retryable_error_types: FrozenSet[ToolErrorType] = field(
        default_factory=lambda: frozenset(
            {ToolErrorType.TIMEOUT, ToolErrorType.TRANSIENT}
        )
    )

    def __post_init__(self) -> None:
        if not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError("max_attempts must be an int >= 1")
        if (
            not isinstance(self.initial_backoff_seconds, (int, float))
            or self.initial_backoff_seconds < 0
        ):
            raise ValueError("initial_backoff_seconds must be >= 0")
        if (
            not isinstance(self.backoff_multiplier, (int, float))
            or self.backoff_multiplier < 1
        ):
            raise ValueError("backoff_multiplier must be >= 1")
        if (
            not isinstance(self.max_backoff_seconds, (int, float))
            or self.max_backoff_seconds < 0
        ):
            raise ValueError("max_backoff_seconds must be >= 0")
        if self.initial_backoff_seconds > self.max_backoff_seconds:
            raise ValueError(
                "initial_backoff_seconds must be <= max_backoff_seconds"
            )
        if not isinstance(self.retryable_error_types, frozenset):
            raise ValueError("retryable_error_types must be a frozenset")


# ---------------------------------------------------------------------------
# Tool spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """Static metadata describing a tool.

    ``input_schema`` is a small JSON-schema-like dict validated by the
    injected ``ToolArgumentValidator``. ``required_permissions`` declares
    what permissions the tool needs; the runtime checks these against the
    per-call ``ToolExecutionContext.granted_permissions``.
    ``retry_policy`` controls per-tool deterministic retry; the default
    ``max_attempts=1`` preserves Phase 2 Step 1 single-attempt behaviour.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    required_permissions: FrozenSet[str]
    timeout_seconds: float
    side_effect: ToolSideEffect
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    source: ToolSource = ToolSource.LOCAL
    provider_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("ToolSpec.name must be non-empty")
        if not self.description or not self.description.strip():
            raise ValueError("ToolSpec.description must be non-empty")
        if not isinstance(self.input_schema, dict):
            raise ValueError("ToolSpec.input_schema must be a dict")
        if not isinstance(self.required_permissions, frozenset):
            raise ValueError("ToolSpec.required_permissions must be a frozenset")
        if not all(isinstance(p, str) and p for p in self.required_permissions):
            raise ValueError("required_permissions must be non-empty strings")
        if not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0:
            raise ValueError("ToolSpec.timeout_seconds must be > 0")
        if not isinstance(self.side_effect, ToolSideEffect):
            raise ValueError("ToolSpec.side_effect must be a ToolSideEffect")
        if not isinstance(self.retry_policy, RetryPolicy):
            raise ValueError("ToolSpec.retry_policy must be a RetryPolicy")
        if not isinstance(self.source, ToolSource):
            raise ValueError("ToolSpec.source must be a ToolSource")
        # Invariant: LOCAL -> provider_id is None; MCP -> provider_id non-empty.
        if self.source is ToolSource.LOCAL:
            if self.provider_id is not None:
                raise ValueError(
                    "ToolSpec.provider_id must be None when source is LOCAL"
                )
        elif self.source is ToolSource.MCP:
            if not isinstance(self.provider_id, str) or not self.provider_id:
                raise ValueError(
                    "ToolSpec.provider_id must be a non-empty str when source is MCP"
                )


# ---------------------------------------------------------------------------
# Tool handler contract
# ---------------------------------------------------------------------------


class ToolHandler(Protocol):
    """Async tool handler contract.

    A handler receives the validated arguments dict and returns any
    JSON-ish value (str / dict / list / number / bool / None). To signal
    a controlled failure it raises ``ToolReportedFailure``; any other
    exception is treated as an ordinary EXECUTION failure by the
    runtime.
    """

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        ...


@dataclass(frozen=True)
class RegisteredTool:
    """A spec + handler pair stored in the registry."""

    spec: ToolSpec
    handler: ToolHandler


# ---------------------------------------------------------------------------
# Invocation input & context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation request.

    Deliberately carries no retry count, timeout attempts or LLM
    messages — those concerns live elsewhere.
    """

    tool_name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.tool_name or not self.tool_name.strip():
            raise ValueError("ToolCall.tool_name must be non-empty")
        if not isinstance(self.arguments, dict):
            raise ValueError("ToolCall.arguments must be a dict")


@dataclass(frozen=True)
class ToolExecutionContext:
    """Per-call execution context.

    ``granted_permissions`` is what the caller has been granted for this
    invocation; the runtime checks ``spec.required_permissions <=
    granted_permissions``.
    """

    granted_permissions: FrozenSet[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not isinstance(self.granted_permissions, frozenset):
            raise ValueError("granted_permissions must be a frozenset")


# ---------------------------------------------------------------------------
# Structured errors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolError:
    """Structured tool error data.

    This is data, not runtime control flow. The runtime produces it and
    nests it inside a failed ``ToolExecutionResult``.
    """

    error_type: ToolErrorType
    message: str
    retryable: bool
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.error_type, ToolErrorType):
            raise ValueError("error_type must be a ToolErrorType")
        if not isinstance(self.message, str):
            raise ValueError("message must be a str")
        if not isinstance(self.retryable, bool):
            raise ValueError("retryable must be a bool")
        if not isinstance(self.details, dict):
            raise ValueError("details must be a dict")


# ---------------------------------------------------------------------------
# Normalized result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolExecutionResult:
    """Normalized result of a single tool invocation (with retry metadata).

    Invariants:

    * ``success=True``  -> ``error is None``
    * ``success=False`` -> ``error is not None``
    * ``attempt_count >= 0``

    ``attempt_count`` semantics:

    * preflight failure (NOT_FOUND / VALIDATION / PERMISSION) -> 0
      (handler never executed)
    * handler success or handler failure -> >= 1

    ``retry_history`` holds the errors that caused the runtime to decide
    to retry. The final error (on a failed result) is in ``error`` and
    is NOT duplicated in ``retry_history``.

    ``success=True`` with ``attempt_count=0`` is not produced by the
    runtime (a handler success always has ``attempt_count >= 1``) but is
    not forbidden by the data model itself.
    """

    tool_name: str
    success: bool
    output: Any = None
    error: Optional[ToolError] = None
    attempt_count: int = 0
    retry_history: tuple[ToolError, ...] = ()

    def __post_init__(self) -> None:
        if not self.tool_name or not self.tool_name.strip():
            raise ValueError("tool_name must be non-empty")
        if self.success and self.error is not None:
            raise ValueError("a successful result must not carry an error")
        if not self.success and self.error is None:
            raise ValueError("a failed result must carry an error")
        if not isinstance(self.attempt_count, int) or self.attempt_count < 0:
            raise ValueError("attempt_count must be an int >= 0")
        if not isinstance(self.retry_history, tuple):
            raise ValueError("retry_history must be a tuple")
