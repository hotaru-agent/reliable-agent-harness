"""OpenTelemetry tracing foundation for the reliable agent harness.

Defines the instrumentation scope, stable low-cardinality span names,
the project-specific attribute namespace, and small safe-telemetry
helpers that isolate instrumentation failures from business logic.

This module does NOT configure a global ``TracerProvider`` and does NOT
install any exporter. Application bootstrap is responsible for SDK
configuration; core runtime modules only accept an injected ``Tracer``
(or fall back to the no-op API default when no SDK is configured).

Telemetry failure must never become business failure.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from opentelemetry.trace import Tracer, get_tracer
from opentelemetry.trace.status import Status, StatusCode

# ---------------------------------------------------------------------------
# Instrumentation scope
# ---------------------------------------------------------------------------

INSTRUMENTATION_SCOPE = "reliable_agent_harness"

# ---------------------------------------------------------------------------
# Span names — low-cardinality, stable
# ---------------------------------------------------------------------------

SPAN_HARNESS_RUN = "harness.run"
SPAN_HARNESS_STEP = "harness.step"
SPAN_TOOL_EXECUTE = "tool.execute"
SPAN_TOOL_ATTEMPT = "tool.attempt"
SPAN_AGENT_ACTION = "agent.action"
SPAN_AGENT_REPLAN_ACKNOWLEDGE = "agent.replan.acknowledge"
SPAN_CONTEXT_ASSEMBLE = "context.assemble"
SPAN_ARTIFACT_PROCESS_OUTPUT = "artifact.process_output"
SPAN_MCP_DISCOVER = "mcp.discover"

# ---------------------------------------------------------------------------
# Attribute keys — project-specific namespace
# ---------------------------------------------------------------------------

# Harness attributes
ATTR_TASK_ID = "harness.task.id"
ATTR_RUN_ID = "harness.run.id"
ATTR_RUN_STATUS = "harness.run.status"
ATTR_RUN_RESUME = "harness.run.resume"
ATTR_RESUME_FROM_CHECKPOINT_ID = "harness.resume.from_checkpoint_id"
ATTR_STEP_INDEX = "harness.step.index"
ATTR_CHECKPOINT_ID = "harness.checkpoint.id"

# Tool attributes
ATTR_TOOL_NAME = "tool.name"
ATTR_TOOL_SOURCE = "tool.source"
ATTR_TOOL_PROVIDER_ID = "tool.provider_id"
ATTR_TOOL_SIDE_EFFECT = "tool.side_effect"
ATTR_TOOL_ATTEMPT_NUMBER = "tool.attempt.number"
ATTR_TOOL_ATTEMPT_MAX = "tool.attempt.max"
ATTR_TOOL_RESULT_SUCCESS = "tool.result.success"
ATTR_TOOL_ATTEMPT_COUNT = "tool.attempt.count"
ATTR_TOOL_ERROR_TYPE = "tool.error.type"
ATTR_TOOL_ERROR_RETRYABLE = "tool.error.retryable"
ATTR_TOOL_RETRY_COUNT = "tool.retry.count"
ATTR_TOOL_RETRY_NEXT_ATTEMPT = "tool.retry.next_attempt"
ATTR_TOOL_RETRY_BACKOFF_SECONDS = "tool.retry.backoff_seconds"

# Agent action attributes
ATTR_AGENT_ACTION_INDEX = "agent.action.index"
ATTR_AGENT_ACTION_NEXT_INDEX = "agent.action.next_index"
ATTR_AGENT_ACTION_SUCCESS = "agent.action.success"
ATTR_AGENT_ACTION_LOOP_DETECTED = "agent.action.loop_detected"
ATTR_AGENT_CONTROLLER_STATE = "agent.controller.state"
ATTR_AGENT_LOOP_REASON = "agent.loop.reason"
ATTR_AGENT_LOOP_EVIDENCE_COUNT = "agent.loop.evidence_count"
ATTR_AGENT_REPLAN_EVIDENCE_COUNT = "agent.replan.evidence_count"

# Context attributes
ATTR_CONTEXT_INPUT_COUNT = "context.item.input_count"
ATTR_CONTEXT_INCLUDED_COUNT = "context.item.included_count"
ATTR_CONTEXT_OMITTED_COUNT = "context.item.omitted_count"
ATTR_CONTEXT_TOKEN_MAX = "context.token.max"
ATTR_CONTEXT_TOKEN_USED = "context.token.used"
ATTR_CONTEXT_TOKEN_REMAINING = "context.token.remaining"
ATTR_CONTEXT_MUST_KEEP_COUNT = "context.must_keep.count"
ATTR_CONTEXT_TOKEN_REQUIRED = "context.token.required"

# Artifact attributes
ATTR_ARTIFACT_EXTERNALIZED = "artifact.externalized"
ATTR_ARTIFACT_ID = "artifact.id"
ATTR_ARTIFACT_ORIGINAL_SIZE_BYTES = "artifact.original.size_bytes"
ATTR_ARTIFACT_ORIGINAL_ESTIMATED_TOKENS = "artifact.original.estimated_tokens"
ATTR_ARTIFACT_CONTEXT_ESTIMATED_TOKENS = "artifact.context.estimated_tokens"
ATTR_ARTIFACT_SOURCE_TOOL = "artifact.source_tool"

# MCP attributes
ATTR_MCP_PROVIDER_ID = "mcp.provider.id"
ATTR_MCP_DISCOVERY_TOOL_COUNT = "mcp.discovery.tool_count"
ATTR_MCP_DISCOVERY_PAGE_COUNT = "mcp.discovery.page_count"
ATTR_MCP_DISCOVERY_REGISTERED_COUNT = "mcp.discovery.registered_count"
ATTR_MCP_ANNOTATION_TRUSTED = "mcp.annotation.trusted"

# Exception attributes
ATTR_ERROR_EXCEPTION_TYPE = "error.exception.type"

# ---------------------------------------------------------------------------
# Event names
# ---------------------------------------------------------------------------

EVENT_CHECKPOINT_SAVED = "harness.checkpoint.saved"
EVENT_RUN_RESUMED = "harness.run.resumed"
EVENT_RUN_INTERRUPTED = "harness.run.interrupted"
EVENT_RETRY_SCHEDULED = "tool.retry.scheduled"
EVENT_TOOL_CANCELLED = "tool.execute.cancelled"
EVENT_AGENT_LOOP_DETECTED = "agent.loop.detected"
EVENT_AGENT_REPLAN_REQUIRED = "agent.replan.required"
EVENT_AGENT_REPLAN_BLOCKED = "agent.replan.blocked"

# ---------------------------------------------------------------------------
# Run status values
# ---------------------------------------------------------------------------

RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_FAILED = "failed"
RUN_STATUS_INTERRUPTED = "interrupted"

# ---------------------------------------------------------------------------
# Exception telemetry policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExceptionTelemetryPolicy:
    """Policy for how exception information is recorded in telemetry.

    By default, exception messages and stacktraces are NOT recorded
    because they may contain sensitive application data (tool arguments,
    passwords, tokens, file content, user data). Only the exception
    type/class identity is recorded as a safe attribute.

    Opt-in full exception recording (``record_exception_message=True``)
    may expose application data and should only be enabled in trusted
    debugging environments.
    """

    record_exception_message: bool = False
    record_stacktrace: bool = False


# Default policy: safe/conservative.
DEFAULT_EXCEPTION_POLICY = ExceptionTelemetryPolicy()


# ---------------------------------------------------------------------------
# Tracer acquisition
# ---------------------------------------------------------------------------


def get_default_tracer() -> Tracer:
    """Return the default tracer for the instrumentation scope.

    When no SDK is configured, the OpenTelemetry API returns a no-op
    tracer, so runtime behaviour is unaffected. This function is used
    only as a fallback when no tracer is explicitly injected.
    """
    return get_tracer(INSTRUMENTATION_SCOPE)


def resolve_tracer(tracer: Optional[Tracer]) -> Tracer:
    """Return ``tracer`` or the default no-op tracer if ``None``."""
    if tracer is not None:
        return tracer
    return get_default_tracer()


# ---------------------------------------------------------------------------
# Safe telemetry helpers
# ---------------------------------------------------------------------------


def safe_set_attribute(span: Any, key: str, value: Any) -> None:
    """Set a span attribute, isolating telemetry failures.

    If the span does not support ``set_attribute`` or the value is not
    a valid OTel primitive, the call is silently ignored. Business
    logic is never affected.
    """
    try:
        if value is None:
            return
        if isinstance(value, bool):
            pass
        elif isinstance(value, (int, float)):
            pass
        elif isinstance(value, str):
            pass
        else:
            return
        span.set_attribute(key, value)
    except Exception:
        pass


def safe_add_event(
    span: Any,
    name: str,
    attributes: Optional[dict[str, Any]] = None,
) -> None:
    """Add an event to a span, isolating telemetry failures.

    Only primitive attribute values are passed through; non-primitive
    values are silently dropped from the event attributes.
    """
    try:
        clean: dict[str, Any] = {}
        if attributes:
            for k, v in attributes.items():
                if v is None:
                    continue
                if isinstance(v, bool):
                    clean[k] = v
                elif isinstance(v, (int, float)):
                    clean[k] = v
                elif isinstance(v, str):
                    clean[k] = v
        span.add_event(name, attributes=clean)
    except Exception:
        pass


def safe_record_exception(
    span: Any,
    exc: BaseException,
    *,
    policy: ExceptionTelemetryPolicy = DEFAULT_EXCEPTION_POLICY,
) -> None:
    """Record an exception on a span according to the privacy policy.

    By default (``policy.record_exception_message=False``), this does
    NOT call ``span.record_exception(exc)`` because that would record
    the exception message and stacktrace, which may contain sensitive
    application data. Instead, it records only the exception type/class
    identity as a safe attribute (``error.exception.type``).

    When ``policy.record_exception_message=True``, the full exception
    is recorded via ``span.record_exception(exc)``. This is opt-in and
    may expose application data.
    """
    try:
        if policy.record_exception_message:
            span.record_exception(exc)
        # Always record the exception type as a safe attribute.
        safe_set_attribute(span, ATTR_ERROR_EXCEPTION_TYPE, type(exc).__name__)
    except Exception:
        pass


def safe_set_status(span: Any, status: Status) -> None:
    """Set span status, isolating telemetry failures."""
    try:
        span.set_status(status)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Safe span lifecycle
# ---------------------------------------------------------------------------


class _NoOpSpan:
    """A no-op span used when span creation fails.

    All telemetry operations on this object are silently ignored.
    Business exceptions raised inside the ``with safe_span(...)`` block
    are NOT caught — they propagate normally.
    """

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def add_event(
        self, name: str, attributes: Optional[dict[str, Any]] = None
    ) -> None:
        pass

    def record_exception(self, exc: BaseException) -> None:
        pass

    def set_status(self, status: Status) -> None:
        pass

    @property
    def context(self) -> Any:
        class _NoCtx:
            span_id = 0
            trace_id = 0
        return _NoCtx()

    @property
    def parent(self) -> None:
        return None

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> bool:
        return False


_NO_OP_SPAN = _NoOpSpan()


@contextmanager
def safe_span(
    tracer: Any,
    name: str,
    *,
    policy: ExceptionTelemetryPolicy = DEFAULT_EXCEPTION_POLICY,
) -> Iterator[Any]:
    """Context manager that safely creates and ends a span.

    If ``tracer.start_as_current_span(...)`` raises (e.g. a pathological
    injected tracer), a no-op span is used instead so business logic
    continues normally.

    IMPORTANT: this context manager does NOT catch business exceptions
    raised inside the ``with`` block. Only the span creation/ending
    telemetry operations are isolated. Business exceptions propagate
    normally.

    The span's ``__exit__`` is called with ``None`` to prevent the SDK
    from auto-recording business exceptions (which would leak exception
    messages into telemetry). Exception recording is handled explicitly
    by callers via ``safe_record_exception`` according to the privacy
    policy.
    """
    cm = None
    try:
        cm = tracer.start_as_current_span(name)
        span = cm.__enter__()
    except Exception:
        span = _NO_OP_SPAN
        cm = None

    try:
        yield span
    finally:
        if cm is not None:
            try:
                # Call __exit__ with no exception to prevent the SDK
                # from auto-recording business exceptions.
                cm.__exit__(None, None, None)
            except Exception:
                pass
