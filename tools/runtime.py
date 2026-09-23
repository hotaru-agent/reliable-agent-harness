"""Async tool runtime with deterministic retry (Phase 2 Step 1 + Step 2).

Every tool call flows through one unified boundary:

    preflight (once):
        lookup
          -> NOT_FOUND (structured, attempt_count=0)
        validate arguments
          -> VALIDATION (structured, attempt_count=0)
        permission check
          -> PERMISSION (structured, attempt_count=0)

    handler attempts (with retry):
        execute ONCE under per-attempt timeout
          -> success        -> ToolExecutionResult(success=True, attempt_count=N)
          -> timeout        -> TIMEOUT (structured)
          -> ToolReportedFailure -> reported error (structured)
          -> ordinary Exception   -> EXECUTION (structured)
          -> CancelledError       -> re-raised (external cancellation)

        on failure:
          should_retry(spec, error, attempt_number)?
            no  -> return failure (attempt_count=N, retry_history=...)
            yes -> record error in retry_history
                   sleep(compute_backoff(...))   [injectable]
                   next attempt

Safety guarantees:

* the handler is called AT MOST ``policy.max_attempts`` times;
* SIDE_EFFECTING tools are never automatically retried, even on
  retryable errors (TIMEOUT / TRANSIENT);
* ``retryable=True`` alone does not trigger retry — the policy's
  ``retryable_error_types`` and side-effect safety must also allow it;
* external ``asyncio.CancelledError`` is never converted into a
  structured result — it propagates out, whether during a handler
  attempt or during backoff sleep;
* every attempt receives a fresh deep copy of the canonical argument
  snapshot, so handler mutation of its argument dict cannot affect the
  next retry or leak back to the caller;
* timeout (per-attempt runtime deadline) and external cancellation are
  distinct.

The runtime does NOT touch Harness state (Task / Run / Checkpoint). It
is an independent, testable reliability boundary for a single tool
invocation.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any, Awaitable, Callable, Optional

from opentelemetry.trace import Tracer
from opentelemetry.trace.status import Status, StatusCode

from observability.tracing import (
    ATTR_TOOL_ATTEMPT_COUNT,
    ATTR_TOOL_ATTEMPT_MAX,
    ATTR_TOOL_ATTEMPT_NUMBER,
    ATTR_TOOL_ERROR_RETRYABLE,
    ATTR_TOOL_ERROR_TYPE,
    ATTR_TOOL_NAME,
    ATTR_TOOL_PROVIDER_ID,
    ATTR_TOOL_RESULT_SUCCESS,
    ATTR_TOOL_RETRY_BACKOFF_SECONDS,
    ATTR_TOOL_RETRY_NEXT_ATTEMPT,
    ATTR_TOOL_SIDE_EFFECT,
    ATTR_TOOL_SOURCE,
    EVENT_RETRY_SCHEDULED,
    EVENT_TOOL_CANCELLED,
    SPAN_TOOL_ATTEMPT,
    SPAN_TOOL_EXECUTE,
    resolve_tracer,
    safe_add_event,
    safe_record_exception,
    safe_set_attribute,
    safe_set_status,
    safe_span,
)
from tools.errors import ToolReportedFailure
from tools.models import (
    DEFAULT_RETRYABLE,
    ToolCall,
    ToolError,
    ToolErrorType,
    ToolExecutionContext,
    ToolExecutionResult,
    ToolSource,
)
from tools.registry import ToolRegistry
from tools.retry import compute_backoff, should_retry
from tools.validator import SchemaError, SimpleToolArgumentValidator, ValidationError

# Injectable sleep function: (float) -> Awaitable[None].
SleepFn = Callable[[float], Awaitable[None]]


class ToolRuntime:
    """Async tool runtime with deterministic retry.

    Collaborators are injected: a ``ToolRegistry``, a
    ``ToolArgumentValidator`` (defaults to ``SimpleToolArgumentValidator``),
    a ``sleep`` function (defaults to ``asyncio.sleep``), and an optional
    ``tracer`` for OpenTelemetry tracing. When no tracer is injected, a
    no-op tracer is used so runtime behaviour is unaffected.

    Telemetry failure never becomes business failure.
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        validator: SimpleToolArgumentValidator | None = None,
        sleep: SleepFn | None = None,
        tracer: Optional[Tracer] = None,
    ) -> None:
        self._registry = registry
        self._validator = validator or SimpleToolArgumentValidator()
        self._sleep: SleepFn = sleep or asyncio.sleep
        self._tracer = resolve_tracer(tracer)

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        """Execute a tool call through the unified pipeline with retry."""
        name = call.tool_name

        with safe_span(self._tracer, SPAN_TOOL_EXECUTE) as exec_span:
            # ------------------------------------------------------------------
            # Preflight — done once, before any handler attempt.
            # ------------------------------------------------------------------
            registered = self._registry.get(name)
            if registered is None:
                safe_set_attribute(exec_span, ATTR_TOOL_NAME, name)
                safe_set_attribute(exec_span, ATTR_TOOL_RESULT_SUCCESS, False)
                safe_set_attribute(exec_span, ATTR_TOOL_ATTEMPT_COUNT, 0)
                safe_set_attribute(exec_span, ATTR_TOOL_ERROR_TYPE, ToolErrorType.NOT_FOUND.value)
                safe_set_attribute(exec_span, ATTR_TOOL_ERROR_RETRYABLE, False)
                safe_set_status(exec_span, Status(StatusCode.ERROR))
                return self._preflight_failure(
                    name, ToolErrorType.NOT_FOUND,
                    f"tool {name!r} is not registered",
                )
            spec = registered.spec
            handler = registered.handler

            # Set tool identity attributes on the execute span.
            safe_set_attribute(exec_span, ATTR_TOOL_NAME, name)
            safe_set_attribute(exec_span, ATTR_TOOL_SIDE_EFFECT, spec.side_effect.value)
            safe_set_attribute(exec_span, ATTR_TOOL_SOURCE, spec.source.value)
            if spec.source is ToolSource.MCP and spec.provider_id:
                safe_set_attribute(exec_span, ATTR_TOOL_PROVIDER_ID, spec.provider_id)
            safe_set_attribute(exec_span, ATTR_TOOL_ATTEMPT_MAX, spec.retry_policy.max_attempts)

            try:
                self._validator.validate(spec.input_schema, call.arguments)
            except ValidationError as exc:
                self._set_preflight_failure_telemetry(
                    exec_span, ToolErrorType.VALIDATION
                )
                return self._preflight_failure(
                    name, ToolErrorType.VALIDATION,
                    str(exc),
                    details={"path": exc.path},
                )
            except SchemaError as exc:
                self._set_preflight_failure_telemetry(
                    exec_span, ToolErrorType.EXECUTION
                )
                return self._preflight_failure(
                    name, ToolErrorType.EXECUTION,
                    f"invalid tool schema: {exc}",
                )

            if not spec.required_permissions <= context.granted_permissions:
                missing = sorted(spec.required_permissions - context.granted_permissions)
                self._set_preflight_failure_telemetry(
                    exec_span, ToolErrorType.PERMISSION
                )
                return self._preflight_failure(
                    name, ToolErrorType.PERMISSION,
                    f"missing required permissions: {missing}",
                    details={"missing": missing},
                )

            # ------------------------------------------------------------------
            # Handler attempts with retry.
            # ------------------------------------------------------------------
            # Canonical argument snapshot — all attempts use logically identical
            # arguments. Each attempt receives a fresh deep copy.
            canonical_args = copy.deepcopy(call.arguments)

            retry_history: list[ToolError] = []
            attempt = 0

            while True:
                attempt += 1
                attempt_args = copy.deepcopy(canonical_args)

                with safe_span(self._tracer, SPAN_TOOL_ATTEMPT) as attempt_span:
                    safe_set_attribute(attempt_span, ATTR_TOOL_NAME, name)
                    safe_set_attribute(attempt_span, ATTR_TOOL_ATTEMPT_NUMBER, attempt)
                    safe_set_attribute(attempt_span, ATTR_TOOL_ATTEMPT_MAX, spec.retry_policy.max_attempts)

                    error, output = await self._execute_handler_once(
                        name, spec, handler, attempt_args, attempt_span
                    )

                    if error is None:
                        safe_set_status(attempt_span, Status(StatusCode.OK))
                    else:
                        safe_set_attribute(attempt_span, ATTR_TOOL_ERROR_TYPE, error.error_type.value)
                        safe_set_attribute(attempt_span, ATTR_TOOL_ERROR_RETRYABLE, error.retryable)
                        safe_set_status(attempt_span, Status(StatusCode.ERROR))

                if error is None:
                    # Success.
                    safe_set_attribute(exec_span, ATTR_TOOL_RESULT_SUCCESS, True)
                    safe_set_attribute(exec_span, ATTR_TOOL_ATTEMPT_COUNT, attempt)
                    safe_set_status(exec_span, Status(StatusCode.OK))
                    return ToolExecutionResult(
                        tool_name=name,
                        success=True,
                        output=output,
                        error=None,
                        attempt_count=attempt,
                        retry_history=tuple(retry_history),
                    )

                # Failure — check retry eligibility.
                if not should_retry(
                    spec=spec, error=error, attempt_number=attempt
                ):
                    safe_set_attribute(exec_span, ATTR_TOOL_RESULT_SUCCESS, False)
                    safe_set_attribute(exec_span, ATTR_TOOL_ATTEMPT_COUNT, attempt)
                    safe_set_attribute(exec_span, ATTR_TOOL_ERROR_TYPE, error.error_type.value)
                    safe_set_attribute(exec_span, ATTR_TOOL_ERROR_RETRYABLE, error.retryable)
                    safe_set_status(exec_span, Status(StatusCode.ERROR))
                    return ToolExecutionResult(
                        tool_name=name,
                        success=False,
                        output=None,
                        error=error,
                        attempt_count=attempt,
                        retry_history=tuple(retry_history),
                    )

                # Retry allowed — record the error in history, backoff, then loop.
                retry_history.append(error)
                delay = compute_backoff(
                    spec.retry_policy, retry_index=len(retry_history)
                )
                # Emit retry event ONLY when retry is actually scheduled.
                safe_add_event(
                    exec_span,
                    EVENT_RETRY_SCHEDULED,
                    {
                        ATTR_TOOL_ATTEMPT_NUMBER: attempt,
                        ATTR_TOOL_RETRY_NEXT_ATTEMPT: attempt + 1,
                        ATTR_TOOL_RETRY_BACKOFF_SECONDS: delay,
                        ATTR_TOOL_ERROR_TYPE: error.error_type.value,
                    },
                )
                # sleep may raise CancelledError (external cancellation during
                # backoff) — it propagates out, never converted to a result.
                try:
                    await self._sleep(delay)
                except asyncio.CancelledError:
                    safe_add_event(exec_span, EVENT_TOOL_CANCELLED, {})
                    safe_set_status(exec_span, Status(StatusCode.UNSET))
                    raise

    # ------------------------------------------------------------------
    # Single handler attempt under timeout.
    # ------------------------------------------------------------------

    async def _execute_handler_once(
        self,
        name: str,
        spec: "ToolSpec",  # noqa: F821
        handler: "ToolHandler",  # noqa: F821
        arguments: dict[str, Any],
        attempt_span: Any,
    ) -> tuple["ToolError | None", Any]:
        """Execute the handler once under a per-attempt timeout.

        Returns ``(None, output)`` on success or ``(error, None)`` on
        failure. ``asyncio.CancelledError`` is re-raised (never converted).

        ``attempt_span`` receives exception recording for ordinary
        exceptions; ``ToolReportedFailure`` and timeout are structured
        failures (no ``record_exception``).
        """
        try:
            output = await asyncio.wait_for(
                handler(arguments),
                timeout=spec.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return self._make_error(
                ToolErrorType.TIMEOUT,
                f"tool {name!r} exceeded timeout of {spec.timeout_seconds}s",
            ), None
        except asyncio.CancelledError:
            safe_add_event(attempt_span, EVENT_TOOL_CANCELLED, {})
            safe_set_status(attempt_span, Status(StatusCode.UNSET))
            raise
        except ToolReportedFailure as exc:
            return self._make_error(
                exc.error_type,
                exc.message,
                details=dict(exc.details),
            ), None
        except Exception as exc:
            safe_record_exception(attempt_span, exc)
            return self._make_error(
                ToolErrorType.EXECUTION,
                f"{type(exc).__name__}: {exc}",
            ), None

        return None, output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_preflight_failure_telemetry(
        exec_span: Any,
        error_type: ToolErrorType,
    ) -> None:
        """Set execute-span attributes/status for a preflight failure."""
        safe_set_attribute(exec_span, ATTR_TOOL_RESULT_SUCCESS, False)
        safe_set_attribute(exec_span, ATTR_TOOL_ATTEMPT_COUNT, 0)
        safe_set_attribute(exec_span, ATTR_TOOL_ERROR_TYPE, error_type.value)
        safe_set_attribute(
            exec_span, ATTR_TOOL_ERROR_RETRYABLE, DEFAULT_RETRYABLE[error_type]
        )
        safe_set_status(exec_span, Status(StatusCode.ERROR))

    @staticmethod
    def _make_error(
        error_type: ToolErrorType,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> ToolError:
        return ToolError(
            error_type=error_type,
            message=message,
            retryable=DEFAULT_RETRYABLE[error_type],
            details=details or {},
        )

    @classmethod
    def _preflight_failure(
        cls,
        tool_name: str,
        error_type: ToolErrorType,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> ToolExecutionResult:
        """Build a preflight failure result (attempt_count=0, no history)."""
        return ToolExecutionResult(
            tool_name=tool_name,
            success=False,
            output=None,
            error=cls._make_error(error_type, message, details=details),
            attempt_count=0,
            retry_history=(),
        )
