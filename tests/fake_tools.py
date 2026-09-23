"""Deterministic async fake tools for Phase 2 tests.

All fakes are offline: no network, no real filesystem, no real
side effects. Each fake records how many times its handler was called
so tests can assert the single-attempt or retry guarantees.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tools.errors import ToolReportedFailure
from tools.models import ToolErrorType


class _CallRecorder:
    """Mixin tracking handler invocation count."""

    def __init__(self) -> None:
        self.call_count: int = 0
        self.last_arguments: dict[str, Any] | None = None

    def _record(self, arguments: dict[str, Any]) -> None:
        self.call_count += 1
        self.last_arguments = dict(arguments)


class EchoTool(_CallRecorder):
    """Successful tool that echoes the ``text`` argument."""

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        self._record(arguments)
        return arguments["text"]


class FieldEchoTool(_CallRecorder):
    """Successful tool that echoes a configurable argument field."""

    def __init__(self, field: str) -> None:
        super().__init__()
        self.field = field

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        self._record(arguments)
        return arguments[self.field]


class SuccessTool(_CallRecorder):
    """Successful tool that returns a fixed value regardless of arguments."""

    def __init__(self, value: Any = "ok") -> None:
        super().__init__()
        self.value = value

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        self._record(arguments)
        return self.value


class SlowTool(_CallRecorder):
    """Tool that sleeps longer than its declared timeout."""

    def __init__(self, delay: float = 1.0) -> None:
        super().__init__()
        self.delay = delay

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        self._record(arguments)
        await asyncio.sleep(self.delay)
        return "should-not-reach"


class RaisingTool(_CallRecorder):
    """Tool that raises an ordinary (non-ToolReportedFailure) exception."""

    def __init__(self, exc: BaseException | None = None) -> None:
        super().__init__()
        self.exc = exc or RuntimeError("boom")

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        self._record(arguments)
        raise self.exc


class TransientFailureTool(_CallRecorder):
    """Tool that reports a TRANSIENT (retryable) failure."""

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        self._record(arguments)
        raise ToolReportedFailure(
            ToolErrorType.TRANSIENT,
            "transient hiccup",
            details={"attempt": self.call_count},
        )


class PermanentFailureTool(_CallRecorder):
    """Tool that reports a PERMANENT (non-retryable) failure."""

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        self._record(arguments)
        raise ToolReportedFailure(
            ToolErrorType.PERMANENT,
            "permanent failure",
        )


class ExecutionReportedTool(_CallRecorder):
    """Tool that reports an EXECUTION failure via ToolReportedFailure."""

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        self._record(arguments)
        raise ToolReportedFailure(
            ToolErrorType.EXECUTION,
            "reported execution failure",
        )


class CooperativeCancelTool(_CallRecorder):
    """Tool that awaits a long sleep so the test can cancel the task.

    Records the call so tests can confirm the handler was entered.
    """

    def __init__(self, delay: float = 5.0) -> None:
        super().__init__()
        self.delay = delay

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        self._record(arguments)
        await asyncio.sleep(self.delay)
        return "should-not-reach"


# ---------------------------------------------------------------------------
# Phase 2 Step 2 fakes
# ---------------------------------------------------------------------------


class ScriptedTool(_CallRecorder):
    """Tool whose behaviour is scripted per attempt.

    ``script`` is a list of outcomes applied in order (one per attempt).
    Each outcome is either:

    * ``("success", value)``        -> return value
    * ``("transient", message)``    -> raise ToolReportedFailure(TRANSIENT)
    * ``("permanent", message)``    -> raise ToolReportedFailure(PERMANENT)
    * ``("execution", message)``    -> raise ToolReportedFailure(EXECUTION)
    * ``("raise", exception)``      -> raise an ordinary exception
    * ``("timeout",)``              -> sleep forever (will hit timeout)

    If the handler is called more times than the script has entries, the
    last entry is repeated.
    """

    def __init__(self, script: list[tuple]) -> None:
        super().__init__()
        self.script = script

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        self._record(arguments)
        idx = min(self.call_count - 1, len(self.script) - 1)
        action = self.script[idx]
        kind = action[0]

        if kind == "success":
            return action[1]
        if kind == "transient":
            raise ToolReportedFailure(ToolErrorType.TRANSIENT, action[1])
        if kind == "permanent":
            raise ToolReportedFailure(ToolErrorType.PERMANENT, action[1])
        if kind == "execution":
            raise ToolReportedFailure(ToolErrorType.EXECUTION, action[1])
        if kind == "raise":
            raise action[1]
        if kind == "timeout":
            await asyncio.sleep(100)
            return "should-not-reach"
        raise ValueError(f"unknown scripted action: {kind}")


class MutatingTool(_CallRecorder):
    """Tool that mutates its argument dict, then raises TRANSIENT.

    Used to prove that each retry attempt receives a fresh argument copy
    and that the caller's original dict is not polluted.
    """

    def __init__(self, key: str = "value", new_value: Any = 999) -> None:
        super().__init__()
        self.key = key
        self.new_value = new_value

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        self._record(arguments)
        arguments[self.key] = self.new_value
        raise ToolReportedFailure(ToolErrorType.TRANSIENT, "mutated then failed")


class FakeSleeper:
    """Records requested sleep delays and returns immediately.

    Use as the injected ``sleep`` for ``ToolRuntime`` so retry tests
    don't actually wait.
    """

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)
        # Return immediately — no real sleep.

    @property
    def call_count(self) -> int:
        return len(self.delays)


class ControllableSleeper:
    """Sleeper that blocks on an event so tests can cancel during backoff.

    The first ``n_blocking`` calls block until the test cancels the task;
    subsequent calls return immediately. ``delays`` records all requests.
    """

    def __init__(self, block_forever: bool = True) -> None:
        self.delays: list[float] = []
        self.block_forever = block_forever

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)
        if self.block_forever:
            await asyncio.sleep(100)  # blocks until cancelled
