"""Filesystem tool specs and handlers (Phase 7 Step 6).

Real filesystem tool handlers for ``FilesystemRepository``:
- ``list_files`` — READ_ONLY
- ``read_file`` — READ_ONLY
- ``write_file`` — IDEMPOTENT
- ``run_tests`` — READ_ONLY (real pytest subprocess)

Tool contract names are consistent with the controlled benchmark so
script / reliability semantics are comparable.

Fault injection is supported for ``run_tests`` via the same
``FaultInjector`` pattern: the fault is injected BEFORE the pytest
subprocess spawns, so a transient fault on attempt 1 does NOT leave
an orphan child process.
"""

from __future__ import annotations

from typing import Any, Optional

from evaluation.filesystem_repository import (
    FilesystemFileNotFoundError,
    FilesystemPathError,
    FilesystemRepository,
    PytestSubprocessResult,
)
from evaluation.faults import FaultInjector, FaultType, InjectedProcessInterruption
from tools import RetryPolicy
from tools.models import ToolSideEffect, ToolSpec


# ---------------------------------------------------------------------------
# Tool specs
# ---------------------------------------------------------------------------


def make_filesystem_tool_specs(
    *,
    run_tests_retry_policy: RetryPolicy | None = None,
    run_tests_timeout_seconds: float = 30.0,
) -> dict[str, ToolSpec]:
    """Return the four filesystem tool specs keyed by name.

    Side-effect classifications (consistent with controlled benchmark):
    - ``list_files`` → READ_ONLY
    - ``read_file`` → READ_ONLY
    - ``write_file`` → IDEMPOTENT
    - ``run_tests`` → READ_ONLY

    ``run_tests`` has a longer default timeout (30s) because real
    pytest subprocess startup is slower than the in-memory oracle.
    """
    return {
        "list_files": ToolSpec(
            name="list_files",
            description="List all files in the repository.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            required_permissions=frozenset(),
            timeout_seconds=10.0,
            side_effect=ToolSideEffect.READ_ONLY,
            retry_policy=RetryPolicy(max_attempts=1),
        ),
        "read_file": ToolSpec(
            name="read_file",
            description="Read the content of a file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            required_permissions=frozenset(),
            timeout_seconds=10.0,
            side_effect=ToolSideEffect.READ_ONLY,
            retry_policy=RetryPolicy(max_attempts=1),
        ),
        "write_file": ToolSpec(
            name="write_file",
            description="Write content to a file (creates or overwrites).",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            required_permissions=frozenset(),
            timeout_seconds=10.0,
            side_effect=ToolSideEffect.IDEMPOTENT,
            retry_policy=RetryPolicy(max_attempts=1),
        ),
        "run_tests": ToolSpec(
            name="run_tests",
            description="Run the repository test suite via real pytest subprocess.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            required_permissions=frozenset(),
            timeout_seconds=run_tests_timeout_seconds,
            side_effect=ToolSideEffect.READ_ONLY,
            retry_policy=run_tests_retry_policy
            or RetryPolicy(
                max_attempts=3,
                initial_backoff_seconds=0.01,
                backoff_multiplier=2.0,
                max_backoff_seconds=0.1,
            ),
        ),
    }


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


class _FilesystemListFilesHandler:
    """Handler for ``list_files`` — real filesystem listing.

    Supports fault injection via an optional ``FaultInjector`` (for
    PROCESS_INTERRUPTION on the first logical action).
    """

    def __init__(
        self,
        repo: FilesystemRepository,
        injector: Optional[FaultInjector] = None,
    ) -> None:
        self._repo = repo
        self._injector = injector

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        if self._injector is not None:
            inv = self._injector.current_invocation("list_files")
            await self._injector.maybe_inject("list_files", inv)
        files = self._repo.list_files()
        return {"files": list(files)}


class _FilesystemReadFileHandler:
    """Handler for ``read_file`` — real UTF-8 read.

    Supports fault injection via an optional ``FaultInjector`` (for
    PROCESS_INTERRUPTION on read_file logical #2 in the recovery
    scenario).
    """

    def __init__(
        self,
        repo: FilesystemRepository,
        injector: Optional[FaultInjector] = None,
    ) -> None:
        self._repo = repo
        self._injector = injector

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        if self._injector is not None:
            inv = self._injector.current_invocation("read_file")
            await self._injector.maybe_inject("read_file", inv)
        path = arguments["path"]
        try:
            content = self._repo.read_file(path)
        except FilesystemFileNotFoundError:
            from tools.errors import ToolReportedFailure
            from tools.models import ToolErrorType
            raise ToolReportedFailure(
                error_type=ToolErrorType.NOT_FOUND,
                message=f"file not found: {path!r}",
            )
        except FilesystemPathError as e:
            from tools.errors import ToolReportedFailure
            from tools.models import ToolErrorType
            raise ToolReportedFailure(
                error_type=ToolErrorType.VALIDATION,
                message=str(e),
            )
        return {"path": path, "content": content}


class _FilesystemWriteFileHandler:
    """Handler for ``write_file`` — real UTF-8 write with \\n."""

    def __init__(self, repo: FilesystemRepository) -> None:
        self._repo = repo

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        path = arguments["path"]
        content = arguments["content"]
        try:
            self._repo.write_file(path, content)
        except FilesystemPathError as e:
            from tools.errors import ToolReportedFailure
            from tools.models import ToolErrorType
            raise ToolReportedFailure(
                error_type=ToolErrorType.VALIDATION,
                message=str(e),
            )
        return {"path": path, "bytes_written": len(content)}


class _FilesystemRunTestsHandler:
    """Handler for ``run_tests`` — real pytest subprocess.

    Supports fault injection via an optional ``FaultInjector``.

    Fault handling differs by fault type:
    - ``TRANSIENT_FAILURE`` / ``PERMANENT_FAILURE`` /
      ``PROCESS_INTERRUPTION``: raised BEFORE subprocess spawn (so
      attempt 1 does NOT leave an orphan child process).
    - ``TIMEOUT``: the fault is consumed, then the subprocess is
      spawned with ``RAH_BENCHMARK_TEST_DELAY_SECONDS`` set so the
      test module sleeps. The production ``ToolRuntime`` timeout
      boundary then cancels the handler, triggering cancellation
      cleanup. Retry attempt 2 has no delay (fault already consumed).
    - ``LARGE_OUTPUT``: returns a large payload instead of spawning.
    """

    def __init__(
        self,
        repo: FilesystemRepository,
        injector: Optional[FaultInjector] = None,
    ) -> None:
        self._repo = repo
        self._injector = injector

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        delay_seconds = 0.0
        if self._injector is not None:
            inv = self._injector.current_invocation("run_tests")
            # Use find_fault directly (consumes the fault) so we can
            # handle TIMEOUT differently from maybe_inject's sleep.
            spec = self._injector.find_fault("run_tests", inv)
            if spec is not None:
                if spec.fault_type is FaultType.LARGE_OUTPUT:
                    return FaultInjector.generate_large_output(spec)
                if spec.fault_type is FaultType.TRANSIENT_FAILURE:
                    from tools.errors import ToolReportedFailure
                    from tools.models import ToolErrorType
                    raise ToolReportedFailure(
                        error_type=ToolErrorType.TRANSIENT,
                        message="injected transient failure",
                    )
                if spec.fault_type is FaultType.PERMANENT_FAILURE:
                    from tools.errors import ToolReportedFailure
                    from tools.models import ToolErrorType
                    raise ToolReportedFailure(
                        error_type=ToolErrorType.PERMANENT,
                        message="injected permanent failure",
                    )
                if spec.fault_type is FaultType.PROCESS_INTERRUPTION:
                    raise InjectedProcessInterruption(
                        "injected process interruption"
                    )
                if spec.fault_type is FaultType.TIMEOUT:
                    # Don't sleep in the handler — set delay for the
                    # subprocess so the child is slow and the
                    # ToolRuntime timeout fires on communicate().
                    delay_seconds = spec.timeout_sleep_seconds

        # Real pytest subprocess (with optional delay for timeout fault).
        result: PytestSubprocessResult = await self._repo.run_tests(
            is_oracle=False,
            delay_seconds=delay_seconds,
        )
        return {
            "returncode": result.returncode,
            "passed": result.passed,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }


def make_filesystem_tool_handlers(
    repo: FilesystemRepository,
    injector: Optional[FaultInjector] = None,
) -> dict[str, Any]:
    """Return the four filesystem tool handlers keyed by name."""
    return {
        "list_files": _FilesystemListFilesHandler(repo, injector),
        "read_file": _FilesystemReadFileHandler(repo, injector),
        "write_file": _FilesystemWriteFileHandler(repo),
        "run_tests": _FilesystemRunTestsHandler(repo, injector),
    }
