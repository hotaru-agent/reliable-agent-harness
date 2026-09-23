"""Controlled repository benchmark scenarios (Phase 7 Step 1).

Provides:
- ``ScriptedActionSource`` — a deterministic, non-AI action source that
  produces a fixed sequence of logical tool calls. This is NOT an AI
  agent; it is benchmark control logic.
- Repository tool suite — ``list_files``, ``read_file``, ``write_file``,
  ``run_tests`` tools that operate on a ``ControlledRepository`` and
  go through the existing ``ToolRegistry`` / ``ToolRuntime`` contract.
- Scenario fixtures — a set of benchmark scenarios with repository
  fixtures and fault plans.

All offline, deterministic, no LLM, no network, no real filesystem.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

from evaluation.faults import FaultInjector, FaultPlan, FaultSpec, FaultType
from evaluation.models import BenchmarkScenario, BenchmarkTask
from evaluation.repository import (
    ControlledRepository,
    RepositoryFileNotFoundError,
    RepositoryFixture,
    RepositoryOracle,
    TestDefinition,
)
from tools import (
    RetryPolicy,
    ToolCall,
    ToolSideEffect,
    ToolSpec,
)
from tools.errors import ToolReportedFailure
from tools.models import ToolErrorType


# ===========================================================================
# Repository tool handlers
# ===========================================================================


class _ListFilesHandler:
    """Handler for the ``list_files`` tool.

    Returns a JSON-ish dict with the sorted list of file paths.
    READ_ONLY — no side effects.
    """

    def __init__(self, repo: ControlledRepository) -> None:
        self._repo = repo

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        files = self._repo.list_files()
        return {"files": list(files), "count": len(files)}


class _ReadFileHandler:
    """Handler for the ``read_file`` tool.

    Returns the file content as a string.
    READ_ONLY — no side effects.

    Raises ``ToolReportedFailure(EXECUTION)`` if the file is not found.

    Supports fault injection via an optional ``FaultInjector``. The
    injector uses the active logical invocation identity allocated by
    ``FaultAwareToolInvoker`` so retry attempts share the same logical
    invocation index (see Phase 7 Step 1 Fix).
    """

    def __init__(
        self,
        repo: ControlledRepository,
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
        except RepositoryFileNotFoundError as exc:
            raise ToolReportedFailure(
                error_type=ToolErrorType.EXECUTION,
                message=str(exc),
            ) from exc
        return {"path": path, "content": content}


class _WriteFileHandler:
    """Handler for the ``write_file`` tool.

    Writes content to a file path. IDEMPOTENT — writing the same content
    to the same path twice produces the same resulting state.

    Returns a confirmation dict with the path and content length.
    """

    def __init__(self, repo: ControlledRepository) -> None:
        self._repo = repo

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        path = arguments["path"]
        content = arguments["content"]
        self._repo.write_file(path, content)
        return {"path": path, "bytes_written": len(content)}


class _RunTestsHandler:
    """Handler for the ``run_tests`` tool.

    Runs the deterministic in-process test suite and returns a structured
    result. READ_ONLY — running tests does not modify repository state.

    Supports fault injection via an optional ``FaultInjector``.
    """

    def __init__(
        self,
        repo: ControlledRepository,
        injector: Optional[FaultInjector] = None,
    ) -> None:
        self._repo = repo
        self._injector = injector

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        if self._injector is not None:
            # Use the active logical invocation identity allocated by
            # FaultAwareToolInvoker BEFORE ToolRuntime.execute(). This
            # ensures retry attempts do NOT advance the logical invocation
            # counter. Returns 0 if no invocation is active (safe
            # failure mode — no fault matches index 0).
            inv = self._injector.current_invocation("run_tests")
            spec = await self._injector.maybe_inject("run_tests", inv)
            if spec is not None and spec.fault_type is FaultType.LARGE_OUTPUT:
                return FaultInjector.generate_large_output(spec)

        result = self._repo.run_tests()
        return {
            "passed": result.passed,
            "total": result.total,
            "passed_count": result.passed_count,
            "failed_count": result.failed_count,
            "failing_tests": list(result.failing_tests),
        }


# ===========================================================================
# Repository tool specs
# ===========================================================================


def make_repository_tool_specs(
    *,
    run_tests_retry_policy: RetryPolicy | None = None,
    run_tests_timeout_seconds: float = 5.0,
) -> dict[str, ToolSpec]:
    """Return the four repository tool specs keyed by name.

    Side-effect classifications:
    - ``list_files`` → READ_ONLY
    - ``read_file`` → READ_ONLY
    - ``write_file`` → IDEMPOTENT (same content + same path = same state)
    - ``run_tests`` → READ_ONLY (does not modify repository state)

    ``run_tests`` has a retry policy that allows transient/timeout
    retries, so the ToolRuntime can recover from injected transient
    faults. The timeout is small (default 5s) but the fault injector's
    timeout sleep should be larger.
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
            timeout_seconds=5.0,
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
            timeout_seconds=5.0,
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
            timeout_seconds=5.0,
            side_effect=ToolSideEffect.IDEMPOTENT,
            retry_policy=RetryPolicy(max_attempts=1),
        ),
        "run_tests": ToolSpec(
            name="run_tests",
            description="Run the repository test suite.",
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


def make_repository_tool_handlers(
    repo: ControlledRepository,
    injector: Optional[FaultInjector] = None,
) -> dict[str, Any]:
    """Return the four repository tool handlers keyed by name.

    The handlers close over the given repository and optional fault
    injector. Each handler is a fresh callable — no shared mutable state
    between handler instances.
    """
    return {
        "list_files": _ListFilesHandler(repo),
        "read_file": _ReadFileHandler(repo, injector),
        "write_file": _WriteFileHandler(repo),
        "run_tests": _RunTestsHandler(repo, injector),
    }


# ===========================================================================
# Scripted action source
# ===========================================================================


@dataclass(frozen=True)
class ScriptedAction:
    """A single scripted logical action.

    Attributes:
        tool_name: The tool to call.
        arguments: The arguments dict.
        description: Human-readable description (not parsed).
    """

    tool_name: str
    arguments: dict[str, Any]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.tool_name or not self.tool_name.strip():
            raise ValueError("tool_name must be a non-empty str")
        if not isinstance(self.arguments, dict):
            raise ValueError("arguments must be a dict")
        if not isinstance(self.description, str):
            raise ValueError("description must be a str")


class ScriptedActionSource:
    """Deterministic, non-AI action source for benchmark control.

    This is NOT an AI agent. It does not reason, plan, or adapt. It
    produces a fixed sequence of ``ToolCall`` objects from a scripted
    action list. Its sole purpose is to provide deterministic inputs to
    the benchmark runner so that Harness reliability (not Agent
    reasoning quality) can be isolated and measured.

    The action source is iterable: it yields ``ToolCall`` objects one
    at a time. Once exhausted, it signals completion by raising
    ``StopIteration``.

    Usage::

        source = ScriptedActionSource(actions=[...])
        for call in source:
            result = await runtime.execute(call, ctx)
            ...

    Or equivalently::

        source = ScriptedActionSource(actions=[...])
        while source.has_next():
            call = source.next_call()
            result = await runtime.execute(call, ctx)
            ...
    """

    def __init__(self, actions: list[ScriptedAction]) -> None:
        if not isinstance(actions, list):
            raise ValueError("actions must be a list")
        if not actions:
            raise ValueError("actions must not be empty")
        self._actions: tuple[ScriptedAction, ...] = tuple(actions)
        self._index: int = 0

    @property
    def total_actions(self) -> int:
        """Total number of scripted actions."""
        return len(self._actions)

    @property
    def current_index(self) -> int:
        """0-based index of the next action to produce."""
        return self._index

    def has_next(self) -> bool:
        """Return True if there are more actions to produce."""
        return self._index < len(self._actions)

    def next_call(self) -> ToolCall:
        """Return the next ``ToolCall`` and advance the internal pointer.

        Raises ``StopIteration`` if exhausted.
        """
        if not self.has_next():
            raise StopIteration("scripted action source exhausted")
        action = self._actions[self._index]
        self._index += 1
        return ToolCall(tool_name=action.tool_name, arguments=dict(action.arguments))

    def reset(self) -> None:
        """Reset the internal pointer to the beginning."""
        self._index = 0

    def seek(self, index: int) -> None:
        """Set the internal pointer to ``index`` (0-based).

        Used by the benchmark recovery executor to restore the script
        cursor from a checkpointed ``ExecutionState``. ``index`` must be
        in ``[0, total_actions]``; ``total_actions`` is allowed (it
        means "exhausted", so the next ``has_next()`` returns False).
        """
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index > len(self._actions)
        ):
            raise ValueError(
                f"index must be an int in [0, {len(self._actions)}], got {index!r}"
            )
        self._index = index

    def __iter__(self):
        return self

    def __next__(self) -> ToolCall:
        return self.next_call()


# ===========================================================================
# Standard repository fixtures
# ===========================================================================


def _calculator_add_checker(files: dict[str, str]) -> bool:
    """Check that ``add(2, 3) == 5`` by executing calculator.py in-process."""
    source = files.get("calculator.py")
    if source is None:
        return False
    namespace: dict[str, Any] = {}
    try:
        exec(compile(source, "calculator.py", "exec"), namespace)
    except Exception:
        return False
    add = namespace.get("add")
    if not callable(add):
        return False
    try:
        return add(2, 3) == 5
    except Exception:
        return False


def _calculator_sub_checker(files: dict[str, str]) -> bool:
    """Check that ``sub(10, 3) == 7``."""
    source = files.get("calculator.py")
    if source is None:
        return False
    namespace: dict[str, Any] = {}
    try:
        exec(compile(source, "calculator.py", "exec"), namespace)
    except Exception:
        return False
    sub = namespace.get("sub")
    if not callable(sub):
        return False
    try:
        return sub(10, 3) == 7
    except Exception:
        return False


# The buggy initial calculator: add() is implemented as subtraction.
_BUGGY_CALCULATOR = """\
def add(a, b):
    return a - b


def sub(a, b):
    return a - b
"""

# The corrected calculator: add() is now correct.
_CORRECTED_CALCULATOR = """\
def add(a, b):
    return a + b


def sub(a, b):
    return a - b
"""

# The test file (informational only — the oracle uses in-process checkers).
_TEST_FILE = """\
from calculator import add, sub


def test_add():
    assert add(2, 3) == 5


def test_sub():
    assert sub(10, 3) == 7
"""


def make_calculator_fixture(
    *,
    fixture_id: str = "calculator_buggy_add",
    buggy: bool = True,
) -> RepositoryFixture:
    """Return the standard calculator repository fixture.

    The fixture contains:
    - ``calculator.py`` — with a buggy ``add`` (returns ``a - b``) if
      ``buggy=True``, or the corrected version if ``buggy=False``.
    - ``tests/test_calculator.py`` — informational test file.

    The oracle uses in-process deterministic checkers, NOT pytest.
    """
    return RepositoryFixture(
        fixture_id=fixture_id,
        files={
            "calculator.py": _BUGGY_CALCULATOR if buggy else _CORRECTED_CALCULATOR,
            "tests/test_calculator.py": _TEST_FILE,
        },
        test_definitions=(
            TestDefinition(name="test_add", checker=_calculator_add_checker),
            TestDefinition(name="test_sub", checker=_calculator_sub_checker),
        ),
    )


# The corrected content for write_file actions.
CORRECTED_CALCULATOR_CONTENT = _CORRECTED_CALCULATOR


# ===========================================================================
# Standard scripted action sequences
# ===========================================================================


def make_fix_calculator_actions() -> list[ScriptedAction]:
    """Return the standard scripted action sequence for fixing the
    calculator bug.

    Sequence:
    1. run_tests — observe failure
    2. read_file calculator.py — read the buggy source
    3. write_file calculator.py — write the corrected source
    4. run_tests — verify all tests pass
    """
    return [
        ScriptedAction(
            tool_name="run_tests",
            arguments={},
            description="Run tests to observe the failing test_add.",
        ),
        ScriptedAction(
            tool_name="read_file",
            arguments={"path": "calculator.py"},
            description="Read the buggy calculator source.",
        ),
        ScriptedAction(
            tool_name="write_file",
            arguments={"path": "calculator.py", "content": CORRECTED_CALCULATOR_CONTENT},
            description="Write the corrected calculator source.",
        ),
        ScriptedAction(
            tool_name="run_tests",
            arguments={},
            description="Run tests to verify all tests pass.",
        ),
    ]


# ===========================================================================
# Standard benchmark scenarios
# ===========================================================================


def make_standard_scenarios() -> dict[str, BenchmarkScenario]:
    """Return the standard set of Phase 7 Step 1 benchmark scenarios.

    Scenarios:
    - ``clean_success`` — no faults, scripted fix succeeds.
    - ``transient_failure`` — run_tests invocation #1 fails transiently.
    - ``permanent_failure`` — run_tests invocation #1 fails permanently.
    - ``timeout`` — run_tests invocation #1 times out.
    - ``large_output`` — run_tests invocation #1 returns large output.
    - ``interruption`` — run_tests invocation #1 triggers process
      interruption.
    """
    scenarios: dict[str, BenchmarkScenario] = {}

    # Common task for all scenarios.
    task = BenchmarkTask(
        task_id="fix_calculator_add",
        goal="Fix the buggy add() function in calculator.py so all tests pass.",
        scenario_id="clean_success",
        max_logical_actions=10,
    )
    scenarios["clean_success"] = BenchmarkScenario(
        scenario_id="clean_success",
        task=task,
        fixture_id="calculator_buggy_add",
        fault_plan_id="none",
        description="Clean run with no injected faults.",
    )

    # Transient failure scenario.
    task_t = BenchmarkTask(
        task_id="fix_calculator_add",
        goal="Fix the buggy add() function in calculator.py so all tests pass.",
        scenario_id="transient_failure",
        max_logical_actions=10,
    )
    scenarios["transient_failure"] = BenchmarkScenario(
        scenario_id="transient_failure",
        task=task_t,
        fixture_id="calculator_buggy_add",
        fault_plan_id="transient_run_tests_1",
        description="run_tests invocation #1 fails transiently; retry should succeed.",
    )

    # Permanent failure scenario.
    task_p = BenchmarkTask(
        task_id="fix_calculator_add",
        goal="Fix the buggy add() function in calculator.py so all tests pass.",
        scenario_id="permanent_failure",
        max_logical_actions=10,
    )
    scenarios["permanent_failure"] = BenchmarkScenario(
        scenario_id="permanent_failure",
        task=task_p,
        fixture_id="calculator_buggy_add",
        fault_plan_id="permanent_run_tests_1",
        description="run_tests invocation #1 fails permanently; no retry helps.",
    )

    # Timeout scenario.
    task_to = BenchmarkTask(
        task_id="fix_calculator_add",
        goal="Fix the buggy add() function in calculator.py so all tests pass.",
        scenario_id="timeout",
        max_logical_actions=10,
    )
    scenarios["timeout"] = BenchmarkScenario(
        scenario_id="timeout",
        task=task_to,
        fixture_id="calculator_buggy_add",
        fault_plan_id="timeout_run_tests_1",
        description="run_tests invocation #1 times out; retry should succeed.",
    )

    # Large output scenario.
    task_l = BenchmarkTask(
        task_id="fix_calculator_add",
        goal="Fix the buggy add() function in calculator.py so all tests pass.",
        scenario_id="large_output",
        max_logical_actions=10,
    )
    scenarios["large_output"] = BenchmarkScenario(
        scenario_id="large_output",
        task=task_l,
        fixture_id="calculator_buggy_add",
        fault_plan_id="large_output_run_tests_1",
        description="run_tests invocation #1 returns a large output payload.",
    )

    # Interruption scenario.
    task_i = BenchmarkTask(
        task_id="fix_calculator_add",
        goal="Fix the buggy add() function in calculator.py so all tests pass.",
        scenario_id="interruption",
        max_logical_actions=10,
    )
    scenarios["interruption"] = BenchmarkScenario(
        scenario_id="interruption",
        task=task_i,
        fixture_id="calculator_buggy_add",
        fault_plan_id="interruption_run_tests_1",
        description="run_tests invocation #1 triggers a process interruption signal.",
    )

    # Checkpoint recovery scenario (Phase 7 Step 3).
    # Interruption occurs on read_file logical invocation #1, AFTER
    # step 0 (run_tests) has succeeded and been checkpointed. The
    # interruption occurs before the repository is repaired (read_file
    # is before write_file), so the oracle is still incomplete at the
    # interruption point. This validates resume-from-checkpoint.
    task_cr = BenchmarkTask(
        task_id="fix_calculator_add",
        goal="Fix the buggy add() function in calculator.py so all tests pass.",
        scenario_id="checkpoint_recovery",
        max_logical_actions=10,
    )
    scenarios["checkpoint_recovery"] = BenchmarkScenario(
        scenario_id="checkpoint_recovery",
        task=task_cr,
        fixture_id="calculator_buggy_add",
        fault_plan_id="interruption_read_file_1",
        description=(
            "read_file invocation #1 triggers a process interruption "
            "after step 0 (run_tests) succeeded and checkpointed. "
            "Reliable runner resumes from the checkpoint and completes."
        ),
    )

    # Loop replan scenario (Phase 7 Step 4).
    # No fault injection — this isolates loop detection / replan / gate.
    # The deterministic action policy repeatedly chooses
    # read_file("calculator.py") with no progress. After the third
    # duplicate, the Reliable runner's AgentActionController detects the
    # loop, gates the fourth call, applies a deterministic replan, and
    # executes the repair branch (write_file + run_tests). The Naive
    # runner has no loop detection and loops until max_logical_actions.
    # max_logical_actions=6: 3 reads + 1 write + 1 run_tests = 5 < 6.
    task_lp = BenchmarkTask(
        task_id="fix_calculator_add",
        goal="Fix the buggy add() function in calculator.py so all tests pass.",
        scenario_id="loop_replan",
        max_logical_actions=6,
    )
    scenarios["loop_replan"] = BenchmarkScenario(
        scenario_id="loop_replan",
        task=task_lp,
        fixture_id="calculator_buggy_add",
        fault_plan_id="none",
        description=(
            "Deterministic action policy loops on read_file(calculator.py) "
            "with no progress. Reliable runner detects loop via "
            "AgentActionController, gates the next call, applies "
            "deterministic replan, and completes. Naive runner loops "
            "until max_logical_actions."
        ),
    )

    # Context growth scenario (Phase 7 Step 5).
    # No fault injection — this isolates ContextAssembler value under
    # context growth. The deterministic action policy reads calculator.py
    # and tests/test_calculator.py multiple times, accumulating context.
    # Each individual output is small (stays inline), so the Reliable
    # runner's advantage comes from ContextAssembler bounded selection,
    # not externalization. The Naive runner accumulates all outputs and
    # exceeds the context capacity before the repair action.
    task_cg = BenchmarkTask(
        task_id="fix_calculator_add",
        goal="Fix the buggy add() function in calculator.py so all tests pass.",
        scenario_id="context_growth",
        max_logical_actions=10,
    )
    scenarios["context_growth"] = BenchmarkScenario(
        scenario_id="context_growth",
        task=task_cg,
        fixture_id="calculator_buggy_add",
        fault_plan_id="none",
        description=(
            "Deterministic action policy reads calculator.py and "
            "tests/test_calculator.py multiple times, accumulating "
            "context. Reliable runner uses ContextAssembler to keep "
            "must-keep critical state and omit old observations within "
            "budget. Naive runner accumulates all outputs and exceeds "
            "context capacity before the repair action."
        ),
    )

    # Large output externalization scenario (Phase 7 Step 5).
    # No fault injection — this isolates ToolOutputProcessor +
    # ArtifactStore value. The first action reads a large log file
    # whose output exceeds the externalization threshold and context
    # capacity. The Reliable runner externalizes the large output to
    # the ArtifactStore and keeps a small reference in context. The
    # Naive runner inlines the full output and exceeds context
    # capacity immediately.
    task_lo = BenchmarkTask(
        task_id="fix_calculator_add",
        goal="Fix the buggy add() function in calculator.py so all tests pass.",
        scenario_id="large_output_externalization",
        max_logical_actions=10,
    )
    scenarios["large_output_externalization"] = BenchmarkScenario(
        scenario_id="large_output_externalization",
        task=task_lo,
        fixture_id="calculator_buggy_add_large_log",
        fault_plan_id="none",
        description=(
            "First action reads a large log file whose output exceeds "
            "the externalization threshold and context capacity. "
            "Reliable runner externalizes via ToolOutputProcessor + "
            "ArtifactStore, keeps a small reference in context, and "
            "completes. Naive runner inlines the full output and "
            "exceeds context capacity immediately."
        ),
    )

    return scenarios


# ===========================================================================
# Standard fault plans
# ===========================================================================


def make_standard_fault_plans() -> dict[str, FaultPlan]:
    """Return the standard set of fault plans for Phase 7 Step 1."""
    return {
        "none": FaultPlan.no_faults(),
        "transient_run_tests_1": FaultPlan(
            plan_id="transient_run_tests_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        ),
        "permanent_run_tests_1": FaultPlan(
            plan_id="permanent_run_tests_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.PERMANENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        ),
        "timeout_run_tests_1": FaultPlan(
            plan_id="timeout_run_tests_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TIMEOUT,
                    tool_name="run_tests",
                    logical_invocation=1,
                    timeout_sleep_seconds=10.0,
                ),
            ),
        ),
        "large_output_run_tests_1": FaultPlan(
            plan_id="large_output_run_tests_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.LARGE_OUTPUT,
                    tool_name="run_tests",
                    logical_invocation=1,
                    large_output_size=4096,
                ),
            ),
        ),
        "interruption_run_tests_1": FaultPlan(
            plan_id="interruption_run_tests_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.PROCESS_INTERRUPTION,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        ),
        "interruption_read_file_1": FaultPlan(
            plan_id="interruption_read_file_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.PROCESS_INTERRUPTION,
                    tool_name="read_file",
                    logical_invocation=1,
                ),
            ),
        ),
    }


# ===========================================================================
# Fixture registry
# ===========================================================================


def make_standard_fixtures() -> dict[str, RepositoryFixture]:
    """Return the standard set of repository fixtures."""
    return {
        "calculator_buggy_add": make_calculator_fixture(),
        "calculator_buggy_add_large_log": make_calculator_fixture_with_large_log(),
    }


# ===========================================================================
# Large log fixture (Phase 7 Step 5)
# ===========================================================================


def _make_large_log_content(*, lines: int = 200) -> str:
    """Return a deterministic large log file content.

    The content is deterministic and large enough that its rendered
    tool output (a JSON dict containing the content) exceeds both the
    externalization threshold and the context capacity used in the
    benchmark.
    """
    parts = []
    for i in range(lines):
        parts.append(f"test_log_line_{i:04d}: deterministic large log entry number {i}")
    return "\n".join(parts) + "\n"


def make_calculator_fixture_with_large_log(
    *,
    fixture_id: str = "calculator_buggy_add_large_log",
) -> RepositoryFixture:
    """Return a calculator fixture that also contains a large log file.

    The fixture contains:
    - ``calculator.py`` — buggy add (returns ``a - b``).
    - ``tests/test_calculator.py`` — informational test file.
    - ``large_test_log.txt`` — a deterministic large log file whose
      ``read_file`` output exceeds the externalization threshold and
      context capacity.
    """
    return RepositoryFixture(
        fixture_id=fixture_id,
        files={
            "calculator.py": _BUGGY_CALCULATOR,
            "tests/test_calculator.py": _TEST_FILE,
            "large_test_log.txt": _make_large_log_content(),
        },
        test_definitions=(
            TestDefinition(name="test_add", checker=_calculator_add_checker),
            TestDefinition(name="test_sub", checker=_calculator_sub_checker),
        ),
    )


# ===========================================================================
# Context growth action sequence (Phase 7 Step 5)
# ===========================================================================


def make_context_growth_actions() -> list[ScriptedAction]:
    """Return the scripted action sequence for the context_growth scenario.

    Sequence:
    1. read_file calculator.py — accumulate observation #1
    2. read_file tests/test_calculator.py — accumulate observation #2
    3. read_file calculator.py — accumulate observation #3
    4. read_file tests/test_calculator.py — accumulate observation #4
    5. read_file calculator.py — accumulate observation #5
    6. write_file calculator.py — write the corrected source
    7. run_tests — verify all tests pass

    The repair (write_file) happens AFTER context has grown through
    multiple observations. Each individual output is small enough to
    stay inline (below the externalization threshold), so the
    context_growth scenario isolates the ContextAssembler value.
    """
    return [
        ScriptedAction(
            tool_name="read_file",
            arguments={"path": "calculator.py"},
            description="Read the buggy calculator source (#1).",
        ),
        ScriptedAction(
            tool_name="read_file",
            arguments={"path": "tests/test_calculator.py"},
            description="Read the test file (#1).",
        ),
        ScriptedAction(
            tool_name="read_file",
            arguments={"path": "calculator.py"},
            description="Read the buggy calculator source (#2).",
        ),
        ScriptedAction(
            tool_name="read_file",
            arguments={"path": "tests/test_calculator.py"},
            description="Read the test file (#2).",
        ),
        ScriptedAction(
            tool_name="read_file",
            arguments={"path": "calculator.py"},
            description="Read the buggy calculator source (#3).",
        ),
        ScriptedAction(
            tool_name="write_file",
            arguments={"path": "calculator.py", "content": CORRECTED_CALCULATOR_CONTENT},
            description="Write the corrected calculator source.",
        ),
        ScriptedAction(
            tool_name="run_tests",
            arguments={},
            description="Run tests to verify all tests pass.",
        ),
    ]


# ===========================================================================
# Large output externalization action sequence (Phase 7 Step 5)
# ===========================================================================


def make_large_output_actions() -> list[ScriptedAction]:
    """Return the scripted action sequence for the
    large_output_externalization scenario.

    Sequence:
    1. read_file large_test_log.txt — produces a large output that
       exceeds the externalization threshold and context capacity.
    2. read_file calculator.py — small output, stays inline.
    3. write_file calculator.py — write the corrected source.
    4. run_tests — verify all tests pass.

    The first action immediately produces the large output so the
    next decision boundary tests externalization.
    """
    return [
        ScriptedAction(
            tool_name="read_file",
            arguments={"path": "large_test_log.txt"},
            description="Read the large test log file (large output).",
        ),
        ScriptedAction(
            tool_name="read_file",
            arguments={"path": "calculator.py"},
            description="Read the buggy calculator source.",
        ),
        ScriptedAction(
            tool_name="write_file",
            arguments={"path": "calculator.py", "content": CORRECTED_CALCULATOR_CONTENT},
            description="Write the corrected calculator source.",
        ),
        ScriptedAction(
            tool_name="run_tests",
            arguments={},
            description="Run tests to verify all tests pass.",
        ),
    ]
