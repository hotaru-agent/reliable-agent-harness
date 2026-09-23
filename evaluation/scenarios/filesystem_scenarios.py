"""Filesystem benchmark scenarios (Phase 7 Steps 6–8).

Defines the filesystem_pytest_v1 scenarios:
- ``filesystem_pytest_clean`` — no faults, real filesystem repair.
- ``filesystem_pytest_transient`` — run_tests invocation #1 fails
  transiently; Reliable recovers via existing ToolRuntime RetryPolicy.
- ``filesystem_pytest_timeout`` — run_tests invocation #1 spawns a
  real pytest child that is deliberately slow; the production
  ToolRuntime timeout fires, cancellation cleanup reaps the child,
  and Reliable retries with a fresh subprocess (Phase 7 Step 7).
- ``filesystem_pytest_recovery`` — two-bug fixture; partial repair is
  checkpointed, then PROCESS_INTERRUPTION fires on the next
  read_file; Reliable resumes from checkpoint on the same filesystem
  workspace and completes the final repair (Phase 7 Step 8).

The standard 6-action script (clean/transient/timeout):
0. list_files
1. run_tests
2. read_file("calculator.py")
3. read_file("tests/test_calculator.py")
4. write_file("calculator.py", corrected_content)
5. run_tests

The 8-action recovery script (recovery):
0. list_files
1. run_tests
2. read_file("calculator.py")
3. write_file("calculator.py", partial_repair_content)
4. read_file("calculator.py")  ← PROCESS_INTERRUPTION here
5. read_file("tests/test_calculator.py")
6. write_file("calculator.py", final_repair_content)
7. run_tests

The repository is a real temporary filesystem with real Python files
and real pytest tests. The oracle is a real pytest subprocess.

All offline, deterministic, no LLM, no network.
"""

from __future__ import annotations

from evaluation.faults import FaultPlan, FaultSpec, FaultType
from evaluation.filesystem_repository import FilesystemRepositoryFixture
from evaluation.models import BenchmarkScenario, BenchmarkTask
from evaluation.scenarios.controlled_repo import ScriptedAction


# ---------------------------------------------------------------------------
# Repository content — single-bug fixture (clean/transient/timeout)
# ---------------------------------------------------------------------------

# Buggy initial calculator: add() is implemented as subtraction.
FILESYSTEM_BUGGY_CALCULATOR = """\
def add(a, b):
    return a - b


def subtract(a, b):
    return a - b
"""

# Corrected calculator: add() is now correct.
FILESYSTEM_CORRECTED_CALCULATOR = """\
def add(a, b):
    return a + b


def subtract(a, b):
    return a - b
"""

# Real pytest test file. Supports an optional benchmark delay via
# the RAH_BENCHMARK_TEST_DELAY_SECONDS environment variable so the
# timeout fault can make the pytest child slow enough for the
# ToolRuntime timeout to fire (Phase 7 Step 7).
FILESYSTEM_TEST_FILE = """\
import os
import time

from calculator import add, subtract

delay = float(os.environ.get("RAH_BENCHMARK_TEST_DELAY_SECONDS", "0"))


def test_add():
    if delay > 0:
        time.sleep(delay)
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(5, 3) == 2
"""

# The corrected content for write_file actions.
FILESYSTEM_CORRECTED_CALCULATOR_CONTENT = FILESYSTEM_CORRECTED_CALCULATOR


# ---------------------------------------------------------------------------
# Repository content — two-bug fixture (recovery)
# ---------------------------------------------------------------------------

# Two-bug calculator: add() is subtraction, multiply() is addition.
FILESYSTEM_BUGGY_CALCULATOR_TWO_BUGS = """\
def add(a, b):
    return a - b


def multiply(a, b):
    return a + b
"""

# Partial repair: fix add(), leave multiply() buggy.
FILESYSTEM_PARTIAL_REPAIR_CALCULATOR = """\
def add(a, b):
    return a + b


def multiply(a, b):
    return a + b
"""

# Final repair: fix both add() and multiply().
FILESYSTEM_FINAL_REPAIR_CALCULATOR = """\
def add(a, b):
    return a + b


def multiply(a, b):
    return a * b
"""

# Test file for the two-bug fixture.
FILESYSTEM_TWO_BUG_TEST_FILE = """\
import os
import time

from calculator import add, multiply

delay = float(os.environ.get("RAH_BENCHMARK_TEST_DELAY_SECONDS", "0"))


def test_add():
    if delay > 0:
        time.sleep(delay)
    assert add(2, 3) == 5


def test_multiply():
    assert multiply(2, 3) == 6
"""


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def make_filesystem_calculator_fixture(
    *,
    fixture_id: str = "filesystem_calculator_buggy_add",
) -> FilesystemRepositoryFixture:
    """Return the standard filesystem calculator repository fixture.

    The fixture contains:
    - ``calculator.py`` — with a buggy ``add`` (returns ``a - b``).
    - ``tests/test_calculator.py`` — real pytest test file.

    The oracle uses a real pytest subprocess, NOT in-process checkers.
    """
    return FilesystemRepositoryFixture(
        fixture_id=fixture_id,
        files={
            "calculator.py": FILESYSTEM_BUGGY_CALCULATOR,
            "tests/test_calculator.py": FILESYSTEM_TEST_FILE,
        },
    )


def make_filesystem_standard_fixtures() -> dict[str, FilesystemRepositoryFixture]:
    """Return the standard set of filesystem repository fixtures."""
    return {
        "filesystem_calculator_buggy_add": make_filesystem_calculator_fixture(),
        "filesystem_calculator_two_bugs": FilesystemRepositoryFixture(
            fixture_id="filesystem_calculator_two_bugs",
            files={
                "calculator.py": FILESYSTEM_BUGGY_CALCULATOR_TWO_BUGS,
                "tests/test_calculator.py": FILESYSTEM_TWO_BUG_TEST_FILE,
            },
        ),
    }


# ---------------------------------------------------------------------------
# Action script (6 actions)
# ---------------------------------------------------------------------------


def make_filesystem_fix_calculator_actions() -> list[ScriptedAction]:
    """Return the standard 6-action scripted sequence for filesystem
    repository maintenance.

    Sequence:
    0. list_files — inspect the repository
    1. run_tests — observe failure
    2. read_file("calculator.py") — read the buggy source
    3. read_file("tests/test_calculator.py") — read the tests
    4. write_file("calculator.py", corrected) — write the fix
    5. run_tests — verify all tests pass
    """
    return [
        ScriptedAction(
            tool_name="list_files",
            arguments={},
            description="List all files in the repository.",
        ),
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
            tool_name="read_file",
            arguments={"path": "tests/test_calculator.py"},
            description="Read the test file.",
        ),
        ScriptedAction(
            tool_name="write_file",
            arguments={
                "path": "calculator.py",
                "content": FILESYSTEM_CORRECTED_CALCULATOR_CONTENT,
            },
            description="Write the corrected calculator source.",
        ),
        ScriptedAction(
            tool_name="run_tests",
            arguments={},
            description="Run tests to verify all tests pass.",
        ),
    ]


def make_filesystem_recovery_actions() -> list[ScriptedAction]:
    """Return the 8-action scripted sequence for the recovery scenario.

    Sequence:
    0. list_files — inspect the repository
    1. run_tests — observe failure (2 failing tests)
    2. read_file("calculator.py") — read the buggy source
    3. write_file("calculator.py", partial_repair) — fix add(), leave
       multiply() buggy → checkpoint after this step
    4. read_file("calculator.py") — ← PROCESS_INTERRUPTION here
    5. read_file("tests/test_calculator.py") — read the tests
    6. write_file("calculator.py", final_repair) — fix both bugs
    7. run_tests — verify all tests pass
    """
    return [
        ScriptedAction(
            tool_name="list_files",
            arguments={},
            description="List all files in the repository.",
        ),
        ScriptedAction(
            tool_name="run_tests",
            arguments={},
            description="Run tests to observe 2 failing tests.",
        ),
        ScriptedAction(
            tool_name="read_file",
            arguments={"path": "calculator.py"},
            description="Read the buggy calculator source.",
        ),
        ScriptedAction(
            tool_name="write_file",
            arguments={
                "path": "calculator.py",
                "content": FILESYSTEM_PARTIAL_REPAIR_CALCULATOR,
            },
            description="Write partial repair (fix add, leave multiply buggy).",
        ),
        ScriptedAction(
            tool_name="read_file",
            arguments={"path": "calculator.py"},
            description="Read the partially-repaired calculator source.",
        ),
        ScriptedAction(
            tool_name="read_file",
            arguments={"path": "tests/test_calculator.py"},
            description="Read the test file.",
        ),
        ScriptedAction(
            tool_name="write_file",
            arguments={
                "path": "calculator.py",
                "content": FILESYSTEM_FINAL_REPAIR_CALCULATOR,
            },
            description="Write final repair (fix both add and multiply).",
        ),
        ScriptedAction(
            tool_name="run_tests",
            arguments={},
            description="Run tests to verify all tests pass.",
        ),
    ]


def make_filesystem_integrated_multifault_actions() -> list[ScriptedAction]:
    """Return the 10-action scripted sequence for the integrated
    multi-fault scenario (Phase 7 Step 9).

    Sequence:
    0. list_files
    1. run_tests — initial fail (TRANSIENT on attempt 1, retry succeeds)
    2. read_file("calculator.py")
    3. read_file("tests/test_calculator.py")
    4. write_file("calculator.py", PARTIAL_REPAIR) — checkpointed
    5. run_tests — partial fail (TIMEOUT on attempt 1, retry succeeds)
    6. read_file("calculator.py") — PROCESS_INTERRUPTION here
    7. read_file("tests/test_calculator.py")
    8. write_file("calculator.py", FINAL_REPAIR)
    9. run_tests — all pass
    """
    return [
        ScriptedAction(
            tool_name="list_files",
            arguments={},
            description="List all files in the repository.",
        ),
        ScriptedAction(
            tool_name="run_tests",
            arguments={},
            description="Run tests to observe 2 failing tests (transient fault on attempt 1).",
        ),
        ScriptedAction(
            tool_name="read_file",
            arguments={"path": "calculator.py"},
            description="Read the buggy calculator source.",
        ),
        ScriptedAction(
            tool_name="read_file",
            arguments={"path": "tests/test_calculator.py"},
            description="Read the test file.",
        ),
        ScriptedAction(
            tool_name="write_file",
            arguments={
                "path": "calculator.py",
                "content": FILESYSTEM_PARTIAL_REPAIR_CALCULATOR,
            },
            description="Write partial repair (fix add, leave multiply buggy).",
        ),
        ScriptedAction(
            tool_name="run_tests",
            arguments={},
            description="Run tests to observe 1 failing test (timeout fault on attempt 1).",
        ),
        ScriptedAction(
            tool_name="read_file",
            arguments={"path": "calculator.py"},
            description="Read the partially-repaired calculator (interruption here).",
        ),
        ScriptedAction(
            tool_name="read_file",
            arguments={"path": "tests/test_calculator.py"},
            description="Read the test file.",
        ),
        ScriptedAction(
            tool_name="write_file",
            arguments={
                "path": "calculator.py",
                "content": FILESYSTEM_FINAL_REPAIR_CALCULATOR,
            },
            description="Write final repair (fix both add and multiply).",
        ),
        ScriptedAction(
            tool_name="run_tests",
            arguments={},
            description="Run tests to verify all tests pass.",
        ),
    ]


# ---------------------------------------------------------------------------
# Fault plans
# ---------------------------------------------------------------------------


def make_filesystem_standard_fault_plans() -> dict[str, FaultPlan]:
    """Return the standard set of fault plans for filesystem scenarios."""
    return {
        "none": FaultPlan.no_faults(),
        "filesystem_transient_run_tests_1": FaultPlan(
            plan_id="filesystem_transient_run_tests_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
            ),
        ),
        "filesystem_timeout_run_tests_1": FaultPlan(
            plan_id="filesystem_timeout_run_tests_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TIMEOUT,
                    tool_name="run_tests",
                    logical_invocation=1,
                    # Delay for the subprocess (not handler sleep).
                    # Must be clearly larger than the ToolSpec timeout
                    # (1.0s) so the ToolRuntime timeout fires first.
                    timeout_sleep_seconds=3.0,
                ),
            ),
        ),
        # Recovery: PROCESS_INTERRUPTION at read_file logical #2
        # (action 4 in the 8-action script, after the checkpointed
        # partial write at action 3).
        "filesystem_recovery_read_file_2": FaultPlan(
            plan_id="filesystem_recovery_read_file_2",
            faults=(
                FaultSpec(
                    fault_type=FaultType.PROCESS_INTERRUPTION,
                    tool_name="read_file",
                    logical_invocation=2,
                ),
            ),
        ),
        # No-checkpoint negative: interruption at the very first
        # logical action (list_files #1), before any checkpoint.
        "filesystem_no_checkpoint_list_files_1": FaultPlan(
            plan_id="filesystem_no_checkpoint_list_files_1",
            faults=(
                FaultSpec(
                    fault_type=FaultType.PROCESS_INTERRUPTION,
                    tool_name="list_files",
                    logical_invocation=1,
                ),
            ),
        ),
        # Integrated multi-fault (Phase 7 Step 9): three faults in
        # one plan, encountered sequentially in the same trial.
        # A. run_tests logical #1 → TRANSIENT (action 1)
        # B. run_tests logical #2 → TIMEOUT (action 5)
        # C. read_file logical #3 → PROCESS_INTERRUPTION (action 6)
        "filesystem_integrated_multifault": FaultPlan(
            plan_id="filesystem_integrated_multifault",
            faults=(
                FaultSpec(
                    fault_type=FaultType.TRANSIENT_FAILURE,
                    tool_name="run_tests",
                    logical_invocation=1,
                ),
                FaultSpec(
                    fault_type=FaultType.TIMEOUT,
                    tool_name="run_tests",
                    logical_invocation=2,
                    timeout_sleep_seconds=3.0,
                ),
                FaultSpec(
                    fault_type=FaultType.PROCESS_INTERRUPTION,
                    tool_name="read_file",
                    logical_invocation=3,
                ),
            ),
        ),
    }


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def make_filesystem_standard_scenarios() -> dict[str, BenchmarkScenario]:
    """Return the standard set of Phase 7 Step 6/7/8 filesystem scenarios.

    Scenarios:
    - ``filesystem_pytest_clean`` — no faults, real filesystem repair.
    - ``filesystem_pytest_transient`` — run_tests invocation #1 fails
      transiently; Reliable recovers via ToolRuntime retry.
    - ``filesystem_pytest_timeout`` — run_tests invocation #1 spawns a
      real pytest child that is deliberately slow; the production
      ToolRuntime timeout fires, cancellation cleanup reaps the child,
      and Reliable retries with a fresh subprocess (Phase 7 Step 7).
    - ``filesystem_pytest_recovery`` — two-bug fixture; partial repair
      is checkpointed, then PROCESS_INTERRUPTION fires on the next
      read_file; Reliable resumes from checkpoint on the same
      filesystem workspace and completes the final repair (Phase 7
      Step 8).
    """
    scenarios: dict[str, BenchmarkScenario] = {}

    # Clean scenario.
    task_clean = BenchmarkTask(
        task_id="fix_filesystem_calculator_add",
        goal="Fix the buggy add() function in calculator.py so all pytest tests pass.",
        scenario_id="filesystem_pytest_clean",
        max_logical_actions=10,
    )
    scenarios["filesystem_pytest_clean"] = BenchmarkScenario(
        scenario_id="filesystem_pytest_clean",
        task=task_clean,
        fixture_id="filesystem_calculator_buggy_add",
        fault_plan_id="none",
        description="Clean run on real filesystem with real pytest subprocess.",
    )

    # Transient scenario.
    task_t = BenchmarkTask(
        task_id="fix_filesystem_calculator_add",
        goal="Fix the buggy add() function in calculator.py so all pytest tests pass.",
        scenario_id="filesystem_pytest_transient",
        max_logical_actions=10,
    )
    scenarios["filesystem_pytest_transient"] = BenchmarkScenario(
        scenario_id="filesystem_pytest_transient",
        task=task_t,
        fixture_id="filesystem_calculator_buggy_add",
        fault_plan_id="filesystem_transient_run_tests_1",
        description=(
            "run_tests invocation #1 fails transiently; "
            "Reliable recovers via ToolRuntime retry on real pytest subprocess."
        ),
    )

    # Timeout scenario (Phase 7 Step 7).
    task_to = BenchmarkTask(
        task_id="fix_filesystem_calculator_add",
        goal="Fix the buggy add() function in calculator.py so all pytest tests pass.",
        scenario_id="filesystem_pytest_timeout",
        max_logical_actions=10,
    )
    scenarios["filesystem_pytest_timeout"] = BenchmarkScenario(
        scenario_id="filesystem_pytest_timeout",
        task=task_to,
        fixture_id="filesystem_calculator_buggy_add",
        fault_plan_id="filesystem_timeout_run_tests_1",
        description=(
            "run_tests invocation #1 spawns a real pytest child that "
            "is deliberately slow; production ToolRuntime timeout fires, "
            "cancellation cleanup reaps the child, and Reliable retries "
            "with a fresh subprocess."
        ),
    )

    # Recovery scenario (Phase 7 Step 8).
    task_r = BenchmarkTask(
        task_id="fix_filesystem_calculator_two_bugs",
        goal="Fix both buggy add() and multiply() functions in calculator.py so all pytest tests pass.",
        scenario_id="filesystem_pytest_recovery",
        max_logical_actions=12,
    )
    scenarios["filesystem_pytest_recovery"] = BenchmarkScenario(
        scenario_id="filesystem_pytest_recovery",
        task=task_r,
        fixture_id="filesystem_calculator_two_bugs",
        fault_plan_id="filesystem_recovery_read_file_2",
        description=(
            "Two-bug fixture; partial repair (fix add) is checkpointed, "
            "then PROCESS_INTERRUPTION fires on read_file logical #2; "
            "Reliable resumes from checkpoint on the same filesystem "
            "workspace and completes the final repair."
        ),
    )

    return scenarios


def make_filesystem_no_checkpoint_scenario() -> BenchmarkScenario:
    """Return the no-checkpoint negative scenario (Phase 7 Step 8).

    Interruption at the very first logical action (list_files #1),
    before any checkpoint exists. Reliable cannot resume and must
    fail with NO_CHECKPOINT. This is NOT part of the formal
    filesystem_pytest_v1 suite matrix — it is a regression fixture.
    """
    task = BenchmarkTask(
        task_id="fix_filesystem_calculator_two_bugs",
        goal="Fix both buggy add() and multiply() functions in calculator.py so all pytest tests pass.",
        scenario_id="filesystem_pytest_no_checkpoint",
        max_logical_actions=12,
    )
    return BenchmarkScenario(
        scenario_id="filesystem_pytest_no_checkpoint",
        task=task,
        fixture_id="filesystem_calculator_two_bugs",
        fault_plan_id="filesystem_no_checkpoint_list_files_1",
        description=(
            "Interruption at the first logical action before any "
            "checkpoint; Reliable cannot resume (NO_CHECKPOINT)."
        ),
    )


def make_filesystem_integrated_multifault_scenario() -> BenchmarkScenario:
    """Return the integrated multi-fault scenario (Phase 7 Step 9).

    A single Reliable trial encounters three sequential faults:
    - TRANSIENT_FAILURE on run_tests logical #1 (action 1)
    - TIMEOUT on run_tests logical #2 (action 5)
    - PROCESS_INTERRUPTION on read_file logical #3 (action 6)

    Reliable recovers via existing RetryPolicy (transient + timeout)
    and real HarnessRuntime.resume() (interruption), then completes
    the final repair. This scenario is in the
    ``integrated_filesystem_v1`` suite, NOT in ``filesystem_pytest_v1``.
    """
    task = BenchmarkTask(
        task_id="fix_filesystem_calculator_two_bugs",
        goal="Fix both buggy add() and multiply() functions in calculator.py so all pytest tests pass.",
        scenario_id="filesystem_integrated_multifault",
        max_logical_actions=14,
    )
    return BenchmarkScenario(
        scenario_id="filesystem_integrated_multifault",
        task=task,
        fixture_id="filesystem_calculator_two_bugs",
        fault_plan_id="filesystem_integrated_multifault",
        description=(
            "Integrated multi-fault: transient retry + real subprocess "
            "timeout + checkpoint/resume in one long filesystem trial."
        ),
    )
