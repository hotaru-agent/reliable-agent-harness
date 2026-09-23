"""Controlled repository abstraction (Phase 7 Step 1).

A fully in-memory, deterministic repository that models the semantics of
a small software-engineering maintenance workload. No real filesystem,
no subprocess, no pytest — all evaluation is done by a deterministic
in-process oracle.

The repository has real mutable state: ``write_file`` changes file
contents, and ``run_tests`` reflects the current repository state. This
is not a log recorder; it is a live, inspectable, resettable fixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


# ---------------------------------------------------------------------------
# Repository test result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepositoryTestResult:
    """Structured result of running the repository's test suite.

    Produced by the deterministic in-process evaluator, NOT by a real
    pytest subprocess.

    Attributes:
        passed: True if all tests passed.
        total: Total number of tests.
        passed_count: Number of tests that passed.
        failed_count: Number of tests that failed.
        failing_tests: Names of failing tests, in deterministic order.
    """

    passed: bool
    total: int
    passed_count: int
    failed_count: int
    failing_tests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be a bool")
        if (
            not isinstance(self.total, int)
            or isinstance(self.total, bool)
            or self.total < 0
        ):
            raise ValueError("total must be an int >= 0")
        if (
            not isinstance(self.passed_count, int)
            or isinstance(self.passed_count, bool)
            or self.passed_count < 0
        ):
            raise ValueError("passed_count must be an int >= 0")
        if (
            not isinstance(self.failed_count, int)
            or isinstance(self.failed_count, bool)
            or self.failed_count < 0
        ):
            raise ValueError("failed_count must be an int >= 0")
        if not isinstance(self.failing_tests, tuple):
            raise ValueError("failing_tests must be a tuple")
        if self.passed_count + self.failed_count != self.total:
            raise ValueError(
                "passed_count + failed_count must equal total"
            )
        if self.passed and self.failed_count != 0:
            raise ValueError(
                "passed=True but failed_count > 0"
            )
        if not self.passed and self.failed_count == 0:
            raise ValueError(
                "passed=False but failed_count == 0"
            )


# ---------------------------------------------------------------------------
# Repository oracle
# ---------------------------------------------------------------------------


class RepositoryOracle:
    """Deterministic completion oracle for a ``ControlledRepository``.

    The oracle is the SINGLE source of truth for task completion. An
    agent cannot self-declare completion — the oracle must verify it.

    The default completion criterion is: all repository tests pass.
    """

    def is_complete(self, repo: ControlledRepository) -> bool:
        """Return True if the repository's tests all pass."""
        result = repo.run_tests()
        return result.passed


# ---------------------------------------------------------------------------
# Repository fixture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepositoryFixture:
    """Immutable blueprint for creating fresh ``ControlledRepository``
    instances.

    A fixture is a frozen snapshot of file contents. Each call to
    ``create_repository()`` returns a NEW ``ControlledRepository`` with
    a deep copy of the fixture's files — no mutable state is shared
    between instances.

    Attributes:
        fixture_id: Stable identifier for this fixture.
        files: Mapping of relative path -> file content (str).
        test_definitions: Tuple of ``TestDefinition`` instances that the
            in-process evaluator will run.
    """

    fixture_id: str
    files: Mapping[str, str] = field(default_factory=dict)
    test_definitions: tuple["TestDefinition", ...] = ()

    def __post_init__(self) -> None:
        if not self.fixture_id or not self.fixture_id.strip():
            raise ValueError("fixture_id must be a non-empty str")
        if not isinstance(self.files, Mapping):
            raise ValueError("files must be a Mapping")
        if not isinstance(self.test_definitions, tuple):
            raise ValueError("test_definitions must be a tuple")

    def create_repository(self) -> ControlledRepository:
        """Return a fresh ``ControlledRepository`` from this fixture.

        The returned repository has a deep copy of the fixture's files
        so mutations do not leak back to the fixture.
        """
        return ControlledRepository(
            fixture_id=self.fixture_id,
            files=dict(self.files),
            test_definitions=self.test_definitions,
        )


# ---------------------------------------------------------------------------
# Test definition (in-process deterministic evaluator)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestDefinition:
    """A single deterministic in-process test.

    The ``checker`` is a callable that receives the current repository
    files (as a dict) and returns ``True`` if the test passes.

    This is NOT a real pytest test. It is a deterministic in-process
    function that models the semantics of a repository test.

    Attributes:
        name: Test name (must be unique within a fixture).
        checker: Callable[[dict[str, str]], bool].
    """

    # Tell pytest this is NOT a test class.
    __test__ = False

    name: str
    checker: Any  # Callable[[dict[str, str]], bool]

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("name must be a non-empty str")
        if not callable(self.checker):
            raise ValueError("checker must be callable")


# ---------------------------------------------------------------------------
# ControlledRepository
# ---------------------------------------------------------------------------


class RepositoryFileNotFoundError(Exception):
    """Raised when a file path is not found in the repository."""


@dataclass
class ControlledRepository:
    """In-memory, deterministic, mutable repository.

    Supports:
    - ``list_files()`` — list all file paths.
    - ``read_file(path)`` — read a file's content.
    - ``write_file(path, content)`` — write/overwrite a file.
    - ``run_tests()`` — run the deterministic in-process test suite.
    - ``reset()`` — reset to the original fixture state.

    The repository has REAL mutable state. ``write_file`` changes the
    file contents, and subsequent ``run_tests`` calls reflect the new
    state. ``reset()`` restores the original fixture state.

    Attributes:
        fixture_id: Identifier of the source fixture.
        files: Mutable dict of path -> content.
        test_definitions: Tuple of test definitions (immutable).
    """

    fixture_id: str
    files: dict[str, str] = field(default_factory=dict)
    test_definitions: tuple[TestDefinition, ...] = ()
    _initial_files: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.fixture_id or not self.fixture_id.strip():
            raise ValueError("fixture_id must be a non-empty str")
        if not isinstance(self.files, dict):
            raise ValueError("files must be a dict")
        if not isinstance(self.test_definitions, tuple):
            raise ValueError("test_definitions must be a tuple")
        # Snapshot the initial state for reset.
        self._initial_files = dict(self.files)

    # -- Inspection -----------------------------------------------------

    def list_files(self) -> tuple[str, ...]:
        """Return all file paths in deterministic (sorted) order."""
        return tuple(sorted(self.files.keys()))

    def read_file(self, path: str) -> str:
        """Return the content of ``path``.

        Raises ``RepositoryFileNotFoundError`` if the path does not exist.
        """
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty str")
        if path not in self.files:
            raise RepositoryFileNotFoundError(
                f"file not found: {path!r}"
            )
        return self.files[path]

    # -- Mutation -------------------------------------------------------

    def write_file(self, path: str, content: str) -> None:
        """Write ``content`` to ``path``, creating or overwriting the file.

        This is an IDEMPOTENT operation: writing the same content to the
        same path twice produces the same resulting state.
        """
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty str")
        if not isinstance(content, str):
            raise ValueError("content must be a str")
        self.files[path] = content

    # -- Evaluation -----------------------------------------------------

    def run_tests(self) -> RepositoryTestResult:
        """Run the deterministic in-process test suite.

        Each ``TestDefinition.checker`` is called with the current files
        dict. Tests are run in definition order. Failing test names are
        collected in deterministic order.
        """
        failing: list[str] = []
        total = len(self.test_definitions)
        for td in self.test_definitions:
            try:
                ok = bool(td.checker(self.files))
            except Exception:
                ok = False
            if not ok:
                failing.append(td.name)

        failed_count = len(failing)
        passed_count = total - failed_count
        return RepositoryTestResult(
            passed=(failed_count == 0),
            total=total,
            passed_count=passed_count,
            failed_count=failed_count,
            failing_tests=tuple(failing),
        )

    # -- Reset ----------------------------------------------------------

    def reset(self) -> None:
        """Reset the repository to its initial fixture state."""
        self.files = dict(self._initial_files)
