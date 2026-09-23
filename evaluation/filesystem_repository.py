"""Real filesystem repository + real pytest subprocess (Phase 7 Step 6).

A real temporary filesystem repository that materializes a fixture
blueprint into a ``TemporaryDirectory``, supports real filesystem
reads/writes, and runs real ``sys.executable -m pytest`` subprocesses.

This is NOT a real-world GitHub repository. It is a synthetic
benchmark repository on a fresh temp directory. The action source is
still deterministic scripted — NOT an LLM reasoning agent.

Key invariants:
- Fresh temp directory per trial (no sharing between Naive/Reliable).
- No mutation of the real project source tree.
- Path containment: no absolute paths, no ``..`` escape, no symlinks
  pointing outside the repo root.
- Real async subprocess via ``asyncio.create_subprocess_exec`` (no
  ``shell=True``).
- ``sys.executable`` used (not ``python`` / ``python3``).
- Plugin autoload disabled, pytest cache disabled, bytecode disabled.
- Cancellation cleanup: child process terminated/killed on
  ``CancelledError``.
- Deterministic snapshot (sorted relative paths → content).
- ``returncode == 0`` → tests pass; ``returncode == 1`` → tests fail
  (domain result, NOT a tool error).
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FilesystemRepositoryError(Exception):
    """Base error for filesystem repository operations."""


class FilesystemPathError(FilesystemRepositoryError):
    """Raised when a path violates root containment or is invalid."""


class FilesystemFileNotFoundError(FilesystemRepositoryError):
    """Raised when a file path is not found in the repository."""


# ---------------------------------------------------------------------------
# Pytest subprocess result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PytestSubprocessResult:
    """Structured result of a real pytest subprocess invocation.

    Attributes:
        returncode: The process exit code. 0 = all tests pass,
            1 = tests collected and run but some failed (domain
            result, NOT a tool error).
        passed: True if returncode == 0.
        stdout: Captured stdout (may contain nondeterministic timing).
        stderr: Captured stderr.
    """

    returncode: int
    passed: bool
    stdout: str
    stderr: str

    def __post_init__(self) -> None:
        if not isinstance(self.returncode, int) or isinstance(
            self.returncode, bool
        ):
            raise ValueError("returncode must be an int")
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be a bool")
        if not isinstance(self.stdout, str):
            raise ValueError("stdout must be a str")
        if not isinstance(self.stderr, str):
            raise ValueError("stderr must be a str")


# ---------------------------------------------------------------------------
# Filesystem repository fixture (immutable blueprint)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilesystemRepositoryFixture:
    """Immutable blueprint for materializing a fresh filesystem
    repository.

    The fixture itself does NOT hold a mutable temp path. Each call to
    ``materialize()`` creates a NEW ``TemporaryDirectory`` and writes
    the fixture files into it.

    Attributes:
        fixture_id: Stable identifier.
        files: Mapping of relative POSIX path -> file content (str).
    """

    fixture_id: str
    files: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.fixture_id or not self.fixture_id.strip():
            raise ValueError("fixture_id must be a non-empty str")
        if not isinstance(self.files, Mapping):
            raise ValueError("files must be a Mapping")
        for path, content in self.files.items():
            if not isinstance(path, str) or not path.strip():
                raise ValueError("each file path must be a non-empty str")
            if not isinstance(content, str):
                raise ValueError("each file content must be a str")
            # Reject absolute paths in the fixture blueprint.
            if path.startswith("/") or Path(path).is_absolute():
                raise ValueError(
                    f"fixture path must be relative: {path!r}"
                )

    def materialize(
        self,
        *,
        prefix: str = "rah-benchmark-",
    ) -> "FilesystemRepository":
        """Create a fresh ``TemporaryDirectory``, materialize the
        fixture files, and return a ``FilesystemRepository`` rooted
        there.

        The caller is responsible for cleanup (via
        ``FilesystemRepository.cleanup()`` or context manager).
        """
        tmpdir = tempfile.mkdtemp(prefix=prefix)
        root = Path(tmpdir).resolve()
        for rel_path, content in self.files.items():
            # Normalize to forward slashes, then join.
            target = root / rel_path.replace("/", os.sep)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Write with explicit UTF-8 + \n to avoid platform newline
            # drift.
            target.write_text(content, encoding="utf-8", newline="\n")
        return FilesystemRepository(
            fixture_id=self.fixture_id,
            root=root,
            cleanup_dir=tmpdir,
        )


# ---------------------------------------------------------------------------
# Filesystem repository
# ---------------------------------------------------------------------------


# Directories/files to exclude from list_files and snapshot.
_EXCLUDED_NAMES: frozenset[str] = frozenset({
    "__pycache__",
    ".pytest_cache",
    ".cache",
})


class FilesystemRepository:
    """Real filesystem repository rooted at a temporary directory.

    Supports:
    - ``list_files()`` — recursive, sorted, relative POSIX paths.
    - ``read_file(path)`` — real UTF-8 read.
    - ``write_file(path, content)`` — real UTF-8 write with ``\\n``.
    - ``run_tests()`` — real ``sys.executable -m pytest`` subprocess.
    - ``snapshot()`` — deterministic ``dict[str, str]`` of relative
      path -> content (excludes ``__pycache__``, ``.pytest_cache``).
    - ``cleanup()`` — remove the temp directory.

    Path safety: all paths are validated for root containment. No
    absolute paths, no ``..`` escape, no symlinks pointing outside the
    root.
    """

    def __init__(
        self,
        *,
        fixture_id: str,
        root: Path,
        cleanup_dir: Optional[str] = None,
        process_factory: Optional[Any] = None,
    ) -> None:
        self._fixture_id = fixture_id
        self._root = root
        self._cleanup_dir = cleanup_dir
        self._process_factory = process_factory  # for unit testing
        self._agent_pytest_count = 0
        self._oracle_pytest_count = 0
        # Process diagnostics (Phase 7 Step 7).
        self._agent_spawn_count = 0
        self._agent_completed_count = 0
        self._active_process_count = 0
        self._process_returncodes: list[int] = []

    @property
    def fixture_id(self) -> str:
        return self._fixture_id

    @property
    def root(self) -> Path:
        """The resolved repository root path."""
        return self._root

    @property
    def agent_pytest_run_count(self) -> int:
        """Deprecated alias for ``agent_pytest_spawn_count``.

        Counts calls to ``run_tests()`` from agent tool handlers. For
        backward compat with Step 6 tests.
        """
        return self._agent_pytest_count

    @property
    def agent_pytest_spawn_count(self) -> int:
        """Number of pytest subprocesses actually spawned by agent
        tool handler calls (after ``create_subprocess_exec``)."""
        return self._agent_spawn_count

    @property
    def agent_pytest_completed_count(self) -> int:
        """Number of agent-spawned pytest subprocesses that completed
        normally (``communicate()`` returned without cancellation)."""
        return self._agent_completed_count

    @property
    def oracle_pytest_run_count(self) -> int:
        """Number of pytest subprocess invocations from the objective
        completion oracle."""
        return self._oracle_pytest_count

    @property
    def active_process_count(self) -> int:
        """Number of currently active child processes. Must be 0 after
        each trial (no orphan processes)."""
        return self._active_process_count

    @property
    def process_exit_returncodes(self) -> tuple[int, ...]:
        """Tuple of exit returncodes from all spawned processes, in
        spawn order. ``-1`` indicates the process was cancelled before
        producing a returncode."""
        return tuple(self._process_returncodes)

    # -- Path safety ---------------------------------------------------

    def _validate_path(self, path: str) -> Path:
        """Validate ``path`` and return the resolved absolute path
        within the repo root.

        Raises ``FilesystemPathError`` for:
        - Empty or non-string paths.
        - Absolute paths.
        - Paths that escape the repo root after resolution.
        - Symlinks that resolve outside the repo root.
        """
        if not isinstance(path, str) or not path.strip():
            raise FilesystemPathError("path must be a non-empty str")
        # Reject absolute paths.
        if Path(path).is_absolute():
            raise FilesystemPathError(
                f"absolute paths are not allowed: {path!r}"
            )
        # Join and resolve.
        candidate = (self._root / path).resolve()
        # Containment check: the resolved path must be within root.
        try:
            candidate.relative_to(self._root)
        except ValueError:
            raise FilesystemPathError(
                f"path escapes repository root: {path!r}"
            )
        # Symlink boundary: if the resolved path is a symlink, its
        # target must also be within root.
        if candidate.is_symlink():
            target = candidate.resolve()
            try:
                target.relative_to(self._root)
            except ValueError:
                raise FilesystemPathError(
                    f"symlink target escapes repository root: {path!r}"
                )
        return candidate

    # -- Inspection ----------------------------------------------------

    def list_files(self) -> tuple[str, ...]:
        """Return all file paths (recursive, sorted, relative POSIX).

        Excludes ``__pycache__``, ``.pytest_cache``, and ``*.pyc``.
        """
        results: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self._root):
            # Prune excluded directories in-place.
            dirnames[:] = [
                d for d in dirnames if d not in _EXCLUDED_NAMES
            ]
            for fname in filenames:
                if fname.endswith(".pyc"):
                    continue
                full = Path(dirpath) / fname
                rel = full.relative_to(self._root)
                # Use forward slashes for POSIX-style.
                results.append(rel.as_posix())
        return tuple(sorted(results))

    def read_file(self, path: str) -> str:
        """Read the UTF-8 content of ``path``.

        Raises ``FilesystemFileNotFoundError`` if the path does not
        exist. Raises ``FilesystemPathError`` for invalid paths.
        """
        target = self._validate_path(path)
        if not target.exists() or not target.is_file():
            raise FilesystemFileNotFoundError(
                f"file not found: {path!r}"
            )
        return target.read_text(encoding="utf-8")

    # -- Mutation ------------------------------------------------------

    def write_file(self, path: str, content: str) -> None:
        """Write ``content`` to ``path`` (creates or overwrites).

        Uses explicit UTF-8 encoding and ``\\n`` newlines to avoid
        platform drift. IDEMPOTENT: same path + same content = same
        resulting state.
        """
        target = self._validate_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")

    # -- Evaluation (real pytest subprocess) ---------------------------

    async def run_tests(
        self,
        *,
        is_oracle: bool = False,
        delay_seconds: float = 0.0,
    ) -> PytestSubprocessResult:
        """Run ``sys.executable -m pytest`` as a real subprocess.

        Uses ``asyncio.create_subprocess_exec`` (no ``shell=True``).
        The working directory is the repository root. Plugin autoload
        is disabled, pytest cache is disabled, bytecode is disabled.

        On ``CancelledError``, the child process is terminated (then
        killed if necessary) before re-raising.

        Args:
            is_oracle: If True, this invocation is from the objective
                completion oracle (not an agent tool call). Used only
                for accounting; does not affect execution.
            delay_seconds: If > 0, sets the
                ``RAH_BENCHMARK_TEST_DELAY_SECONDS`` environment
                variable so the test module sleeps for this duration.
                Used by the timeout fault to make the pytest child
                slow enough for the ToolRuntime timeout to fire.

        Returns:
            ``PytestSubprocessResult`` with returncode, passed,
            stdout, stderr.
        """
        if is_oracle:
            self._oracle_pytest_count += 1
        else:
            self._agent_pytest_count += 1

        argv = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--color=no",
            "-p",
            "no:cacheprovider",
            "tests",
        ]
        env = self._build_subprocess_env()
        if delay_seconds > 0:
            env["RAH_BENCHMARK_TEST_DELAY_SECONDS"] = str(delay_seconds)

        if self._process_factory is not None:
            # Unit-test hook: inject a fake process factory.
            return await self._process_factory(
                argv, cwd=str(self._root), env=env, is_oracle=is_oracle
            )

        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(self._root),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Track spawn after successful creation.
        if not is_oracle:
            self._agent_spawn_count += 1
        self._active_process_count += 1
        try:
            stdout_b, stderr_b = await proc.communicate()
            # Normal completion.
            if not is_oracle:
                self._agent_completed_count += 1
        except asyncio.CancelledError:
            # Cancellation cleanup: terminate, then kill if needed.
            await self._cancel_subprocess(proc)
            raise
        finally:
            self._active_process_count -= 1
            rc = proc.returncode if proc.returncode is not None else -1
            self._process_returncodes.append(rc)
        returncode = proc.returncode if proc.returncode is not None else -1
        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        return PytestSubprocessResult(
            returncode=returncode,
            passed=(returncode == 0),
            stdout=stdout,
            stderr=stderr,
        )

    def _build_subprocess_env(self) -> dict[str, str]:
        """Build an isolated environment for the pytest subprocess."""
        env = dict(os.environ)
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONHASHSEED"] = "0"
        env["PYTEST_ADDOPTS"] = ""
        return env

    async def _cancel_subprocess(self, proc: asyncio.subprocess.Process) -> None:
        """Terminate (then kill if necessary) a child process on
        cancellation. Awaits process exit before returning."""
        try:
            proc.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                return
            await proc.wait()

    # -- Snapshot ------------------------------------------------------

    def snapshot(self) -> dict[str, str]:
        """Return a deterministic immutable snapshot of the repository.

        Maps sorted relative POSIX path -> file content. Excludes
        ``__pycache__``, ``.pytest_cache``, ``*.pyc``. Does NOT include
        temp root path, mtime, ctime, inode, or pytest duration.
        """
        result: dict[str, str] = {}
        for rel_path in self.list_files():
            full = self._root / rel_path.replace("/", os.sep)
            result[rel_path] = full.read_text(encoding="utf-8")
        return result

    # -- Cleanup -------------------------------------------------------

    def cleanup(self) -> None:
        """Remove the temporary directory."""
        if self._cleanup_dir is not None and os.path.isdir(self._cleanup_dir):
            import shutil
            shutil.rmtree(self._cleanup_dir, ignore_errors=True)
            self._cleanup_dir = None

    def __enter__(self) -> "FilesystemRepository":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.cleanup()


# ---------------------------------------------------------------------------
# Filesystem repository oracle
# ---------------------------------------------------------------------------


class FilesystemRepositoryOracle:
    """Objective completion oracle for a ``FilesystemRepository``.

    The oracle runs a REAL pytest subprocess (bypassing the fault
    injector) to verify completion. The oracle invocation is NOT an
    agent action, NOT a tool invocation, and NOT a retry attempt.

    Completion truth: ``returncode == 0`` (NOT stdout parsing).
    """

    async def is_complete(self, repo: FilesystemRepository) -> bool:
        """Return True if the repository's tests all pass (returncode
        == 0) as verified by a real pytest subprocess."""
        result = await repo.run_tests(is_oracle=True)
        return result.passed

    async def get_result(
        self, repo: FilesystemRepository
    ) -> PytestSubprocessResult:
        """Return the full pytest subprocess result (for diagnostics)."""
        return await repo.run_tests(is_oracle=True)
