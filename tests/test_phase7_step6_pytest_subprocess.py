"""Phase 7 Step 6 real pytest subprocess tests.

Verifies:
- ``sys.executable -m pytest`` is actually launched (no shell=True).
- Initial fixture returncode == 1 (tests fail).
- After repair returncode == 0 (tests pass).
- stdout/stderr captured.
- cwd is repository root.
- Cache/pyc not persisted into snapshot.
- Cancellation cleanup propagates and terminates child process.
- Subprocess command safety (argv form, no shell=True).

All offline, deterministic, no LLM, no network.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from evaluation.filesystem_repository import (
    FilesystemRepository,
    FilesystemRepositoryFixture,
    FilesystemRepositoryOracle,
    PytestSubprocessResult,
)
from evaluation.scenarios.filesystem_scenarios import (
    FILESYSTEM_CORRECTED_CALCULATOR_CONTENT,
    make_filesystem_calculator_fixture,
)


def _run(coro):
    return asyncio.run(coro)


# ===========================================================================
# Real pytest subprocess
# ===========================================================================


class TestRealPytestSubprocess:
    def test_initial_fixture_fails(self):
        """Materialize fixture → real pytest → returncode == 1."""
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            result = _run(repo.run_tests())
            assert result.returncode == 1
            assert result.passed is False
        finally:
            repo.cleanup()

    def test_repaired_fixture_passes(self):
        """Write corrected calculator → real pytest → returncode == 0."""
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            repo.write_file("calculator.py", FILESYSTEM_CORRECTED_CALCULATOR_CONTENT)
            result = _run(repo.run_tests())
            assert result.returncode == 0
            assert result.passed is True
        finally:
            repo.cleanup()

    def test_stdout_captured(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            result = _run(repo.run_tests())
            assert isinstance(result.stdout, str)
            assert len(result.stdout) > 0
        finally:
            repo.cleanup()

    def test_stderr_captured(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            result = _run(repo.run_tests())
            assert isinstance(result.stderr, str)
        finally:
            repo.cleanup()

    def test_sys_executable_used(self):
        """The subprocess uses sys.executable, not 'python' or
        'python3'."""
        fixture = make_filesystem_calculator_fixture()
        captured_argv: list[str] = []

        async def fake_factory(argv, *, cwd, env, is_oracle):
            captured_argv.extend(argv)
            return PytestSubprocessResult(
                returncode=0, passed=True, stdout="", stderr=""
            )

        repo = FilesystemRepository(
            fixture_id=fixture.fixture_id,
            root=fixture.materialize().root,
            process_factory=fake_factory,
        )
        _run(repo.run_tests())
        assert captured_argv[0] == sys.executable

    def test_no_shell_true(self):
        """The subprocess uses create_subprocess_exec (argv form), not
        shell=True."""
        fixture = make_filesystem_calculator_fixture()
        captured_argv: list[str] = []

        async def fake_factory(argv, *, cwd, env, is_oracle):
            captured_argv.extend(argv)
            return PytestSubprocessResult(
                returncode=0, passed=True, stdout="", stderr=""
            )

        repo = fixture.materialize()
        repo._process_factory = fake_factory
        try:
            _run(repo.run_tests())
            # argv form: [sys.executable, "-m", "pytest", ...]
            assert "-m" in captured_argv
            assert "pytest" in captured_argv
            # No shell string.
            assert len(captured_argv) > 2
            for arg in captured_argv:
                assert isinstance(arg, str)
        finally:
            repo.cleanup()

    def test_cwd_is_repository_root(self):
        fixture = make_filesystem_calculator_fixture()
        captured_cwd: list[str] = []

        async def fake_factory(argv, *, cwd, env, is_oracle):
            captured_cwd.append(cwd)
            return PytestSubprocessResult(
                returncode=0, passed=True, stdout="", stderr=""
            )

        repo = fixture.materialize()
        repo._process_factory = fake_factory
        try:
            _run(repo.run_tests())
            assert captured_cwd[0] == str(repo.root)
        finally:
            repo.cleanup()

    def test_cache_not_persisted_into_snapshot(self):
        """After running pytest, the snapshot should NOT contain
        .pytest_cache or __pycache__."""
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            _run(repo.run_tests())
            snap = repo.snapshot()
            for key in snap:
                assert ".pytest_cache" not in key
                assert "__pycache__" not in key
                assert not key.endswith(".pyc")
        finally:
            repo.cleanup()

    def test_plugin_autoload_disabled(self):
        """PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 in subprocess env."""
        fixture = make_filesystem_calculator_fixture()
        captured_env: list[dict] = []

        async def fake_factory(argv, *, cwd, env, is_oracle):
            captured_env.append(env)
            return PytestSubprocessResult(
                returncode=0, passed=True, stdout="", stderr=""
            )

        repo = fixture.materialize()
        repo._process_factory = fake_factory
        try:
            _run(repo.run_tests())
            assert captured_env[0]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        finally:
            repo.cleanup()

    def test_bytecode_disabled(self):
        """PYTHONDONTWRITEBYTECODE=1 in subprocess env."""
        fixture = make_filesystem_calculator_fixture()
        captured_env: list[dict] = []

        async def fake_factory(argv, *, cwd, env, is_oracle):
            captured_env.append(env)
            return PytestSubprocessResult(
                returncode=0, passed=True, stdout="", stderr=""
            )

        repo = fixture.materialize()
        repo._process_factory = fake_factory
        try:
            _run(repo.run_tests())
            assert captured_env[0]["PYTHONDONTWRITEBYTECODE"] == "1"
        finally:
            repo.cleanup()

    def test_cache_provider_disabled(self):
        """``-p no:cacheprovider`` in argv."""
        fixture = make_filesystem_calculator_fixture()
        captured_argv: list[str] = []

        async def fake_factory(argv, *, cwd, env, is_oracle):
            captured_argv.extend(argv)
            return PytestSubprocessResult(
                returncode=0, passed=True, stdout="", stderr=""
            )

        repo = fixture.materialize()
        repo._process_factory = fake_factory
        try:
            _run(repo.run_tests())
            assert "no:cacheprovider" in captured_argv
        finally:
            repo.cleanup()


# ===========================================================================
# Oracle
# ===========================================================================


class TestFilesystemOracle:
    def test_oracle_initial_incomplete(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            oracle = FilesystemRepositoryOracle()
            assert not _run(oracle.is_complete(repo))
        finally:
            repo.cleanup()

    def test_oracle_complete_after_fix(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            repo.write_file("calculator.py", FILESYSTEM_CORRECTED_CALCULATOR_CONTENT)
            oracle = FilesystemRepositoryOracle()
            assert _run(oracle.is_complete(repo))
        finally:
            repo.cleanup()

    def test_oracle_does_not_count_as_agent_pytest(self):
        """Oracle pytest invocations do NOT increment
        agent_pytest_run_count."""
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            oracle = FilesystemRepositoryOracle()
            _run(oracle.is_complete(repo))
            assert repo.agent_pytest_run_count == 0
            assert repo.oracle_pytest_run_count == 1
        finally:
            repo.cleanup()


# ===========================================================================
# Cancellation cleanup
# ===========================================================================


class TestCancellationCleanup:
    def test_cancellation_propagates_and_cleans_up(self):
        """When the pytest await is cancelled, the child process is
        terminated and CancelledError is re-raised."""
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            # Use a fake process factory that simulates a long-running
            # subprocess that gets cancelled.
            class FakeProc:
                def __init__(self):
                    self._terminated = False
                    self._killed = False
                    self._returncode = None
                async def communicate(self):
                    await asyncio.sleep(100)  # never completes
                    return b"", b""
                def terminate(self):
                    self._terminated = True
                def kill(self):
                    self._killed = True
                async def wait(self):
                    await asyncio.sleep(0.01)
                    self._returncode = 0
                    return 0
                @property
                def returncode(self):
                    return self._returncode

            fake_proc = FakeProc()

            async def fake_factory(argv, *, cwd, env, is_oracle):
                # Simulate create_subprocess_exec returning our fake proc.
                # We need to actually use the real path but inject the proc.
                # Instead, we'll patch the repo's run_tests to use our fake.
                raise NotImplementedError("use direct patch")

            # Patch the repo to use a fake subprocess.
            original_exec = asyncio.create_subprocess_exec
            def patched_exec(*args, **kwargs):
                # Return a coroutine that yields our fake proc.
                async def coro():
                    return fake_proc
                return coro()
            asyncio.create_subprocess_exec = patched_exec
            try:
                async def run_and_cancel():
                    task = asyncio.create_task(repo.run_tests())
                    await asyncio.sleep(0.05)  # let it start
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                _run(run_and_cancel())
            finally:
                asyncio.create_subprocess_exec = original_exec
            # The fake proc should have been terminated.
            assert fake_proc._terminated
        finally:
            repo.cleanup()
