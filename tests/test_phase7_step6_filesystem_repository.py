"""Phase 7 Step 6 filesystem repository tests.

Verifies the real filesystem repository:
- Fixture materializes into a fresh temp directory.
- list_files is recursive, sorted, relative POSIX paths.
- read_file reads real UTF-8 content.
- write_file changes disk.
- snapshot is deterministic.
- Fresh copies are independent.
- Path safety: absolute paths rejected, ``..`` escape rejected,
  unknown file raises explicit error.

All offline, deterministic, no LLM, no network.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from evaluation.filesystem_repository import (
    FilesystemFileNotFoundError,
    FilesystemPathError,
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
# Fixture materialization
# ===========================================================================


class TestFixtureMaterialization:
    def test_materialize_creates_temp_dir(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            assert repo.root.exists()
            assert repo.root.is_dir()
        finally:
            repo.cleanup()

    def test_materialize_writes_files(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            assert (repo.root / "calculator.py").exists()
            assert (repo.root / "tests" / "test_calculator.py").exists()
        finally:
            repo.cleanup()

    def test_materialize_content_correct(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            content = (repo.root / "calculator.py").read_text(encoding="utf-8")
            assert "def add(a, b):" in content
            assert "return a - b" in content  # buggy
        finally:
            repo.cleanup()

    def test_fixture_does_not_hold_temp_path(self):
        fixture = make_filesystem_calculator_fixture()
        # The fixture is immutable and does not hold a temp path.
        assert not hasattr(fixture, "_root")
        assert not hasattr(fixture, "_cleanup_dir")


# ===========================================================================
# list_files
# ===========================================================================


class TestListFiles:
    def test_list_files_sorted_relative_posix(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            files = repo.list_files()
            assert files == ("calculator.py", "tests/test_calculator.py")
        finally:
            repo.cleanup()

    def test_list_files_recursive(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            files = repo.list_files()
            # Should include nested test file.
            assert "tests/test_calculator.py" in files
        finally:
            repo.cleanup()

    def test_list_files_excludes_pycache(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            # Create a __pycache__ dir with a .pyc file.
            pycache = repo.root / "__pycache__"
            pycache.mkdir(exist_ok=True)
            (pycache / "calculator.cpython-311.pyc").write_bytes(b"fake")
            files = repo.list_files()
            assert "__pycache__" not in " ".join(files)
            assert not any(f.endswith(".pyc") for f in files)
        finally:
            repo.cleanup()

    def test_list_files_excludes_pytest_cache(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            cache = repo.root / ".pytest_cache"
            cache.mkdir(exist_ok=True)
            (cache / "v").mkdir(exist_ok=True)
            (cache / "v" / "cache").write_text("lastfailed", encoding="utf-8")
            files = repo.list_files()
            assert ".pytest_cache" not in " ".join(files)
        finally:
            repo.cleanup()


# ===========================================================================
# read_file
# ===========================================================================


class TestReadFile:
    def test_read_file_real_content(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            content = repo.read_file("calculator.py")
            assert "def add" in content
        finally:
            repo.cleanup()

    def test_read_file_nested_path(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            content = repo.read_file("tests/test_calculator.py")
            assert "def test_add" in content
        finally:
            repo.cleanup()

    def test_read_file_unknown_raises(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            with pytest.raises(FilesystemFileNotFoundError):
                repo.read_file("nonexistent.py")
        finally:
            repo.cleanup()


# ===========================================================================
# write_file
# ===========================================================================


class TestWriteFile:
    def test_write_file_changes_disk(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            repo.write_file("calculator.py", FILESYSTEM_CORRECTED_CALCULATOR_CONTENT)
            content = repo.read_file("calculator.py")
            assert "return a + b" in content
        finally:
            repo.cleanup()

    def test_write_file_creates_new_file(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            repo.write_file("new_file.py", "# new file\n")
            content = repo.read_file("new_file.py")
            assert content == "# new file\n"
        finally:
            repo.cleanup()

    def test_write_file_creates_nested_dirs(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            repo.write_file("src/deep/nested.py", "# nested\n")
            content = repo.read_file("src/deep/nested.py")
            assert content == "# nested\n"
        finally:
            repo.cleanup()

    def test_write_file_idempotent(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            repo.write_file("calculator.py", "content_a\n")
            repo.write_file("calculator.py", "content_a\n")
            assert repo.read_file("calculator.py") == "content_a\n"
        finally:
            repo.cleanup()


# ===========================================================================
# Snapshot
# ===========================================================================


class TestSnapshot:
    def test_snapshot_deterministic(self):
        fixture = make_filesystem_calculator_fixture()
        repo1 = fixture.materialize()
        repo2 = fixture.materialize()
        try:
            snap1 = repo1.snapshot()
            snap2 = repo2.snapshot()
            assert snap1 == snap2
        finally:
            repo1.cleanup()
            repo2.cleanup()

    def test_snapshot_excludes_temp_root_path(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            snap = repo.snapshot()
            # Snapshot keys are relative paths, not absolute.
            for key in snap:
                assert not os.path.isabs(key)
        finally:
            repo.cleanup()

    def test_snapshot_reflects_changes(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            snap_before = repo.snapshot()
            repo.write_file("calculator.py", FILESYSTEM_CORRECTED_CALCULATOR_CONTENT)
            snap_after = repo.snapshot()
            assert snap_before != snap_after
            assert "return a + b" in snap_after["calculator.py"]
        finally:
            repo.cleanup()


# ===========================================================================
# Fresh copies independence
# ===========================================================================


class TestFreshCopiesIndependence:
    def test_different_temp_roots(self):
        fixture = make_filesystem_calculator_fixture()
        repo1 = fixture.materialize()
        repo2 = fixture.materialize()
        try:
            assert repo1.root != repo2.root
        finally:
            repo1.cleanup()
            repo2.cleanup()

    def test_initial_snapshots_equal(self):
        fixture = make_filesystem_calculator_fixture()
        repo1 = fixture.materialize()
        repo2 = fixture.materialize()
        try:
            assert repo1.snapshot() == repo2.snapshot()
        finally:
            repo1.cleanup()
            repo2.cleanup()

    def test_mutating_one_does_not_affect_other(self):
        fixture = make_filesystem_calculator_fixture()
        repo1 = fixture.materialize()
        repo2 = fixture.materialize()
        try:
            repo1.write_file("calculator.py", "modified\n")
            assert repo1.read_file("calculator.py") == "modified\n"
            # repo2 should still have the original buggy content.
            content2 = repo2.read_file("calculator.py")
            assert "return a - b" in content2
        finally:
            repo1.cleanup()
            repo2.cleanup()


# ===========================================================================
# Path safety
# ===========================================================================


class TestPathSafety:
    def test_absolute_path_rejected(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            with pytest.raises(FilesystemPathError):
                repo.read_file("/etc/passwd")
        finally:
            repo.cleanup()

    def test_windows_absolute_path_rejected(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            if os.name == "nt":
                # Drive-letter paths are absolute on Windows and are
                # rejected before touching the filesystem.
                with pytest.raises(FilesystemPathError):
                    repo.read_file("C:\\Windows\\System32\\drivers\\etc\\hosts")
            else:
                # On POSIX a drive-letter path is not absolute — it is
                # a relative filename inside the repo root and can
                # never escape. Since no such file exists, the safe
                # outcome is a not-found error.
                with pytest.raises(FilesystemFileNotFoundError):
                    repo.read_file("C:\\Windows\\System32\\drivers\\etc\\hosts")
        finally:
            repo.cleanup()

    def test_traversal_escape_rejected(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            with pytest.raises(FilesystemPathError):
                repo.read_file("../../etc/passwd")
        finally:
            repo.cleanup()

    def test_traversal_write_escape_rejected(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            with pytest.raises(FilesystemPathError):
                repo.write_file("../../evil.py", "malicious\n")
        finally:
            repo.cleanup()

    def test_empty_path_rejected(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        try:
            with pytest.raises(FilesystemPathError):
                repo.read_file("")
        finally:
            repo.cleanup()


# ===========================================================================
# Cleanup
# ===========================================================================


class TestCleanup:
    def test_cleanup_removes_temp_dir(self):
        fixture = make_filesystem_calculator_fixture()
        repo = fixture.materialize()
        root = repo.root
        assert root.exists()
        repo.cleanup()
        assert not root.exists()

    def test_context_manager_cleanup(self):
        fixture = make_filesystem_calculator_fixture()
        with fixture.materialize() as repo:
            assert repo.root.exists()
        assert not repo.root.exists()
