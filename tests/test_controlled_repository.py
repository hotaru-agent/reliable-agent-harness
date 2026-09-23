"""Tests for ControlledRepository (Phase 7 Step 1).

Covers:
* Initial repository fails oracle (buggy add)
* read_file returns expected content
* write_file changes repository state
* Correct patch causes tests to pass
* reset / fresh fixture restores original bug
* Unknown file behavior deterministic
* Repository snapshots do not alias mutable state

All offline, deterministic, no real filesystem, no pytest subprocess.
"""

from __future__ import annotations

import pytest

from evaluation.repository import (
    ControlledRepository,
    RepositoryFileNotFoundError,
    RepositoryFixture,
    RepositoryOracle,
    RepositoryTestResult,
    TestDefinition,
)
from evaluation.scenarios.controlled_repo import (
    CORRECTED_CALCULATOR_CONTENT,
    make_calculator_fixture,
)


# ===========================================================================
# Repository test result validation
# ===========================================================================


class TestRepositoryTestResult:
    def test_valid_passing_result(self):
        result = RepositoryTestResult(
            passed=True, total=3, passed_count=3, failed_count=0,
        )
        assert result.passed is True
        assert result.failing_tests == ()

    def test_valid_failing_result(self):
        result = RepositoryTestResult(
            passed=False, total=3, passed_count=2, failed_count=1,
            failing_tests=("test_add",),
        )
        assert result.passed is False
        assert result.failing_tests == ("test_add",)

    def test_count_mismatch_rejected(self):
        with pytest.raises(ValueError, match="passed_count"):
            RepositoryTestResult(
                passed=False, total=3, passed_count=1, failed_count=1,
            )

    def test_passed_with_failures_rejected(self):
        with pytest.raises(ValueError, match="passed=True"):
            RepositoryTestResult(
                passed=True, total=3, passed_count=2, failed_count=1,
                failing_tests=("test_add",),
            )

    def test_failed_with_zero_failures_rejected(self):
        with pytest.raises(ValueError, match="passed=False"):
            RepositoryTestResult(
                passed=False, total=3, passed_count=3, failed_count=0,
            )

    def test_is_frozen(self):
        result = RepositoryTestResult(
            passed=True, total=1, passed_count=1, failed_count=0,
        )
        with pytest.raises(Exception):
            result.passed = False  # type: ignore[misc]


# ===========================================================================
# Initial repository state
# ===========================================================================


class TestInitialRepository:
    def test_initial_repo_fails_oracle(self):
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        oracle = RepositoryOracle()
        assert oracle.is_complete(repo) is False

    def test_initial_tests_show_failure(self):
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        result = repo.run_tests()
        assert result.passed is False
        assert result.total == 2
        assert result.failed_count == 1
        assert "test_add" in result.failing_tests
        assert "test_sub" not in result.failing_tests

    def test_initial_files_listed(self):
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        files = repo.list_files()
        assert "calculator.py" in files
        assert "tests/test_calculator.py" in files


# ===========================================================================
# read_file
# ===========================================================================


class TestReadFile:
    def test_read_file_returns_content(self):
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        content = repo.read_file("calculator.py")
        assert "def add(a, b):" in content
        assert "return a - b" in content  # the bug

    def test_read_unknown_file_raises(self):
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        with pytest.raises(RepositoryFileNotFoundError):
            repo.read_file("nonexistent.py")

    def test_read_empty_path_rejected(self):
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        with pytest.raises(ValueError):
            repo.read_file("")


# ===========================================================================
# write_file
# ===========================================================================


class TestWriteFile:
    def test_write_file_changes_state(self):
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        original = repo.read_file("calculator.py")
        repo.write_file("calculator.py", "new content")
        assert repo.read_file("calculator.py") == "new content"
        assert repo.read_file("calculator.py") != original

    def test_write_file_creates_new_file(self):
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        repo.write_file("new_file.py", "print('hello')")
        assert "new_file.py" in repo.list_files()
        assert repo.read_file("new_file.py") == "print('hello')"

    def test_write_file_idempotent_same_content(self):
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        repo.write_file("calculator.py", "same")
        state1 = dict(repo.files)
        repo.write_file("calculator.py", "same")
        state2 = dict(repo.files)
        assert state1 == state2

    def test_write_empty_path_rejected(self):
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        with pytest.raises(ValueError):
            repo.write_file("", "content")

    def test_write_non_string_content_rejected(self):
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        with pytest.raises(ValueError):
            repo.write_file("f.py", 123)  # type: ignore[arg-type]


# ===========================================================================
# Correct patch causes tests to pass
# ===========================================================================


class TestCorrectPatch:
    def test_correct_patch_makes_tests_pass(self):
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        oracle = RepositoryOracle()

        # Before fix: incomplete.
        assert oracle.is_complete(repo) is False

        # Apply the fix.
        repo.write_file("calculator.py", CORRECTED_CALCULATOR_CONTENT)

        # After fix: complete.
        assert oracle.is_complete(repo) is True

    def test_correct_patch_all_tests_pass(self):
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        repo.write_file("calculator.py", CORRECTED_CALCULATOR_CONTENT)
        result = repo.run_tests()
        assert result.passed is True
        assert result.failed_count == 0
        assert result.passed_count == 2


# ===========================================================================
# Reset / fresh fixture
# ===========================================================================


class TestResetAndFreshFixture:
    def test_reset_restores_original_bug(self):
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()

        # Fix the bug.
        repo.write_file("calculator.py", CORRECTED_CALCULATOR_CONTENT)
        assert repo.run_tests().passed is True

        # Reset.
        repo.reset()
        assert repo.run_tests().passed is False
        assert "return a - b" in repo.read_file("calculator.py")

    def test_fresh_fixture_not_modified_by_prior_run(self):
        fixture = make_calculator_fixture()

        # Run A: modify the repository.
        repo_a = fixture.create_repository()
        repo_a.write_file("calculator.py", CORRECTED_CALCULATOR_CONTENT)
        assert repo_a.run_tests().passed is True

        # Run B: fresh repository from same fixture.
        repo_b = fixture.create_repository()
        assert repo_b.run_tests().passed is False
        assert "return a - b" in repo_b.read_file("calculator.py")

    def test_reset_after_creating_new_file(self):
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        repo.write_file("new.py", "x")
        assert "new.py" in repo.list_files()
        repo.reset()
        assert "new.py" not in repo.list_files()


# ===========================================================================
# Snapshots do not alias mutable state
# ===========================================================================


class TestSnapshotIsolation:
    def test_two_repos_from_same_fixture_do_not_share_state(self):
        fixture = make_calculator_fixture()
        repo_a = fixture.create_repository()
        repo_b = fixture.create_repository()

        repo_a.write_file("calculator.py", "modified by A")
        assert repo_b.read_file("calculator.py") != "modified by A"

    def test_fixture_files_not_mutated_by_repo(self):
        fixture = make_calculator_fixture()
        original_files = dict(fixture.files)
        repo = fixture.create_repository()
        repo.write_file("calculator.py", "changed")
        assert fixture.files == original_files


# ===========================================================================
# Deterministic behavior
# ===========================================================================


class TestDeterministicBehavior:
    def test_run_tests_deterministic(self):
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        r1 = repo.run_tests()
        r2 = repo.run_tests()
        assert r1 == r2

    def test_list_files_sorted(self):
        fixture = make_calculator_fixture()
        repo = fixture.create_repository()
        files = repo.list_files()
        assert list(files) == sorted(files)


# ===========================================================================
# TestDefinition validation
# ===========================================================================


class TestTestDefinition:
    def test_valid_test_definition(self):
        td = TestDefinition(name="test_x", checker=lambda f: True)
        assert td.name == "test_x"

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="name"):
            TestDefinition(name="", checker=lambda f: True)

    def test_non_callable_checker_rejected(self):
        with pytest.raises(ValueError, match="checker"):
            TestDefinition(name="test_x", checker="not callable")  # type: ignore[arg-type]


# ===========================================================================
# RepositoryFixture validation
# ===========================================================================


class TestRepositoryFixture:
    def test_valid_fixture(self):
        fixture = RepositoryFixture(
            fixture_id="f1",
            files={"a.py": "x"},
            test_definitions=(TestDefinition(name="t1", checker=lambda f: True),),
        )
        assert fixture.fixture_id == "f1"

    def test_empty_fixture_id_rejected(self):
        with pytest.raises(ValueError, match="fixture_id"):
            RepositoryFixture(fixture_id="", files={})

    def test_fixture_is_frozen(self):
        fixture = RepositoryFixture(fixture_id="f1", files={"a.py": "x"})
        with pytest.raises(Exception):
            fixture.fixture_id = "f2"  # type: ignore[misc]
