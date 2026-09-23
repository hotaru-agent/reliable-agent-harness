"""GitHub Release G3 packaging tests.

Verifies LICENSE, requirements-dev.txt, README license section, and
public-facing reproduction workflow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# LICENSE
# ---------------------------------------------------------------------------

class TestLicense:
    def test_license_exists(self):
        assert (_PROJECT_ROOT / "LICENSE").exists()

    def test_license_is_mit(self):
        text = _read(_PROJECT_ROOT / "LICENSE")
        assert "MIT License" in text

    def test_license_has_copyright(self):
        text = _read(_PROJECT_ROOT / "LICENSE")
        assert "Copyright (c) 2026" in text

    def test_license_has_permission_notice(self):
        text = _read(_PROJECT_ROOT / "LICENSE")
        assert "Permission is hereby granted" in text


# ---------------------------------------------------------------------------
# README license section
# ---------------------------------------------------------------------------

class TestReadmeLicense:
    def test_readme_links_license(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "LICENSE" in text

    def test_readme_mentions_mit(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "MIT" in text


# ---------------------------------------------------------------------------
# requirements-dev.txt
# ---------------------------------------------------------------------------

class TestRequirementsDev:
    def test_requirements_dev_exists(self):
        assert (_PROJECT_ROOT / "requirements-dev.txt").exists()

    def test_references_requirements(self):
        text = _read(_PROJECT_ROOT / "requirements-dev.txt")
        assert "-r requirements.txt" in text

    def test_declares_pytest(self):
        text = _read(_PROJECT_ROOT / "requirements-dev.txt")
        assert "pytest" in text

    def test_no_unnecessary_tools(self):
        text = _read(_PROJECT_ROOT / "requirements-dev.txt")
        # These should NOT be declared unless actually used
        for tool in ("black", "ruff", "mypy", "coverage", "tox", "nox",
                     "flake8", "isort"):
            assert tool not in text.lower(), f"{tool} should not be in requirements-dev.txt"


# ---------------------------------------------------------------------------
# requirements.txt
# ---------------------------------------------------------------------------

class TestRequirements:
    def test_requirements_exists(self):
        assert (_PROJECT_ROOT / "requirements.txt").exists()

    def test_has_mcp(self):
        text = _read(_PROJECT_ROOT / "requirements.txt")
        assert "mcp" in text

    def test_has_opentelemetry(self):
        text = _read(_PROJECT_ROOT / "requirements.txt")
        assert "opentelemetry" in text

    def test_no_pytest_in_runtime(self):
        """pytest should be in requirements-dev.txt, not requirements.txt."""
        text = _read(_PROJECT_ROOT / "requirements.txt")
        assert "pytest" not in text


# ---------------------------------------------------------------------------
# Reproduction workflow
# ---------------------------------------------------------------------------

class TestReproductionWorkflow:
    def test_reproducing_exists(self):
        assert (_PROJECT_ROOT / "REPRODUCING.md").exists()

    def test_uses_requirements_dev(self):
        text = _read(_PROJECT_ROOT / "REPRODUCING.md")
        assert "requirements-dev.txt" in text

    def test_no_parent_venv_dependency(self):
        text = _read(_PROJECT_ROOT / "REPRODUCING.md")
        assert "..\\.venv" not in text
        assert ".." + chr(92) + ".venv" not in text

    def test_mentions_python_3_11_9(self):
        text = _read(_PROJECT_ROOT / "REPRODUCING.md")
        assert "3.11.9" in text

    def test_has_venv_creation(self):
        text = _read(_PROJECT_ROOT / "REPRODUCING.md")
        assert "venv" in text

    def test_has_pip_install(self):
        text = _read(_PROJECT_ROOT / "REPRODUCING.md")
        assert "pip install" in text

    def test_has_warnings_command(self):
        text = _read(_PROJECT_ROOT / "REPRODUCING.md")
        assert "DeprecationWarning" in text
        assert "ResourceWarning" in text

    def test_has_expected_baseline(self):
        text = _read(_PROJECT_ROOT / "REPRODUCING.md")
        assert "1521" in text

    def test_no_python_3_9_claim(self):
        text = _read(_PROJECT_ROOT / "REPRODUCING.md")
        assert "3.9+" not in text
        assert "Python 3.9" not in text

    def test_states_python_3_10_floor(self):
        text = _read(_PROJECT_ROOT / "REPRODUCING.md")
        assert "3.10" in text


# ---------------------------------------------------------------------------
# README Quick Start
# ---------------------------------------------------------------------------

class TestReadmeQuickStart:
    def test_uses_requirements_dev(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "requirements-dev.txt" in text

    def test_no_parent_venv(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "..\\.venv" not in text
        assert ".." + chr(92) + ".venv" not in text

    def test_has_venv_creation(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "venv" in text

    def test_has_baseline(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "1521" in text

    def test_no_python_3_9_claim(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "3.9+" not in text
        assert "Python 3.9" not in text


# ---------------------------------------------------------------------------
# No .env.example (not needed)
# ---------------------------------------------------------------------------

class TestNoEnvExample:
    def test_no_env_example(self):
        """No .env file is needed for the default test/benchmark workflow."""
        assert not (_PROJECT_ROOT / ".env.example").exists()


# ---------------------------------------------------------------------------
# .gitignore still covers key patterns
# ---------------------------------------------------------------------------

class TestGitignore:
    def test_has_venv(self):
        text = _read(_PROJECT_ROOT / ".gitignore")
        assert ".venv/" in text

    def test_has_env(self):
        text = _read(_PROJECT_ROOT / ".gitignore")
        assert ".env" in text

    def test_has_interview_file(self):
        text = _read(_PROJECT_ROOT / ".gitignore")
        assert "面试话术.md" in text
        # docs/ is local development history and must not enter
        # the public repository.
        assert "docs/" in text


# ---------------------------------------------------------------------------
# Final benchmark report baseline
# ---------------------------------------------------------------------------

class TestFinalReportBaseline:
    def test_final_report_has_current_baseline(self):
        text = _read(_PROJECT_ROOT / "FINAL_BENCHMARK_REPORT.md")
        assert "1521" in text

    def test_final_report_no_stale_baseline(self):
        text = _read(_PROJECT_ROOT / "FINAL_BENCHMARK_REPORT.md")
        # 1444 was the Step 10 baseline, should not appear as current
        assert "1444 passed" not in text


# ---------------------------------------------------------------------------
# Dependency floor regression
# ---------------------------------------------------------------------------

class TestDependencyFloor:
    def test_mcp_requires_python_3_10(self):
        """The mcp>=2,<3 dependency requires Python 3.10+.
        Public docs must not claim Python 3.9+."""
        text = _read(_PROJECT_ROOT / "requirements.txt")
        assert "mcp>=2,<3" in text

    def test_readme_no_python_3_9(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "Python 3.9" not in text
        assert "3.9+" not in text

    def test_reproducing_no_python_3_9(self):
        text = _read(_PROJECT_ROOT / "REPRODUCING.md")
        assert "Python 3.9" not in text
        assert "3.9+" not in text
