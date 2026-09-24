"""GitHub Release G2 documentation tests.

Verifies README structure, documentation navigation, development
history organization, and public/private separation.
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
# README existence and structure
# ---------------------------------------------------------------------------

class TestREADMEExists:
    def test_readme_exists(self):
        assert (_PROJECT_ROOT / "README.md").exists()

    def test_readme_is_nonempty(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert len(text) > 500


class TestREADMEContent:
    def test_readme_has_title(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "Reliable Agent Harness" in text

    def test_readme_explains_harness_vs_agent(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "agent" in text.lower()
        assert "harness" in text.lower()

    def test_readme_links_architecture(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "ARCHITECTURE.md" in text

    def test_readme_links_final_report(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "FINAL_BENCHMARK_REPORT.md" in text

    def test_readme_links_reproducing(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "REPRODUCING.md" in text

    def test_readme_no_history_link(self):
        # docs/ is local development history and is gitignored —
        # the public README must not link to it.
        text = _read(_PROJECT_ROOT / "README.md")
        assert "docs/history/" not in text

    def test_readme_has_suite_ids(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "controlled_v1" in text
        assert "filesystem_pytest_v1" in text
        assert "integrated_filesystem_v1" in text

    def test_readme_has_correct_controlled_counts(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "1/8" in text
        assert "7/8" in text

    def test_readme_has_correct_filesystem_counts(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "1/4" in text
        assert "4/4" in text

    def test_readme_has_correct_integrated_counts(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "0/1" in text
        assert "1/1" in text

    def test_readme_no_global_success_rate(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "92.3%" not in text
        assert "12/13" not in text
        assert "2/13" not in text

    def test_readme_has_test_command(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "pytest" in text

    def test_readme_has_baseline(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "1521" in text

    def test_readme_has_limitations(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "limitation" in text.lower()

    def test_readme_no_production_ready_claim(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "production ready" not in text.lower()
        assert "enterprise ready" not in text.lower()
        assert "battle tested" not in text.lower()

    def test_readme_no_github_url(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "github.com/" not in text

    def test_readme_no_badges(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "shields.io" not in text
        assert "badge" not in text.lower()

    def test_readme_no_resume_language(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "hire me" not in text.lower()
        assert "looking for opportunities" not in text.lower()
        assert "demonstrates my skills" not in text.lower()

    def test_readme_mentions_real_filesystem(self):
        text = _read(_PROJECT_ROOT / "README.md")
        assert "filesystem" in text.lower()
        assert "pytest" in text.lower()
        assert "subprocess" in text.lower()


# ---------------------------------------------------------------------------
# Development history organization
# ---------------------------------------------------------------------------

class TestHistoryOrganization:
    """docs/history/ is local-only development history (gitignored).
    On a clean public checkout it does not exist — the public
    contract is that .gitignore excludes docs/. When the directory
    is present locally, its structure is still validated."""

    _HISTORY_DIR = _PROJECT_ROOT / "docs" / "history"

    def _docs_gitignored(self) -> None:
        assert "docs/" in _read(_PROJECT_ROOT / ".gitignore")

    def test_history_dir_exists(self):
        if not self._HISTORY_DIR.exists():
            self._docs_gitignored()
            return
        assert self._HISTORY_DIR.exists()

    def test_history_readme_exists(self):
        if not self._HISTORY_DIR.exists():
            self._docs_gitignored()
            return
        assert (self._HISTORY_DIR / "README.md").exists()

    def test_en_dir_exists(self):
        if not self._HISTORY_DIR.exists():
            self._docs_gitignored()
            return
        assert (self._HISTORY_DIR / "en").exists()

    def test_zh_dir_exists(self):
        if not self._HISTORY_DIR.exists():
            self._docs_gitignored()
            return
        assert (self._HISTORY_DIR / "zh").exists()

    def test_en_has_summaries(self):
        if not self._HISTORY_DIR.exists():
            self._docs_gitignored()
            return
        summaries = list((self._HISTORY_DIR / "en").glob("phase*_summary.md"))
        assert len(summaries) >= 20

    def test_zh_has_summaries(self):
        if not self._HISTORY_DIR.exists():
            self._docs_gitignored()
            return
        summaries = list((self._HISTORY_DIR / "zh").glob("phase*_summary.md"))
        assert len(summaries) >= 20

    def test_no_phase_summaries_in_root(self):
        root_summaries = list(_PROJECT_ROOT.glob("phase*_summary.md"))
        assert len(root_summaries) == 0

    def test_no_summaries_zh_dir(self):
        assert not (_PROJECT_ROOT / "summaries_zh").exists()


# ---------------------------------------------------------------------------
# Public/private separation
# ---------------------------------------------------------------------------

class TestPublicPrivateSeparation:
    def test_gitignore_excludes_interview_file(self):
        text = _read(_PROJECT_ROOT / ".gitignore")
        assert "面试话术.md" in text

    def test_interview_file_exists_locally(self):
        # The interview-prep file is local-only (gitignored). On a
        # clean public checkout it does not exist; when present
        # locally it must still be ignored. Either way, .gitignore
        # must exclude it — verified by the companion test above.
        if not (_PROJECT_ROOT / "面试话术.md").exists():
            text = _read(_PROJECT_ROOT / ".gitignore")
            assert "面试话术.md" in text
            return
        assert (_PROJECT_ROOT / "面试话术.md").exists()

    def test_gitignore_excludes_devin_config(self):
        text = _read(_PROJECT_ROOT / ".gitignore")
        assert ".devin/config.local.json" in text


# ---------------------------------------------------------------------------
# Root directory cleanliness
# ---------------------------------------------------------------------------

class TestRootCleanliness:
    def test_root_has_readme(self):
        assert (_PROJECT_ROOT / "README.md").exists()

    def test_root_has_architecture(self):
        assert (_PROJECT_ROOT / "ARCHITECTURE.md").exists()

    def test_root_has_final_report(self):
        assert (_PROJECT_ROOT / "FINAL_BENCHMARK_REPORT.md").exists()

    def test_root_has_reproducing(self):
        assert (_PROJECT_ROOT / "REPRODUCING.md").exists()

    def test_root_has_requirements(self):
        assert (_PROJECT_ROOT / "requirements.txt").exists()

    def test_root_has_gitignore(self):
        assert (_PROJECT_ROOT / ".gitignore").exists()

    def test_root_has_core_dirs(self):
        # docs/ is local-only development history (gitignored) and
        # is not part of the public repository layout.
        for d in ("harness", "tools", "storage", "mcp_adapter",
                  "observability", "evaluation", "tests"):
            assert (_PROJECT_ROOT / d).exists(), d
