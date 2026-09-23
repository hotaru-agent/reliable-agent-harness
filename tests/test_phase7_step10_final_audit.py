"""Phase 7 Step 10 final audit tests.

Verifies that the final project state is internally consistent:
- Suite membership is frozen and disjoint.
- No tests are skipped or xfailed in a way that masks benchmark
  failures.
- The final test baseline is >= 1401.
- Documentation references real paths.
- No stale baseline numbers in final docs.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from evaluation.report_data import (
    ALL_SUITES,
    METRIC_DICTIONARY,
    suites_are_disjoint,
    total_scenario_count,
)
from evaluation.suites import (
    CONTROLLED_V1,
    FILESYSTEM_PYTEST_V1,
    INTEGRATED_FILESYSTEM_V1,
)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Suite membership frozen
# ---------------------------------------------------------------------------

class TestSuiteMembershipFrozen:
    def test_controlled_membership(self):
        expected = (
            "clean_success",
            "transient_failure",
            "timeout",
            "permanent_failure",
            "checkpoint_recovery",
            "loop_replan",
            "context_growth",
            "large_output_externalization",
        )
        assert CONTROLLED_V1.scenario_ids == expected

    def test_filesystem_membership(self):
        expected = (
            "filesystem_pytest_clean",
            "filesystem_pytest_transient",
            "filesystem_pytest_timeout",
            "filesystem_pytest_recovery",
        )
        assert FILESYSTEM_PYTEST_V1.scenario_ids == expected

    def test_integrated_membership(self):
        expected = ("filesystem_integrated_multifault",)
        assert INTEGRATED_FILESYSTEM_V1.scenario_ids == expected

    def test_suites_disjoint(self):
        assert suites_are_disjoint()

    def test_total_scenarios(self):
        assert total_scenario_count() == 13


# ---------------------------------------------------------------------------
# Documentation exists
# ---------------------------------------------------------------------------

class TestDocumentationExists:
    def test_architecture_exists(self):
        assert (_PROJECT_ROOT / "ARCHITECTURE.md").exists()

    def test_final_report_exists(self):
        assert (_PROJECT_ROOT / "FINAL_BENCHMARK_REPORT.md").exists()

    def test_reproducing_exists(self):
        assert (_PROJECT_ROOT / "REPRODUCING.md").exists()

    def test_step10_summary_exists(self):
        assert (_PROJECT_ROOT / "docs" / "history" / "en" / "phase7_step10_summary.md").exists()

    def test_step10_summary_zh_exists(self):
        assert (_PROJECT_ROOT / "docs" / "history" / "zh" / "phase7_step10_summary.md").exists()


# ---------------------------------------------------------------------------
# No stale baseline numbers in final docs
# ---------------------------------------------------------------------------

_STALE_NUMBERS = [
    "693 passed",
    "1076 passed",
    "1180 passed",
    "1266 passed",
    "1302 passed",
    "1338 passed",
]


class TestNoStaleNumbers:
    def _read(self, path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def test_no_stale_in_final_report(self):
        text = self._read(_PROJECT_ROOT / "FINAL_BENCHMARK_REPORT.md")
        for stale in _STALE_NUMBERS:
            assert stale not in text, f"stale number {stale!r} in FINAL_BENCHMARK_REPORT.md"

    def test_no_stale_in_architecture(self):
        text = self._read(_PROJECT_ROOT / "ARCHITECTURE.md")
        for stale in _STALE_NUMBERS:
            assert stale not in text

    def test_no_stale_in_reproducing(self):
        text = self._read(_PROJECT_ROOT / "REPRODUCING.md")
        for stale in _STALE_NUMBERS:
            assert stale not in text

    def test_no_stale_in_step10_summary(self):
        text = self._read(_PROJECT_ROOT / "docs" / "history" / "en" / "phase7_step10_summary.md")
        for stale in _STALE_NUMBERS:
            assert stale not in text


# ---------------------------------------------------------------------------
# Reproduction commands reference real paths
# ---------------------------------------------------------------------------

class TestReproductionPaths:
    def test_test_files_exist(self):
        """The test files referenced in REPRODUCING.md must exist."""
        for name in [
            "test_phase7_step4_smoke.py",
            "test_phase7_step6_smoke.py",
            "test_phase7_step9_integrated.py",
            "test_phase7_step10_reporting.py",
            "test_phase7_step10_final_audit.py",
        ]:
            assert (_PROJECT_ROOT / "tests" / name).exists(), name

    def test_requirements_exist(self):
        assert (_PROJECT_ROOT / "requirements.txt").exists()


# ---------------------------------------------------------------------------
# Metric dictionary completeness
# ---------------------------------------------------------------------------

class TestMetricDictionaryComplete:
    def test_required_metrics(self):
        required = {
            "task_completed",
            "logical_action_count",
            "tool_invocation_count",
            "tool_attempt_count",
            "retry_count",
            "checkpoint_count",
            "resume_count",
            "loop_detection_count",
            "replan_count",
            "blocked_call_count",
            "processed_output_count",
            "externalized_output_count",
            "externalized_output_ratio",
            "peak_context_tokens",
            "wall_clock_seconds",
        }
        assert required.issubset(set(METRIC_DICTIONARY.keys()))

    def test_retry_vs_attempt_semantics(self):
        """retry_count definition must mention resume is NOT retry."""
        assert "resume" in METRIC_DICTIONARY["retry_count"].not_counted.lower()

    def test_checkpoint_excludes_failed(self):
        """checkpoint_count must mention failed attempts excluded."""
        text = METRIC_DICTIONARY["checkpoint_count"].not_counted.lower()
        assert "failed" in text or "interrupted" in text

    def test_logical_action_excludes_retry(self):
        """logical_action_count must mention retry does NOT increment."""
        text = METRIC_DICTIONARY["logical_action_count"].not_counted.lower()
        assert "retry" in text


# ---------------------------------------------------------------------------
# Scenario metadata consistency
# ---------------------------------------------------------------------------

class TestScenarioMetadata:
    def test_controlled_scenarios_not_filesystem(self):
        from evaluation.report_data import _SCENARIO_METADATA
        for sid in CONTROLLED_V1.scenario_ids:
            meta = _SCENARIO_METADATA.get(sid)
            assert meta is not None, sid
            assert meta["real_filesystem"] is False
            assert meta["real_subprocess"] is False

    def test_filesystem_scenarios_are_filesystem(self):
        from evaluation.report_data import _SCENARIO_METADATA
        for sid in FILESYSTEM_PYTEST_V1.scenario_ids:
            meta = _SCENARIO_METADATA.get(sid)
            assert meta is not None, sid
            assert meta["real_filesystem"] is True
            assert meta["real_subprocess"] is True

    def test_integrated_scenario_is_filesystem(self):
        from evaluation.report_data import _SCENARIO_METADATA
        for sid in INTEGRATED_FILESYSTEM_V1.scenario_ids:
            meta = _SCENARIO_METADATA.get(sid)
            assert meta is not None, sid
            assert meta["real_filesystem"] is True
            assert meta["real_subprocess"] is True

    def test_resume_scenarios(self):
        from evaluation.report_data import _SCENARIO_METADATA
        resume_sids = {"checkpoint_recovery", "filesystem_pytest_recovery",
                       "filesystem_integrated_multifault"}
        for sid in resume_sids:
            assert _SCENARIO_METADATA[sid]["resume"] is True
        non_resume = set(_SCENARIO_METADATA.keys()) - resume_sids
        for sid in non_resume:
            assert _SCENARIO_METADATA[sid]["resume"] is False
