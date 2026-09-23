"""Tests for benchmark data model contracts (Phase 7 Step 1).

Covers:
* BenchmarkTask validation (empty IDs, max_logical_actions)
* BenchmarkScenario validation (association, fixture_id)
* Metric definitions completeness
* Fault definitions immutability

All offline, deterministic.
"""

from __future__ import annotations

import pytest

from evaluation.models import (
    BenchmarkConfigurationError,
    BenchmarkScenario,
    BenchmarkTask,
    METRIC_DEFINITIONS,
    MetricDefinition,
    MetricName,
    UnknownScenarioError,
)


# ===========================================================================
# BenchmarkTask validation
# ===========================================================================


class TestBenchmarkTaskValidation:
    def test_valid_task(self):
        task = BenchmarkTask(
            task_id="T-001",
            goal="Fix the bug",
            scenario_id="clean_success",
            max_logical_actions=10,
        )
        assert task.task_id == "T-001"
        assert task.goal == "Fix the bug"
        assert task.scenario_id == "clean_success"
        assert task.max_logical_actions == 10

    def test_empty_task_id_rejected(self):
        with pytest.raises(ValueError, match="task_id"):
            BenchmarkTask(
                task_id="",
                goal="Fix the bug",
                scenario_id="s1",
                max_logical_actions=10,
            )

    def test_whitespace_task_id_rejected(self):
        with pytest.raises(ValueError, match="task_id"):
            BenchmarkTask(
                task_id="   ",
                goal="Fix the bug",
                scenario_id="s1",
                max_logical_actions=10,
            )

    def test_empty_goal_rejected(self):
        with pytest.raises(ValueError, match="goal"):
            BenchmarkTask(
                task_id="T-001",
                goal="",
                scenario_id="s1",
                max_logical_actions=10,
            )

    def test_empty_scenario_id_rejected(self):
        with pytest.raises(ValueError, match="scenario_id"):
            BenchmarkTask(
                task_id="T-001",
                goal="Fix the bug",
                scenario_id="",
                max_logical_actions=10,
            )

    def test_max_logical_actions_zero_rejected(self):
        with pytest.raises(ValueError, match="max_logical_actions"):
            BenchmarkTask(
                task_id="T-001",
                goal="Fix the bug",
                scenario_id="s1",
                max_logical_actions=0,
            )

    def test_max_logical_actions_negative_rejected(self):
        with pytest.raises(ValueError, match="max_logical_actions"):
            BenchmarkTask(
                task_id="T-001",
                goal="Fix the bug",
                scenario_id="s1",
                max_logical_actions=-1,
            )

    def test_max_logical_actions_must_be_int(self):
        with pytest.raises(ValueError, match="max_logical_actions"):
            BenchmarkTask(
                task_id="T-001",
                goal="Fix the bug",
                scenario_id="s1",
                max_logical_actions=10.5,
            )

    def test_max_logical_actions_bool_rejected(self):
        with pytest.raises(ValueError, match="max_logical_actions"):
            BenchmarkTask(
                task_id="T-001",
                goal="Fix the bug",
                scenario_id="s1",
                max_logical_actions=True,
            )

    def test_task_is_frozen(self):
        task = BenchmarkTask(
            task_id="T-001",
            goal="Fix the bug",
            scenario_id="s1",
            max_logical_actions=10,
        )
        with pytest.raises(Exception):
            task.task_id = "T-002"  # type: ignore[misc]


# ===========================================================================
# BenchmarkScenario validation
# ===========================================================================


class TestBenchmarkScenarioValidation:
    def _make_task(self, scenario_id="s1"):
        return BenchmarkTask(
            task_id="T-001",
            goal="Fix the bug",
            scenario_id=scenario_id,
            max_logical_actions=10,
        )

    def test_valid_scenario(self):
        task = self._make_task()
        scenario = BenchmarkScenario(
            scenario_id="s1",
            task=task,
            fixture_id="calculator_buggy_add",
            fault_plan_id="none",
            description="Clean run",
        )
        assert scenario.scenario_id == "s1"
        assert scenario.fixture_id == "calculator_buggy_add"

    def test_empty_scenario_id_rejected(self):
        task = self._make_task()
        with pytest.raises(ValueError, match="scenario_id"):
            BenchmarkScenario(
                scenario_id="",
                task=task,
                fixture_id="f1",
            )

    def test_empty_fixture_id_rejected(self):
        task = self._make_task()
        with pytest.raises(ValueError, match="fixture_id"):
            BenchmarkScenario(
                scenario_id="s1",
                task=task,
                fixture_id="",
            )

    def test_empty_fault_plan_id_defaults_to_none(self):
        task = self._make_task()
        scenario = BenchmarkScenario(
            scenario_id="s1",
            task=task,
            fixture_id="f1",
        )
        assert scenario.fault_plan_id == "none"

    def test_task_scenario_id_mismatch_rejected(self):
        task = self._make_task(scenario_id="other")
        with pytest.raises(ValueError, match="scenario_id"):
            BenchmarkScenario(
                scenario_id="s1",
                task=task,
                fixture_id="f1",
            )

    def test_scenario_is_frozen(self):
        task = self._make_task()
        scenario = BenchmarkScenario(
            scenario_id="s1",
            task=task,
            fixture_id="f1",
        )
        with pytest.raises(Exception):
            scenario.scenario_id = "s2"  # type: ignore[misc]


# ===========================================================================
# Metric definitions
# ===========================================================================


class TestMetricDefinitions:
    def test_all_metric_names_have_definitions(self):
        for name in MetricName:
            assert name in METRIC_DEFINITIONS, f"missing definition for {name}"

    def test_all_definitions_have_correct_name(self):
        for name, defn in METRIC_DEFINITIONS.items():
            assert defn.name is name

    def test_all_definitions_have_descriptions(self):
        for defn in METRIC_DEFINITIONS.values():
            assert defn.description and defn.description.strip()

    def test_recovery_success_requires_interruption(self):
        assert METRIC_DEFINITIONS[MetricName.RECOVERY_SUCCESS].requires_interruption is True

    def test_task_completion_does_not_require_interruption(self):
        assert METRIC_DEFINITIONS[MetricName.TASK_COMPLETION].requires_interruption is False

    def test_duplicate_execution_lower_is_better(self):
        assert METRIC_DEFINITIONS[MetricName.DUPLICATE_EXECUTION].higher_is_better is False

    def test_unnecessary_retry_lower_is_better(self):
        assert METRIC_DEFINITIONS[MetricName.UNNECESSARY_RETRY].higher_is_better is False

    def test_metric_count(self):
        assert len(MetricName) == 10

    def test_metric_definition_is_frozen(self):
        defn = METRIC_DEFINITIONS[MetricName.TASK_COMPLETION]
        with pytest.raises(Exception):
            defn.description = "changed"  # type: ignore[misc]


# ===========================================================================
# Benchmark errors
# ===========================================================================


class TestBenchmarkErrors:
    def test_unknown_scenario_error_is_configuration_error(self):
        assert issubclass(UnknownScenarioError, BenchmarkConfigurationError)

    def test_configuration_error_is_exception(self):
        assert issubclass(BenchmarkConfigurationError, Exception)
