"""Evaluation / benchmark foundation (Phase 7 Step 1 + Step 2).

Controlled Repository Benchmark & Deterministic Fault-Injection Foundation
with comparable benchmark runners.

All offline, deterministic, no LLM, no network, no real filesystem.
"""

from evaluation.comparison import BenchmarkComparison, BenchmarkComparator
from evaluation.context_workload import (
    CharTokenEstimator,
    ContextBenchmarkStepExecutor,
    ContextLimitExceededError,
    ContextTrialState,
    ContextWorkload,
    NaiveContextExecutor,
)
from evaluation.filesystem_repository import (
    FilesystemFileNotFoundError,
    FilesystemPathError,
    FilesystemRepository,
    FilesystemRepositoryError,
    FilesystemRepositoryFixture,
    FilesystemRepositoryOracle,
    PytestSubprocessResult,
)
from evaluation.filesystem_runners import (
    FilesystemNaiveBenchmarkRunner,
    FilesystemReliableHarnessBenchmarkRunner,
    FilesystemTrialState,
)
from evaluation.faults import (
    FaultInjector,
    FaultPlan,
    FaultSpec,
    FaultType,
    InjectedProcessInterruption,
)
from evaluation.invoker import FaultAwareToolInvoker
from evaluation.loop_workload import (
    DeterministicReplanStrategy,
    LoopBenchmarkStepExecutor,
    ReplanAwareActionSource,
    ReplanPlan,
    RepositoryProgressProvider,
)
from evaluation.models import (
    BenchmarkConfigurationError,
    BenchmarkScenario,
    BenchmarkTask,
    MetricDefinition,
    MetricName,
    METRIC_DEFINITIONS,
    UnknownScenarioError,
)
from evaluation.records import (
    BenchmarkEvent,
    BenchmarkEventType,
    BenchmarkResult,
    BenchmarkRunRecord,
)
from evaluation.repository import (
    ControlledRepository,
    RepositoryFileNotFoundError,
    RepositoryFixture,
    RepositoryOracle,
    RepositoryTestResult,
    TestDefinition,
)
from evaluation.runners import (
    BenchmarkRunner,
    BenchmarkStepExecutor,
    BenchmarkStepFailure,
    MaxActionsExceededError,
    NaiveBenchmarkRunner,
    ReliableHarnessBenchmarkRunner,
)
from evaluation.scenarios.controlled_repo import (
    ScriptedActionSource,
    make_fix_calculator_actions,
    make_repository_tool_handlers,
    make_repository_tool_specs,
    make_standard_fault_plans,
    make_standard_fixtures,
    make_standard_scenarios,
)
from evaluation.suites import (
    BenchmarkSuite,
    CONTROLLED_V1,
    FILESYSTEM_PYTEST_V1,
    make_standard_suites,
)
from evaluation.workloads import (
    LoopWorkload,
    ScriptedWorkload,
    ToolStack,
    Workload,
    WorkloadRegistry,
    make_default_workload_registry,
)

__all__ = [
    "BenchmarkComparison",
    "BenchmarkComparator",
    "BenchmarkConfigurationError",
    "BenchmarkEvent",
    "BenchmarkEventType",
    "BenchmarkResult",
    "BenchmarkRunRecord",
    "BenchmarkRunner",
    "BenchmarkScenario",
    "BenchmarkStepExecutor",
    "BenchmarkStepFailure",
    "BenchmarkSuite",
    "BenchmarkTask",
    "CharTokenEstimator",
    "ContextBenchmarkStepExecutor",
    "ContextLimitExceededError",
    "ContextTrialState",
    "ContextWorkload",
    "CONTROLLED_V1",
    "ControlledRepository",
    "DeterministicReplanStrategy",
    "FaultAwareToolInvoker",
    "FaultInjector",
    "FaultPlan",
    "FaultSpec",
    "FaultType",
    "FilesystemFileNotFoundError",
    "FilesystemNaiveBenchmarkRunner",
    "FilesystemPathError",
    "FilesystemReliableHarnessBenchmarkRunner",
    "FilesystemRepository",
    "FilesystemRepositoryError",
    "FilesystemRepositoryFixture",
    "FilesystemRepositoryOracle",
    "FilesystemTrialState",
    "FILESYSTEM_PYTEST_V1",
    "InjectedProcessInterruption",
    "LoopBenchmarkStepExecutor",
    "LoopWorkload",
    "MaxActionsExceededError",
    "METRIC_DEFINITIONS",
    "MetricDefinition",
    "MetricName",
    "NaiveBenchmarkRunner",
    "NaiveContextExecutor",
    "PytestSubprocessResult",
    "ReplanAwareActionSource",
    "ReplanPlan",
    "ReliableHarnessBenchmarkRunner",
    "RepositoryFileNotFoundError",
    "RepositoryFixture",
    "RepositoryOracle",
    "RepositoryProgressProvider",
    "RepositoryTestResult",
    "ScriptedWorkload",
    "TestDefinition",
    "ToolStack",
    "UnknownScenarioError",
    "Workload",
    "WorkloadRegistry",
    "make_default_workload_registry",
    "make_fix_calculator_actions",
    "make_repository_tool_handlers",
    "make_repository_tool_specs",
    "make_standard_fault_plans",
    "make_standard_fixtures",
    "make_standard_scenarios",
    "make_standard_suites",
]
