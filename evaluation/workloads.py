"""Workload registry for scenario-specific action sources and executors
(Phase 7 Step 4).

Maps ``scenario_id`` → ``Workload`` so runners can transparently select
the appropriate action source and step executor without scattering
``if scenario.scenario_id == "loop_replan"`` branches across runner /
handler code.

Workloads:
- ``ScriptedWorkload`` — for existing Step 1/2/3 scenarios. Uses
  ``ScriptedActionSource`` + ``BenchmarkStepExecutor``.
- ``LoopWorkload`` — for the ``loop_replan`` scenario. Uses
  ``ReplanAwareActionSource`` + ``LoopBenchmarkStepExecutor`` (which
  integrates the real ``AgentActionController`` + ``LoopDetector``).

All offline, deterministic, no LLM, no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

from evaluation.loop_workload import (
    DeterministicReplanStrategy,
    LoopBenchmarkStepExecutor,
    ReplanAwareActionSource,
    RepositoryProgressProvider,
    make_loop_looping_call,
    make_loop_repair_calls,
)
from evaluation.context_workload import (
    ContextWorkload,
)
from evaluation.repository import ControlledRepository
from evaluation.scenarios.controlled_repo import (
    CORRECTED_CALCULATOR_CONTENT,
    ScriptedActionSource,
    make_context_growth_actions,
    make_large_output_actions,
)
from harness.action_controller import AgentActionController
from harness.loop_detection import (
    ActionHistory,
    LoopDetector,
    LoopDetectorConfig,
)
from tools import ToolExecutionContext, ToolRuntime
from evaluation.invoker import FaultAwareToolInvoker


# ===========================================================================
# Tool stack (built by the runner, passed to the workload)
# ===========================================================================


@dataclass(frozen=True)
class ToolStack:
    """The tool execution stack built by the runner.

    Attributes:
        runtime: The ``ToolRuntime`` (for ``AgentActionController``).
        invoker: The ``FaultAwareToolInvoker`` (for
            ``BenchmarkStepExecutor``). Wraps ``runtime``; when no
            injector is active, it is a pass-through.
        ctx: The ``ToolExecutionContext``.
    """

    runtime: ToolRuntime
    invoker: FaultAwareToolInvoker
    ctx: ToolExecutionContext


# ===========================================================================
# Workload protocol
# ===========================================================================


class Workload(Protocol):
    """A workload that can create fresh action sources and step
    executors for a benchmark scenario.

    Each call to ``create_source()`` or ``create_step_executor()`` must
    return a FRESH instance — no shared mutable state between trials.
    """

    def create_source(self) -> Any:
        """Return a fresh action source for this workload."""
        ...

    def create_step_executor(
        self,
        *,
        source: Any,
        tool_stack: ToolStack,
        repo: ControlledRepository,
        max_logical_actions: int,
    ) -> Any:
        """Return a fresh step executor for the Reliable runner."""
        ...


# ===========================================================================
# Scripted workload (for existing Step 1/2/3 scenarios)
# ===========================================================================


class ScriptedWorkload:
    """Workload for existing scenarios using ``ScriptedActionSource`` +
    ``BenchmarkStepExecutor``.

    This is the default workload for all non-loop scenarios.
    """

    def __init__(
        self,
        action_factory: Callable[[], list],
    ) -> None:
        self._action_factory = action_factory

    def create_source(self) -> ScriptedActionSource:
        return ScriptedActionSource(self._action_factory())

    def create_step_executor(
        self,
        *,
        source: Any,
        tool_stack: ToolStack,
        repo: ControlledRepository,
        max_logical_actions: int,
    ) -> Any:
        # Lazy import to avoid circular import (runners → workloads →
        # runners).
        from evaluation.runners import BenchmarkStepExecutor

        if not isinstance(source, ScriptedActionSource):
            raise ValueError("ScriptedWorkload requires ScriptedActionSource")
        return BenchmarkStepExecutor(
            source=source,
            invoker=tool_stack.invoker,
            ctx=tool_stack.ctx,
            max_logical_actions=max_logical_actions,
        )


# ===========================================================================
# Loop workload (for the loop_replan scenario)
# ===========================================================================


class LoopWorkload:
    """Workload for the ``loop_replan`` scenario.

    Uses ``ReplanAwareActionSource`` (two-phase: LOOPING → REPAIR) and
    ``LoopBenchmarkStepExecutor`` (integrates the real
    ``AgentActionController`` + ``LoopDetector`` + replan gate).

    The ``LoopDetectorConfig`` is explicitly configured with
    ``duplicate_threshold=3`` so the third consecutive duplicate action
    triggers ``DUPLICATE_CALL``.
    """

    def __init__(
        self,
        *,
        looping_call_factory: Callable[[], Any] = make_loop_looping_call,
        repair_call_factory: Callable[[], list] = lambda: make_loop_repair_calls(
            CORRECTED_CALCULATOR_CONTENT
        ),
        detector_config: Optional[LoopDetectorConfig] = None,
        history_size: int = 64,
    ) -> None:
        self._looping_call_factory = looping_call_factory
        self._repair_call_factory = repair_call_factory
        self._detector_config = detector_config or LoopDetectorConfig(
            duplicate_threshold=3,
        )
        self._history_size = history_size

    @property
    def detector_config(self) -> LoopDetectorConfig:
        return self._detector_config

    def create_source(self) -> ReplanAwareActionSource:
        return ReplanAwareActionSource(
            looping_call=self._looping_call_factory(),
            repair_calls=self._repair_call_factory(),
        )

    def create_step_executor(
        self,
        *,
        source: Any,
        tool_stack: ToolStack,
        repo: ControlledRepository,
        max_logical_actions: int,
    ) -> LoopBenchmarkStepExecutor:
        if not isinstance(source, ReplanAwareActionSource):
            raise ValueError("LoopWorkload requires ReplanAwareActionSource")

        # Build the real AgentActionController with the real
        # LoopDetector and RepositoryProgressProvider.
        detector = LoopDetector(
            config=self._detector_config,
            history=ActionHistory(max_size=self._history_size),
        )
        progress_provider = RepositoryProgressProvider(repo)
        controller = AgentActionController(
            tool_runtime=tool_stack.runtime,
            loop_detector=detector,
            progress_provider=progress_provider,
        )
        strategy = DeterministicReplanStrategy()

        return LoopBenchmarkStepExecutor(
            source=source,
            controller=controller,
            ctx=tool_stack.ctx,
            strategy=strategy,
            max_logical_actions=max_logical_actions,
        )


# ===========================================================================
# Workload registry
# ===========================================================================


class WorkloadRegistry:
    """Maps ``scenario_id`` → ``Workload``.

    Runners use this to select the appropriate workload for each
    scenario without scattering ``if scenario_id == ...`` branches.

    If a scenario_id is not registered, the default workload is used.
    """

    def __init__(
        self,
        default_workload: Workload,
    ) -> None:
        self._default = default_workload
        self._workloads: dict[str, Workload] = {}

    def register(self, scenario_id: str, workload: Workload) -> None:
        if not scenario_id or not scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty str")
        self._workloads[scenario_id] = workload

    def get(self, scenario_id: str) -> Workload:
        return self._workloads.get(scenario_id, self._default)


def make_default_workload_registry(
    action_factory: Callable[[], list],
) -> WorkloadRegistry:
    """Return a ``WorkloadRegistry`` with the default configuration.

    - Default workload: ``ScriptedWorkload`` (for all existing
      Step 1/2/3 scenarios).
    - Registered: ``loop_replan`` → ``LoopWorkload``.
    - Registered: ``context_growth`` → ``ContextWorkload`` (context
      growth scenario with bounded context capacity).
    - Registered: ``large_output_externalization`` → ``ContextWorkload``
      (large output externalization scenario with bounded context
      capacity).
    """
    registry = WorkloadRegistry(
        default_workload=ScriptedWorkload(action_factory=action_factory),
    )
    registry.register("loop_replan", LoopWorkload())
    registry.register("context_growth", ContextWorkload(
        action_factory=make_context_growth_actions,
        max_context_tokens=120,
        externalization_policy=_make_step5_externalization_policy(),
    ))
    registry.register("large_output_externalization", ContextWorkload(
        action_factory=make_large_output_actions,
        max_context_tokens=120,
        externalization_policy=_make_step5_externalization_policy(),
    ))
    return registry


def _make_step5_externalization_policy():
    """Return the externalization policy for Step 5 scenarios.

    The ``inline_token_limit`` is set so that the context_growth
    individual outputs stay inline (below the threshold) while the
    large log output exceeds it. The value is chosen relative to the
    ``CharTokenEstimator`` (``len // 4``).
    """
    from harness.output_externalization import OutputExternalizationPolicy
    return OutputExternalizationPolicy(
        inline_token_limit=50,
        preview_chars=80,
    )
