"""Context growth & large tool output externalization benchmark
workload (Phase 7 Step 5).

This module provides benchmark-only, deterministic, non-AI components
for the ``context_growth`` and ``large_output_externalization``
scenarios:

- ``CharTokenEstimator`` — a deterministic ``TokenEstimator`` using a
  simple character-count heuristic (``len(text) // 4``, min 1 for
  non-empty). NOT a model tokenizer.

- ``ContextTrialState`` — lightweight benchmark diagnostics for
  deterministic test verification (candidate/selected/omitted item
  counts, peak context tokens, externalized output count, artifact
  IDs).

- ``ContextBenchmarkStepExecutor`` — adapts a ``ScriptedActionSource``
  to the ``StepExecutor`` protocol using the real
  ``ContextAssembler`` + ``ToolOutputProcessor`` + ``ArtifactStore``.
  Before each tool call, it assembles the context within budget. After
  each successful tool call, it processes the output through
  ``ToolOutputProcessor`` and accumulates the resulting
  ``ContextItem``. If the assembled context would exceed the budget,
  the trial terminates with ``CONTEXT_LIMIT``.

- ``NaiveContextExecutor`` — the append-all / inline-all ablation
  baseline. It accumulates every tool output in full as a
  ``ContextItem``, never externalizes, never omits. Before each tool
  call, it checks whether the total accumulated context exceeds the
  budget. If so, it terminates with ``CONTEXT_LIMIT``.

- ``ContextWorkload`` — a ``Workload`` that creates fresh instances of
  the above for each trial. Configurable with a shared
  ``max_context_tokens``, ``TokenEstimator``, ``ToolOutputRenderer``,
  and ``OutputExternalizationPolicy`` so Naive and Reliable use
  identical context-capacity parameters.

All offline, deterministic, no LLM, no network, no real filesystem.
Production ``ContextAssembler``, ``ToolOutputProcessor``,
``ArtifactStore``, ``HarnessRuntime``, ``ToolRuntime`` are NOT
modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from evaluation.records import BenchmarkEvent, BenchmarkEventType
from evaluation.repository import ControlledRepository
from harness.context import (
    ContextAssembler,
    ContextBudget,
    ContextBudgetExceededError,
    ContextItem,
    ContextKind,
    ContextPriority,
    ContextSelectionResult,
)
from harness.execution import ExecutionState, StepOutcome, StepResult
from harness.output_externalization import (
    ArtifactReference,
    DeterministicToolOutputRenderer,
    OutputExternalizationPolicy,
    TokenEstimator,
    ToolOutputProcessingResult,
    ToolOutputProcessor,
    ToolOutputRenderer,
)
from harness.state import Run, Task
from storage.artifact_store import ArtifactStore, InMemoryArtifactStore
from tools import ToolCall, ToolExecutionContext


# Lazy imports to avoid circular import:
# runners → workloads → context_workload → runners
# These are resolved at first use, not at module load.
_BenchmarkStepFailure = None
_MaxActionsExceededError = None
_SCRIPT_CURSOR_KEY = None


def _ensure_runner_symbols():
    global _BenchmarkStepFailure, _MaxActionsExceededError, _SCRIPT_CURSOR_KEY
    if _BenchmarkStepFailure is None:
        from evaluation.runners import (
            BenchmarkStepFailure,
            MaxActionsExceededError,
            SCRIPT_CURSOR_KEY,
        )
        _BenchmarkStepFailure = BenchmarkStepFailure
        _MaxActionsExceededError = MaxActionsExceededError
        _SCRIPT_CURSOR_KEY = SCRIPT_CURSOR_KEY


# ===========================================================================
# Deterministic token estimator
# ===========================================================================


class CharTokenEstimator:
    """Deterministic token estimator using ``len(text) // 4``.

    NOT a model tokenizer. Returns 0 for empty text, ``max(1, len//4)``
    for non-empty text. This is a deterministic cost estimate for
    context budgeting, consistent with the ``TokenEstimator`` protocol.
    """

    def __init__(self, divisor: int = 4) -> None:
        if divisor <= 0:
            raise ValueError("divisor must be > 0")
        self._divisor = divisor

    def estimate(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // self._divisor)


# ===========================================================================
# Context trial state (lightweight benchmark diagnostics)
# ===========================================================================


@dataclass
class ContextTrialState:
    """Lightweight benchmark diagnostics for context scenarios.

    This is NOT a second result system — ``BenchmarkRunRecord`` remains
    the formal evaluation record. This exists for deterministic test
    verification of internal context/externalization state.
    """

    # Candidate items presented to the assembler at the last assembly.
    candidate_item_count: int = 0
    candidate_tokens: int = 0
    # Selected items after assembly.
    selected_item_count: int = 0
    selected_tokens: int = 0
    # Omitted items (candidate - selected).
    omitted_item_count: int = 0
    # Peak context tokens observed at any decision boundary.
    peak_context_tokens: int = 0
    # Final accumulated context history count (all items ever added,
    # including outputs from the last action that has no subsequent
    # assembly). This is distinct from candidate_item_count, which is
    # the candidate set at the last decision boundary.
    accumulated_context_item_count: int = 0
    # Number of tool outputs externalized to the artifact store.
    externalized_output_count: int = 0
    # Total number of successfully processed tool outputs.
    processed_output_count: int = 0
    # Artifact IDs created (for externalized outputs).
    artifact_ids: list[str] = field(default_factory=list)
    # The ArtifactStore (for readback tests).
    artifact_store: Optional[ArtifactStore] = None
    # All accumulated context items (candidate pool).
    context_items: list[ContextItem] = field(default_factory=list)
    # All assembly results (one per decision boundary).
    assembly_results: list[ContextSelectionResult] = field(default_factory=list)
    # Whether the trial terminated due to context limit.
    context_limit_exceeded: bool = False


# ===========================================================================
# Must-keep critical context items
# ===========================================================================


def make_critical_context_items(
    task_goal: str,
    *,
    sequence_index_start: int = 0,
) -> list[ContextItem]:
    """Return the must-keep critical context items for a context scenario.

    These represent the task goal and repository-maintenance constraint.
    They are ``CRITICAL_STATE`` with ``must_keep=True`` so the
    ``ContextAssembler`` always retains them. Each item gets a distinct
    ``sequence_index`` to satisfy the assembler's uniqueness constraint.
    """
    return [
        ContextItem(
            item_id="critical:task_goal",
            kind=ContextKind.CRITICAL_STATE,
            content=f"Task goal: {task_goal}",
            estimated_tokens=max(1, len(f"Task goal: {task_goal}") // 4),
            sequence_index=sequence_index_start,
            priority=ContextPriority.CRITICAL,
            must_keep=True,
        ),
        ContextItem(
            item_id="critical:constraint",
            kind=ContextKind.CRITICAL_STATE,
            content="Constraint: fix calculator.py so all tests pass.",
            estimated_tokens=max(
                1, len("Constraint: fix calculator.py so all tests pass.") // 4
            ),
            sequence_index=sequence_index_start + 1,
            priority=ContextPriority.CRITICAL,
            must_keep=True,
        ),
    ]


# ===========================================================================
# Reliable context benchmark step executor
# ===========================================================================


class ContextBenchmarkStepExecutor:
    """Adapts ``ScriptedActionSource`` to the ``StepExecutor`` protocol
    using the real ``ContextAssembler`` + ``ToolOutputProcessor`` +
    ``ArtifactStore``.

    Timing:
    1. Before each tool call: assemble context within budget. If the
       assembled context exceeds the budget (should not happen with a
       working assembler), terminate with ``CONTEXT_LIMIT``.
    2. Execute the tool call through ``ToolRuntime``.
    3. After each successful tool call: process the output through
       ``ToolOutputProcessor`` (inline or externalize), accumulate the
       resulting ``ContextItem``.

    The context assembly happens BEFORE the tool call (decision
    boundary). The output processing happens AFTER the tool call.

    Context preparation and output processing do NOT count as logical
    actions or tool invocations.
    """

    def __init__(
        self,
        *,
        source: Any,
        invoker: Any,
        ctx: ToolExecutionContext,
        max_logical_actions: int,
        max_context_tokens: int,
        token_estimator: TokenEstimator,
        output_renderer: ToolOutputRenderer,
        externalization_policy: OutputExternalizationPolicy,
        artifact_store: ArtifactStore,
        artifact_id_factory: Callable[[], str],
        context_item_id_factory: Callable[[], str],
        clock: Callable[[], datetime],
        critical_items: list[ContextItem],
        task_goal: str,
    ) -> None:
        self._source = source
        self._invoker = invoker
        self._ctx = ctx
        self._max_logical_actions = max_logical_actions
        self._budget = ContextBudget(max_tokens=max_context_tokens)
        self._assembler = ContextAssembler()
        self._processor = ToolOutputProcessor(
            artifact_store=artifact_store,
            token_estimator=token_estimator,
            output_renderer=output_renderer,
            externalization_policy=externalization_policy,
            artifact_id_factory=artifact_id_factory,
            context_item_id_factory=context_item_id_factory,
            clock=clock,
        )
        self._critical_items = list(critical_items)
        self._task_goal = task_goal

        # Accumulated context items (candidate pool).
        self._context_items: list[ContextItem] = list(critical_items)
        # Sequence index counter for new context items. Start after the
        # critical items (which use sequence_index 0 and 1).
        self._next_seq: int = max(
            (it.sequence_index for it in critical_items), default=-1
        ) + 1

        # Accumulated stats.
        self.logical_action_count: int = 0
        self.tool_invocation_count: int = 0
        self.tool_attempt_count: int = 0
        self.retry_count: int = 0
        self.successful_step_count: int = 0
        self.loop_detection_count: int = 0
        self.replan_count: int = 0
        self.blocked_call_count: int = 0
        self.events: list[BenchmarkEvent] = []
        self._seq: int = 0
        self.execution_history: list[tuple[str, int, str, str]] = []

        # Context trial state for test verification.
        self.trial_state = ContextTrialState(artifact_store=artifact_store)

    async def _assemble_context(self) -> ContextSelectionResult:
        """Assemble the current context items within budget."""
        result = self._assembler.assemble(self._context_items, self._budget)
        self.trial_state.candidate_item_count = len(self._context_items)
        self.trial_state.candidate_tokens = sum(
            it.estimated_tokens for it in self._context_items
        )
        self.trial_state.selected_item_count = len(result.included_items)
        self.trial_state.selected_tokens = result.used_tokens
        self.trial_state.omitted_item_count = len(result.omitted_items)
        self.trial_state.assembly_results.append(result)
        if result.used_tokens > self.trial_state.peak_context_tokens:
            self.trial_state.peak_context_tokens = result.used_tokens
        return result

    async def _process_output(
        self,
        tool_name: str,
        output: Any,
    ) -> ToolOutputProcessingResult:
        """Process a successful tool output through ToolOutputProcessor."""
        result = await self._processor.process_success_output(
            tool_name=tool_name,
            output=output,
            sequence_index=self._next_seq,
            priority=ContextPriority.NORMAL,
        )
        self._next_seq += 1
        self._context_items.append(result.context_item)
        self.trial_state.processed_output_count += 1
        if result.externalized:
            self.trial_state.externalized_output_count += 1
            if result.artifact_reference is not None:
                self.trial_state.artifact_ids.append(
                    result.artifact_reference.artifact_id
                )
        return result

    async def execute_step(self, task, run: Run, state: ExecutionState) -> StepResult:
        _ensure_runner_symbols()
        # Verify cursor consistency.
        restored_cursor = state.agent_state.get(_SCRIPT_CURSOR_KEY, 0)
        if restored_cursor != run.current_step:
            raise RuntimeError(
                f"cursor mismatch: restored script_next_index={restored_cursor}, "
                f"run.current_step={run.current_step}"
            )

        # Script exhausted → COMPLETE.
        if not self._source.has_next():
            self.successful_step_count += 1
            return StepResult(StepOutcome.COMPLETE, state)

        # Max actions exceeded.
        if self.logical_action_count >= self._max_logical_actions:
            raise _MaxActionsExceededError(
                f"max_logical_actions ({self._max_logical_actions}) exceeded"
            )

        # --- Decision boundary: assemble context BEFORE tool call ---
        try:
            assembly = await self._assemble_context()
        except ContextBudgetExceededError:
            # Must-keep items alone exceed budget — this is a config
            # error, not a normal context-limit failure.
            raise

        self.events.append(BenchmarkEvent(
            event_type=BenchmarkEventType.CONTEXT_ASSEMBLED,
            sequence=self._seq,
            tool_name=None,
            success=True,
            details={
                "candidate_count": self.trial_state.candidate_item_count,
                "selected_count": self.trial_state.selected_item_count,
                "omitted_count": self.trial_state.omitted_item_count,
                "used_tokens": assembly.used_tokens,
                "max_tokens": assembly.max_tokens,
            },
        ))
        self._seq += 1

        # --- Execute the tool call ---
        call = self._source.next_call()
        self.logical_action_count += 1
        self.tool_invocation_count += 1

        result = await self._invoker.execute(call, self._ctx)
        self.tool_attempt_count += result.attempt_count
        self.retry_count += max(0, result.attempt_count - 1)

        self.events.append(BenchmarkEvent(
            event_type=BenchmarkEventType.LOGICAL_ACTION,
            sequence=self._seq,
            tool_name=call.tool_name,
            success=result.success,
            details={"attempt_count": result.attempt_count},
        ))
        self._seq += 1

        if not result.success:
            self.execution_history.append(
                (run.run_id, run.current_step, call.tool_name, "failed")
            )
            raise _BenchmarkStepFailure(result.error.error_type, call.tool_name)

        # --- Process the tool output AFTER tool call ---
        proc_result = await self._process_output(call.tool_name, result.output)

        if proc_result.externalized:
            self.events.append(BenchmarkEvent(
                event_type=BenchmarkEventType.OUTPUT_EXTERNALIZED,
                sequence=self._seq,
                tool_name=call.tool_name,
                success=True,
                details={
                    "artifact_id": proc_result.artifact_reference.artifact_id
                    if proc_result.artifact_reference
                    else None,
                    "original_tokens": (
                        proc_result.artifact_reference.original_estimated_tokens
                        if proc_result.artifact_reference
                        else 0
                    ),
                    "reference_tokens": proc_result.context_item.estimated_tokens,
                },
            ))
            self._seq += 1

        # Update trial state context items snapshot and accumulated count.
        self.trial_state.context_items = list(self._context_items)
        self.trial_state.accumulated_context_item_count = len(
            self._context_items
        )

        # Success — grow completed_actions, store cursor, return CONTINUE.
        new_agent_state = dict(state.agent_state)
        new_agent_state[_SCRIPT_CURSOR_KEY] = self._source.current_index
        new_state = ExecutionState(
            agent_state=new_agent_state,
            critical_context=dict(state.critical_context),
            tool_state=dict(state.tool_state),
            completed_actions=state.completed_actions + (call.tool_name,),
            pending_action=state.pending_action,
        )
        self.successful_step_count += 1
        self.execution_history.append(
            (run.run_id, run.current_step, call.tool_name, "success")
        )
        return StepResult(StepOutcome.CONTINUE, new_state)


# ===========================================================================
# Naive context executor (append-all / inline-all ablation)
# ===========================================================================


class NaiveContextExecutor:
    """Naive append-all / inline-all context executor.

    This is a controlled ablation baseline. It:
    - Accumulates every tool output in full as a ``ContextItem``
      (``RECENT_INTERACTION``, full rendered content).
    - Never externalizes.
    - Never omits.
    - Before each tool call, checks whether the total accumulated
      context (critical + all outputs) exceeds the budget. If so,
      terminates with ``CONTEXT_LIMIT``.

    It uses the same ``TokenEstimator`` and ``ToolOutputRenderer`` as
    the Reliable executor so the raw outputs and token estimates are
    identical. The only difference is the absence of
    ``ContextAssembler`` and ``ToolOutputProcessor``.
    """

    def __init__(
        self,
        *,
        source: Any,
        invoker: Any,
        ctx: ToolExecutionContext,
        max_logical_actions: int,
        max_context_tokens: int,
        token_estimator: TokenEstimator,
        output_renderer: ToolOutputRenderer,
        critical_items: list[ContextItem],
        task_goal: str,
    ) -> None:
        self._source = source
        self._invoker = invoker
        self._ctx = ctx
        self._max_logical_actions = max_logical_actions
        self._max_context_tokens = max_context_tokens
        self._estimator = token_estimator
        self._renderer = output_renderer
        self._critical_items = list(critical_items)
        self._task_goal = task_goal

        # Accumulated context items (append-all).
        self._context_items: list[ContextItem] = list(critical_items)
        self._next_seq: int = max(
            (it.sequence_index for it in critical_items), default=-1
        ) + 1
        self._context_item_id_counter: int = 0

        # Accumulated stats.
        self.logical_action_count: int = 0
        self.tool_invocation_count: int = 0
        self.tool_attempt_count: int = 0
        self.retry_count: int = 0
        self.successful_step_count: int = 0
        self.loop_detection_count: int = 0
        self.replan_count: int = 0
        self.blocked_call_count: int = 0
        self.events: list[BenchmarkEvent] = []
        self._seq: int = 0
        self.execution_history: list[tuple[str, int, str, str]] = []

        # Context trial state for test verification.
        self.trial_state = ContextTrialState()
        self.trial_state.context_items = list(self._context_items)

    def _total_context_tokens(self) -> int:
        """Return the total estimated tokens of all accumulated items."""
        return sum(it.estimated_tokens for it in self._context_items)

    async def execute_step(self, task, run: Run, state: ExecutionState) -> StepResult:
        _ensure_runner_symbols()
        # Script exhausted → COMPLETE.
        if not self._source.has_next():
            self.successful_step_count += 1
            return StepResult(StepOutcome.COMPLETE, state)

        # Max actions exceeded.
        if self.logical_action_count >= self._max_logical_actions:
            raise _MaxActionsExceededError(
                f"max_logical_actions ({self._max_logical_actions}) exceeded"
            )

        # --- Decision boundary: check context capacity BEFORE tool ---
        total_tokens = self._total_context_tokens()
        if total_tokens > self.trial_state.peak_context_tokens:
            self.trial_state.peak_context_tokens = total_tokens

        if total_tokens > self._max_context_tokens:
            # Context limit exceeded — terminate.
            self.trial_state.context_limit_exceeded = True
            self.events.append(BenchmarkEvent(
                event_type=BenchmarkEventType.CONTEXT_LIMIT_EXCEEDED,
                sequence=self._seq,
                tool_name=None,
                success=False,
                details={
                    "total_tokens": total_tokens,
                    "max_tokens": self._max_context_tokens,
                },
            ))
            self._seq += 1
            raise ContextLimitExceededError(
                total_tokens=total_tokens,
                max_tokens=self._max_context_tokens,
            )

        # --- Execute the tool call ---
        call = self._source.next_call()
        self.logical_action_count += 1
        self.tool_invocation_count += 1

        result = await self._invoker.execute(call, self._ctx)
        self.tool_attempt_count += result.attempt_count
        self.retry_count += max(0, result.attempt_count - 1)

        self.events.append(BenchmarkEvent(
            event_type=BenchmarkEventType.LOGICAL_ACTION,
            sequence=self._seq,
            tool_name=call.tool_name,
            success=result.success,
            details={"attempt_count": result.attempt_count},
        ))
        self._seq += 1

        if not result.success:
            self.execution_history.append(
                (run.run_id, run.current_step, call.tool_name, "failed")
            )
            raise _BenchmarkStepFailure(result.error.error_type, call.tool_name)

        # --- Append full output inline (no externalization) ---
        rendered = self._renderer.render(result.output)
        tokens = self._estimator.estimate(rendered)
        if not rendered:
            tokens = 0
        self._context_item_id_counter += 1
        inline_item = ContextItem(
            item_id=f"naive_ctx_{self._context_item_id_counter:04d}",
            kind=ContextKind.RECENT_INTERACTION,
            content=rendered,
            estimated_tokens=tokens,
            sequence_index=self._next_seq,
            priority=ContextPriority.NORMAL,
            must_keep=False,
        )
        self._next_seq += 1
        self._context_items.append(inline_item)
        self.trial_state.processed_output_count += 1
        self.trial_state.context_items = list(self._context_items)
        self.trial_state.accumulated_context_item_count = len(
            self._context_items
        )

        # Success.
        new_agent_state = dict(state.agent_state)
        new_agent_state[_SCRIPT_CURSOR_KEY] = self._source.current_index
        new_state = ExecutionState(
            agent_state=new_agent_state,
            critical_context=dict(state.critical_context),
            tool_state=dict(state.tool_state),
            completed_actions=state.completed_actions + (call.tool_name,),
            pending_action=state.pending_action,
        )
        self.successful_step_count += 1
        self.execution_history.append(
            (run.run_id, run.current_step, call.tool_name, "success")
        )
        return StepResult(StepOutcome.CONTINUE, new_state)


# ===========================================================================
# Context limit error
# ===========================================================================


class ContextLimitExceededError(Exception):
    """Raised when the accumulated context exceeds the budget.

    This is a benchmark control failure, NOT a tool error. The runner
    maps it to the ``CONTEXT_LIMIT`` failure reason.
    """

    def __init__(self, total_tokens: int, max_tokens: int) -> None:
        self.total_tokens = total_tokens
        self.max_tokens = max_tokens
        super().__init__(
            f"context limit exceeded: {total_tokens} > {max_tokens}"
        )


# ===========================================================================
# Context workload
# ===========================================================================


class ContextWorkload:
    """Workload for the ``context_growth`` and
    ``large_output_externalization`` scenarios.

    Uses ``ScriptedActionSource`` + ``ContextBenchmarkStepExecutor``
    (Reliable) or ``NaiveContextExecutor`` (Naive).

    Both Naive and Reliable share the same:
    - ``max_context_tokens``
    - ``TokenEstimator``
    - ``ToolOutputRenderer``
    - ``OutputExternalizationPolicy``
    - ``critical_items``

    The only difference is that Reliable uses ``ContextAssembler`` +
    ``ToolOutputProcessor`` + ``ArtifactStore``, while Naive appends
    all outputs inline and never omits.
    """

    def __init__(
        self,
        *,
        action_factory: Callable[[], list],
        max_context_tokens: int,
        token_estimator: Optional[TokenEstimator] = None,
        output_renderer: Optional[ToolOutputRenderer] = None,
        externalization_policy: Optional[OutputExternalizationPolicy] = None,
        task_goal: str = "Fix the buggy add() function in calculator.py so all tests pass.",
    ) -> None:
        self._action_factory = action_factory
        self._max_context_tokens = max_context_tokens
        self._token_estimator = token_estimator or CharTokenEstimator()
        self._output_renderer = output_renderer or DeterministicToolOutputRenderer()
        self._externalization_policy = externalization_policy or OutputExternalizationPolicy(
            inline_token_limit=50,
            preview_chars=80,
        )
        self._task_goal = task_goal

    def create_source(self) -> Any:
        from evaluation.scenarios.controlled_repo import ScriptedActionSource
        return ScriptedActionSource(self._action_factory())

    def _make_critical_items(self) -> list[ContextItem]:
        return make_critical_context_items(self._task_goal)

    def create_step_executor(
        self,
        *,
        source: Any,
        tool_stack: Any,
        repo: ControlledRepository,
        max_logical_actions: int,
    ) -> ContextBenchmarkStepExecutor:
        from datetime import datetime, timezone

        artifact_store = InMemoryArtifactStore()

        def _artifact_id_factory() -> str:
            _artifact_id_factory.counter += 1
            return f"A-{_artifact_id_factory.counter:03d}"
        _artifact_id_factory.counter = 0

        def _context_item_id_factory() -> str:
            _context_item_id_factory.counter += 1
            return f"CTX-{_context_item_id_factory.counter:03d}"
        _context_item_id_factory.counter = 0

        def _clock() -> datetime:
            _clock.t = getattr(_clock, "t", datetime(2026, 1, 1, tzinfo=timezone.utc))
            from datetime import timedelta
            result = _clock.t
            _clock.t = _clock.t + timedelta(seconds=1)
            return result

        return ContextBenchmarkStepExecutor(
            source=source,
            invoker=tool_stack.invoker,
            ctx=tool_stack.ctx,
            max_logical_actions=max_logical_actions,
            max_context_tokens=self._max_context_tokens,
            token_estimator=self._token_estimator,
            output_renderer=self._output_renderer,
            externalization_policy=self._externalization_policy,
            artifact_store=artifact_store,
            artifact_id_factory=_artifact_id_factory,
            context_item_id_factory=_context_item_id_factory,
            clock=_clock,
            critical_items=self._make_critical_items(),
            task_goal=self._task_goal,
        )

    def create_naive_executor(
        self,
        *,
        source: Any,
        invoker: Any,
        ctx: ToolExecutionContext,
        max_logical_actions: int,
    ) -> NaiveContextExecutor:
        return NaiveContextExecutor(
            source=source,
            invoker=invoker,
            ctx=ctx,
            max_logical_actions=max_logical_actions,
            max_context_tokens=self._max_context_tokens,
            token_estimator=self._token_estimator,
            output_renderer=self._output_renderer,
            critical_items=self._make_critical_items(),
            task_goal=self._task_goal,
        )
