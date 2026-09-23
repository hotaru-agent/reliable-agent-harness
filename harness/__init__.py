"""Harness runtime foundation (Phase 1 + Phase 3 + Phase 4).

Exposes the core state models, errors, execution-boundary types, the
async run loop, the deterministic loop detector, the agent action
controller with replan gate, the deterministic context assembler, and
the tool output externalization processor.
"""

from harness.action_controller import (
    ActionControllerState,
    AgentActionController,
    AgentActionResult,
    InvalidActionControllerStateError,
    ProgressProvider,
    ReplanRequiredError,
    ReplanSignal,
)
from harness.context import (
    ContextAssemblyError,
    ContextAssembler,
    ContextBudget,
    ContextBudgetExceededError,
    ContextItem,
    ContextKind,
    ContextPriority,
    ContextSelectionResult,
)
from harness.errors import (
    HarnessRuntimeError,
    InvalidRuntimeStateError,
    InvalidStateTransitionError,
    ResumeError,
)
from harness.execution import (
    ExecutionState,
    StepExecutor,
    StepOutcome,
    StepResult,
)
from harness.loop_detection import (
    ActionFingerprint,
    ActionHistory,
    ActionNormalizationError,
    ActionRecord,
    LoopDetectionReason,
    LoopDetectionResult,
    LoopDetector,
    LoopDetectorConfig,
    canonicalize_arguments,
)
from harness.output_externalization import (
    ArtifactReference,
    DeterministicToolOutputRenderer,
    OutputExternalizationPolicy,
    TokenEstimator,
    ToolOutputProcessingResult,
    ToolOutputProcessor,
    ToolOutputRenderer,
    ToolOutputSerializationError,
)
from harness.runtime import HarnessRuntime
from harness.state import (
    Checkpoint,
    Run,
    RunStatus,
    Task,
    TaskStatus,
    transition_run_status,
    transition_task_status,
)

__all__ = [
    "ActionControllerState",
    "ActionFingerprint",
    "ActionHistory",
    "ActionNormalizationError",
    "ActionRecord",
    "AgentActionController",
    "AgentActionResult",
    "ArtifactReference",
    "Checkpoint",
    "ContextAssemblyError",
    "ContextAssembler",
    "ContextBudget",
    "ContextBudgetExceededError",
    "ContextItem",
    "ContextKind",
    "ContextPriority",
    "ContextSelectionResult",
    "DeterministicToolOutputRenderer",
    "ExecutionState",
    "HarnessRuntime",
    "HarnessRuntimeError",
    "InvalidActionControllerStateError",
    "InvalidRuntimeStateError",
    "InvalidStateTransitionError",
    "LoopDetectionReason",
    "LoopDetectionResult",
    "LoopDetector",
    "LoopDetectorConfig",
    "OutputExternalizationPolicy",
    "ProgressProvider",
    "ReplanRequiredError",
    "ReplanSignal",
    "ResumeError",
    "Run",
    "RunStatus",
    "StepExecutor",
    "StepOutcome",
    "StepResult",
    "Task",
    "TaskStatus",
    "TokenEstimator",
    "ToolOutputProcessingResult",
    "ToolOutputProcessor",
    "ToolOutputRenderer",
    "ToolOutputSerializationError",
    "canonicalize_arguments",
    "transition_run_status",
    "transition_task_status",
]
