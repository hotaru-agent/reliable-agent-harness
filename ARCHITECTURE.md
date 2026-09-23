# Architecture

## 1. Project Goal

**Reliable Agent Harness for Long-Horizon Tool-Using Agents.**

A business-agnostic reliability harness that makes execution state
explicit, governs retries and side effects, detects loops, bounds
context, externalizes large outputs, supports tool interoperability
through MCP, emits privacy-aware OpenTelemetry traces, and evaluates
those mechanisms through deterministic repository-maintenance
benchmarks including real filesystem and real pytest subprocess
execution.

The central principle:

> Agent handles Reasoning / Planning / Choosing Actions.
> Harness / Runtime handles Reliable Execution.

## 2. System Boundary

**In scope:**
- Task / Run / Checkpoint lifecycle
- Tool execution with retry, timeout, permissions, validation
- Side-effect-aware retry safety
- Loop detection and replan gating
- Context budgeting and selection
- Large-output externalization
- MCP tool discovery and adapter
- OpenTelemetry tracing with privacy boundaries
- Deterministic benchmark evaluation

**Out of scope (not implemented):**
- Real LLM action source (benchmarks use deterministic scripted actions)
- Durable checkpoint persistence (InMemoryCheckpointStore)
- Durable artifact persistence (in-memory ArtifactStore)
- Workspace snapshot / rollback
- Exactly-once external side-effect semantics
- Real OS process crash recovery
- Git integration
- Real external repository workloads
- SWE-bench
- Statistical repeated LLM trials

## 3. Runtime Architecture

```
Task
  └── HarnessRuntime
        ├── Run (source)
        │     └── StepExecutor (BenchmarkStepExecutor)
        │           └── ScriptedActionSource → AgentActionController
        │                 └── ToolRuntime
        │                       ├── ToolRegistry → Tool Handler
        │                       ├── RetryPolicy
        │                       ├── Timeout (asyncio.wait_for)
        │                       └── Permission / Validation
        ├── CheckpointStore (InMemoryCheckpointStore)
        └── Resume → Run (resume, new Run ID)
              └── ... (same StepExecutor/ToolRuntime stack)
```

## 4. Task / Run / Checkpoint Model

- **Task**: A unit of work with a goal. Transitions: PENDING →
  RUNNING → COMPLETED / FAILED / WAITING (on interruption).
- **Run**: One execution attempt of a Task. Transitions: PENDING →
  RUNNING → COMPLETED / FAILED / INTERRUPTED. A terminal Run cannot
  restart.
- **Checkpoint**: Snapshot of ExecutionState after a successful
  Harness step. Contains `step_index`, `agent_state` (including
  `benchmark.script_next_index`), and `run_id`.
- **Resume**: Creates a NEW Run with `resumed_from_checkpoint_id`.
  The old Run remains in its terminal state (INTERRUPTED). The
  checkpoint restores the execution cursor; checkpointed steps are
  NOT re-executed.

Key invariants:
- `run.current_step` means the NEXT step to execute.
- Checkpointed steps are not replayed.
- Uncheckpointed interrupted steps MAY be legally replayed.
- Checkpoint recovery is NOT exactly-once external side-effect
  semantics.

## 5. Tool Runtime

- **ToolSpec**: Schema, permissions, timeout, side-effect
  classification, retry policy.
- **ToolRegistry**: Validates tool calls against specs.
- **ToolRuntime.execute()**: One logical invocation = one
  `ToolCall`. Wraps handler call in `asyncio.wait_for` for timeout.
  Applies RetryPolicy on retryable errors.
- **Tool Handler**: The actual tool implementation (filesystem,
  controlled repository, MCP adapter).

## 6. Retry Safety

Retry is governed by `ToolSideEffect` classification:
- `READ_ONLY`: Automatic retry on TRANSIENT / TIMEOUT.
- `IDEMPOTENT`: Automatic retry on TRANSIENT / TIMEOUT.
- `SIDE_EFFECTING`: NO automatic retry. Timeout on a
  side-effecting tool has ambiguous external outcome.

RetryPolicy: `max_attempts`, `initial_backoff_seconds`,
`backoff_multiplier`, `max_backoff_seconds`. Exponential backoff
with cap.

## 7. Loop Governance

- **AgentActionController**: Tracks action fingerprints (tool name +
  canonical arguments). Detects:
  - Duplicate calls (same fingerprint repeated)
  - Repeating sequences (cycle of fingerprints)
  - No-progress (same canonical arguments across different tools)
- **Replan Gate**: On loop detection, emits `REPLAN_REQUIRED`.
  Subsequent tool calls are BLOCKED before ToolRuntime until
  `acknowledge_replan()` is called. Blocked calls consume no action
  index.

## 8. Context Management

- **ContextAssembler**: Selects context items to fit a token budget.
  Selection priority: `priority DESC, recency DESC`. Render order:
  `sequence ASC`.
- **must_keep**: Items marked `must_keep` cannot be silently
  dropped. Overflow raises an explicit error.
- **Omitted != deleted**: Omitted items remain in history but are
  not rendered into the presented context.

## 9. Artifact Externalization

- **ToolOutputProcessor**: Processes tool outputs. Large outputs
  are externalized to ArtifactStore; a compact reference is returned
  instead.
- **ArtifactStore**: Stores full output content. In-memory.
- **Reference**: Token cost is based on the rendered reference, NOT
  the full artifact content.
- **ContextAssembler does not know ArtifactStore**: The reference
  is just another context item.

## 10. MCP Integration

- **MCP Adapter**: Discovers tools from MCP servers and registers
  them in the unified `ToolRegistry`. MCP tools are invoked through
  the same `ToolRuntime` path as native tools.
- Discovery and invocation are traced through OpenTelemetry.

## 11. OpenTelemetry

- Tracing spans for: runtime, tools, action controller, context
  assembly, output processing, MCP discovery.
- Parent-child propagation: action → tool spans.
- **Privacy boundaries**: Sensitive arguments and outputs are
  redacted based on tool spec configuration.

## 12. Evaluation Architecture

The `evaluation/` package is OUTSIDE the production runtime:
- **BenchmarkScenario**: Fixture + fault plan + task.
- **BenchmarkRunner**: Protocol for Naive and Reliable runners.
- **NaiveBenchmarkRunner**: Sequential loop, `max_attempts=1`, no
  Harness, no checkpoints, no loop governance.
- **ReliableHarnessBenchmarkRunner**: Real HarnessRuntime with
  full retry, checkpoint, loop, context, and artifact mechanisms.
- **BenchmarkComparison**: Side-by-side result comparison.
- **FaultInjector**: Injects TRANSIENT, PERMANENT, TIMEOUT,
  PROCESS_INTERRUPTION, LARGE_OUTPUT faults at specific logical
  invocations.

## 13. Filesystem Benchmark Architecture

- **FilesystemRepository**: Real temp filesystem with real Python
  files. Spawns real `sys.executable -m pytest` subprocess via
  `asyncio.create_subprocess_exec`.
- **FilesystemRepositoryOracle**: Independent real pytest subprocess
  that bypasses fault injection. `returncode == 0` is completion
  truth.
- **Cancellation cleanup**: On `asyncio.CancelledError` during
  `proc.communicate()`: terminate → bounded wait → kill → await
  exit → re-raise.
- **Process diagnostics**: `agent_pytest_spawn_count`,
  `agent_pytest_completed_count`, `active_process_count`,
  `process_exit_returncodes`.

## 14. Recovery Semantics

- Source Run is interrupted by `PROCESS_INTERRUPTION` (mapped to
  `asyncio.CancelledError`).
- HarnessRuntime transitions Run → INTERRUPTED, Task → WAITING.
- Resume loads the real checkpoint from CheckpointStore.
- A NEW HarnessRuntime instance (same CheckpointStore) calls
  `HarnessRuntime.resume()`.
- A NEW Run is created with `resumed_from_checkpoint_id`.
- The script cursor is restored from the checkpointed
  `ExecutionState.agent_state["benchmark.script_next_index"]`.
- Checkpointed steps are NOT re-executed.
- The interrupted uncheckpointed step IS legally replayed.
- The same FaultInjector spans source + resume (faults fire once).

## 15. Important Invariants

### Runtime invariants
- Terminal Run cannot restart.
- Resume creates a new Run.
- `run.current_step` means next step to execute.
- Checkpointed steps are not replayed.
- Uncheckpointed interrupted step may replay.
- Checkpoint recovery is not exactly-once external side effects.

### Tool invariants
- One `ToolRuntime.execute` = one logical ToolCall.
- Internal retry attempts share the same logical call.
- READ_ONLY / IDEMPOTENT can auto-retry.
- SIDE_EFFECTING does not automatically retry.
- SIDE_EFFECTING timeout has ambiguous external outcome and is not
  auto-retried.

### Loop invariants
- Loop triggering action has already executed.
- REPLAN_REQUIRED blocks subsequent Tool calls before ToolRuntime.
- Blocked calls consume no action index.
- `acknowledge_replan` resets detector episode but not global
  action index.

### Context invariants
- `must_keep` items cannot be silently dropped.
- `must_keep` overflow raises explicit error.
- Selection priority: priority DESC, recency DESC.
- Render order: sequence ASC.
- Omitted != deleted.

### Artifact invariants
- Full output saved before reference returned.
- ContextAssembler does not know ArtifactStore.
- Reference token cost is based on rendered reference.
- Externalization preserves full artifact content.
- Reference omission does not delete artifact.

### Benchmark invariants
- Retry attempt != logical action.
- Legal recovery replay != erroneous duplicate.
- Context preparation != logical action.
- Output processing != logical action.
- Oracle execution != agent action/tool invocation.
- Benchmark result uses objective completion oracle where
  applicable.

## 16. Non-Goals

- Exactly-once external side-effect semantics.
- Durable recovery after real OS process death.
- Filesystem rollback / transactional workspace state.
- Durable ArtifactStore / CheckpointStore.
- Automatic context persistence across resume.
- Automatic LoopDetector episode persistence across resume.
- Real autonomous LLM planning.
- Real-world SWE benchmark performance.
- Statistical model success rates.
