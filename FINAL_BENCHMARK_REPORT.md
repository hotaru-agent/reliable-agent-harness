# Final Benchmark Report

## 1. Executive Summary

The benchmark compares a deliberately naive execution baseline with
the Reliable Harness across deterministic failure-oriented
repository-maintenance workloads. The benchmark evaluates retry,
timeout cleanup, checkpoint/resume, loop governance, context
budgeting, and large-output externalization through controlled
mechanism scenarios, and additionally includes a real-filesystem
integrated multi-fault stress scenario covering retry, timeout, and
checkpoint/resume together.

Results by suite:

| Suite | Naive | Reliable | Advantage |
|---|---|---|---|
| controlled_v1 (8 scenarios) | 1 / 8 | 7 / 8 | 6 / 8 |
| filesystem_pytest_v1 (4 scenarios) | 1 / 4 | 4 / 4 | 3 / 4 |
| integrated_filesystem_v1 (1 scenario) | 0 / 1 | 1 / 1 | 1 / 1 |

The suites are NOT the same sampling population and must NOT be
combined into a single success rate. The integrated scenario is a
single stress test, not a statistical success rate.

**This does not establish universal real-world LLM reliability.**
The benchmark is deterministic mechanism verification with scripted
action sources, not a large-scale empirical LLM study.

## 2. Evaluation Goal

Verify that the Reliable Harness's existing production mechanisms
(retry, timeout, checkpoint/resume, loop detection, context
budgeting, output externalization) can recover from deterministic
failure scenarios where a deliberately naive baseline terminates.

## 3. What Is Being Compared

- **Naive baseline**: Sequential deterministic action execution,
  `max_attempts=1`, no Harness checkpoint/resume, no loop
  governance, append-all context in context ablation, inline-all
  output in externalization ablation.
- **Reliable Harness**: Real `HarnessRuntime` with `ToolRuntime`
  `RetryPolicy`, `AgentActionController`, `LoopDetector`,
  `ContextAssembler`, `ToolOutputProcessor`, `ArtifactStore`, and
  real filesystem adapter where the scenario requires it.

## 4. Benchmark Architecture

The `evaluation/` package is outside the production runtime. It
provides scenarios, runners, fault injection, and an objective
completion oracle. See `ARCHITECTURE.md` for the full runtime
architecture.

## 5. Benchmark Suites

Three frozen suites, explicitly disjoint:

- **controlled_v1** — 8 in-memory mechanism-verification scenarios.
- **filesystem_pytest_v1** — 4 real-filesystem + real-pytest scenarios.
- **integrated_filesystem_v1** — 1 integrated multi-fault stress
  scenario.

Total: 13 deterministic benchmark scenarios. Per-suite aggregates
are reported separately.

## 6. Scenario Matrix

| Suite | Scenario | Naive | Reliable | Mechanism | Real FS | Real Subprocess | Resume |
|---|---|---|---|---|---|---|---|
| controlled_v1 | clean_success | PASS | PASS | No faults; baseline | No | No | No |
| controlled_v1 | transient_failure | FAIL | PASS | Transient retry | No | No | No |
| controlled_v1 | timeout | FAIL | PASS | Tool timeout retry | No | No | No |
| controlled_v1 | permanent_failure | FAIL | FAIL | Permanent; no retry | No | No | No |
| controlled_v1 | checkpoint_recovery | FAIL | PASS | Checkpoint/resume | No | No | Yes |
| controlled_v1 | loop_replan | FAIL | PASS | Loop detection; replan | No | No | No |
| controlled_v1 | context_growth | FAIL | PASS | Context budget | No | No | No |
| controlled_v1 | large_output_externalization | FAIL | PASS | Output externalization | No | No | No |
| filesystem_pytest_v1 | filesystem_pytest_clean | PASS | PASS | Real FS + pytest | Yes | Yes | No |
| filesystem_pytest_v1 | filesystem_pytest_transient | FAIL | PASS | Transient retry on real pytest | Yes | Yes | No |
| filesystem_pytest_v1 | filesystem_pytest_timeout | FAIL | PASS | Real subprocess timeout + retry | Yes | Yes | No |
| filesystem_pytest_v1 | filesystem_pytest_recovery | FAIL | PASS | Filesystem checkpoint/resume | Yes | Yes | Yes |
| integrated_filesystem_v1 | filesystem_integrated_multifault | FAIL | PASS | Transient + timeout + resume | Yes | Yes | Yes |

## 7. Metric Definitions

See `evaluation/report_data.py` (`METRIC_DICTIONARY`) for the
machine-readable metric glossary. Key definitions:

- **task_completed**: Oracle-declared completion (real pytest
  returncode == 0 for filesystem scenarios).
- **logical_action_count**: Actual logical action executions. Retry
  attempts do NOT increment this. Legal recovery replay DOES count.
- **tool_invocation_count**: Logical `ToolRuntime.execute()`
  invocations. Internal retries share the same invocation.
- **tool_attempt_count**: Total handler attempts (initial + retries).
  Always >= tool_invocation_count.
- **retry_count**: Attempts beyond the first per logical invocation.
  Resume replay is NOT a retry.
- **checkpoint_count**: Real Harness checkpoints (successful steps +
  COMPLETE). Failed attempts and interrupted uncheckpointed steps
  do NOT produce checkpoints.
- **resume_count**: Times `HarnessRuntime.resume()` was called and a
  new Run was created.
- **wall_clock_seconds**: Real duration. NOT part of deterministic
  functional equality.

## 8. Controlled Benchmark Results

**controlled_v1**: Naive 1 / 8, Reliable 7 / 8, advantage 6 / 8.

Each scenario is a mechanism verification, not a random sample. The
one Naive success (clean_success) is the no-fault baseline. The one
Reliable failure (permanent_failure) is by design — permanent
failures are not retryable.

## 9. Filesystem Benchmark Results

**filesystem_pytest_v1**: Naive 1 / 4, Reliable 4 / 4, advantage 3 / 4.

These scenarios use a real temporary filesystem with real Python
files and a real `sys.executable -m pytest` subprocess. The oracle
is an independent real pytest subprocess.

## 10. Integrated Multi-Fault Result

**integrated_filesystem_v1**: Naive 0 / 1, Reliable 1 / 1.

A single stress scenario where one Reliable trial sequentially
encounters transient failure, real subprocess timeout, and process
interruption. This is a single stress test, NOT a statistical
success rate.

## 11. Checkpoint / Recovery Findings

- The Reliable Harness resumes from real checkpoints stored in
  `InMemoryCheckpointStore`.
- A NEW `HarnessRuntime` instance is created on resume (same
  CheckpointStore, same filesystem workspace, same FaultInjector).
- Checkpointed steps are NOT re-executed.
- The interrupted uncheckpointed step is legally replayed.
- The source Run remains INTERRUPTED forever; the resume Run is a
  new Run with a new ID.

## 12. Retry / Timeout Findings

- Retry comes ONLY from the existing `RetryPolicy` — no
  benchmark-specific retry loop.
- Timeout is owned by the production `ToolRuntime` — no competing
  repository-level timeout.
- Real subprocess timeout: a real pytest child is spawned, the
  ToolRuntime timeout fires, the child is terminated/killed/reaped,
  and the retry spawns a fresh subprocess.
- `active_process_count == 0` at trial end in all scenarios.

## 13. Loop / Replan Findings

- Loop detection fires on duplicate calls, repeating sequences, and
  no-progress patterns.
- `REPLAN_REQUIRED` blocks subsequent tool calls before ToolRuntime.
- Blocked calls consume no action index.
- `acknowledge_replan` resets the detector episode.

## 14. Context Budget Findings

- The Reliable Harness selects context items within a token budget
  using priority and recency.
- `must_keep` items cannot be silently dropped.
- Naive append-all context exceeds the budget in the context-growth
  scenario.

## 15. Large Output Externalization Findings

- Large tool outputs are externalized to `ArtifactStore`.
- A compact reference is returned instead of the full output.
- The reference token cost is based on the rendered reference, not
  the full artifact.

## 16. Fairness

- Same scenario definition, repository fixture, action sequence
  (where scripted), fault placement, and fresh state per trial.
- Same benchmark capacity/estimator where context is compared.
- Naive and Reliable use the same raw tool outputs where compared.

## 17. Determinism

Functional results are deterministic across repeated runs:
completion, failure reason, counts, fault placement, final
repository snapshot, oracle result.

NOT compared (nondeterministic): UUIDs, timestamps, wall clock,
temp paths, PIDs, raw pytest duration text.

## 18. Interpretation

The benchmark is primarily **deterministic mechanism verification**
and **integration validation**, not a large-scale empirical LLM
study. The Reliable Harness completes scenarios in which the
controlled Naive baseline terminates on retryable tool failure,
timeout, interruption, loop gate, or context-capacity pressure.

This does NOT support claims like "Reliable Harness makes agents 6x
more reliable in general."

## 19. Limitations

- Deterministic scripted action source, not real LLM.
- Synthetic repository workloads.
- No external GitHub repository.
- No SWE-bench.
- No statistical repeated LLM trials.
- `CheckpointStore` is in-memory.
- `ArtifactStore` is in-memory.
- Filesystem recovery remains in the same Python process.
- No durable OS-crash recovery.
- No workspace rollback.
- No exactly-once external side-effect guarantee.
- No context-state persistence across resume.
- No LoopDetector episode persistence across resume.
- No real package installation workload.
- No network-dependent tools in benchmark.
- Integrated scenario does not combine every mechanism.
- Real pytest subprocess timing is nondeterministic.

## 20. Reproduction

See [REPRODUCING.md](REPRODUCING.md) for environment, dependencies,
and commands. Quick start:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## 21. Final Test Baseline

```
1521 passed
0 DeprecationWarning
0 ResourceWarning
```
