"""Tests for HarnessRuntime OpenTelemetry tracing (Phase 6 Step 1).

All offline, deterministic, no network, no collector. Uses a recording
span exporter to assert on span names, attributes, events, statuses
and parent-child relationships.

Tests prove:
* successful run produces harness.run + harness.step spans + checkpoint events
* step spans are children of the run span
* failed run records exception, sets ERROR status, re-raises
* cancelled run sets INTERRUPTED, UNSET status, re-raises CancelledError
* resume produces resume=True attribute + resumed event
* no SDK configured -> business behavior unchanged
* tracer injection isolation (separate providers don't cross-contaminate)
* sensitive checkpoint state is not recorded in telemetry
* span names are deterministic / low-cardinality
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from opentelemetry.trace.status import Status, StatusCode

from harness import (
    HarnessRuntime,
    RunStatus,
    Task,
    TaskStatus,
)
from harness.state import Run
from observability import (
    ATTR_CHECKPOINT_ID,
    ATTR_RESUME_FROM_CHECKPOINT_ID,
    ATTR_RUN_ID,
    ATTR_RUN_RESUME,
    ATTR_RUN_STATUS,
    ATTR_STEP_INDEX,
    ATTR_TASK_ID,
    EVENT_CHECKPOINT_SAVED,
    EVENT_RUN_INTERRUPTED,
    EVENT_RUN_RESUMED,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_INTERRUPTED,
    SPAN_HARNESS_RUN,
    SPAN_HARNESS_STEP,
)
from storage import InMemoryCheckpointStore
from tests.fakes import FakeClock, FakeStepExecutor, make_id_factory
from tests.fakes_telemetry import make_recording_tracer

T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _run(coro):
    return asyncio.run(coro)


def _make_task(task_id: str = "T-001") -> Task:
    return Task(task_id=task_id, goal="Diagnose the failing experiment", created_at=T0)


def _make_traced_runtime(
    tracer,
    *,
    store=None,
    executor=None,
    run_prefix: str = "R",
    cp_prefix: str = "C",
):
    store = store or InMemoryCheckpointStore()
    executor = executor or FakeStepExecutor(complete_at_step=2)
    clock = FakeClock(T0)
    runtime = HarnessRuntime(
        checkpoint_store=store,
        step_executor=executor,
        clock=clock,
        run_id_factory=make_id_factory(run_prefix),
        checkpoint_id_factory=make_id_factory(cp_prefix),
        tracer=tracer,
    )
    return runtime, store, executor


# ---------------------------------------------------------------------------
# Section 51 — Successful Harness Run
# ---------------------------------------------------------------------------


class TestSuccessfulRunTracing:
    def test_run_span_exists(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)
        task = _make_task()

        _run(runtime.start(task))

        run_spans = exporter.spans_named(SPAN_HARNESS_RUN)
        assert len(run_spans) == 1

    def test_run_span_has_task_and_run_id(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)
        task = _make_task("T-001")

        run = _run(runtime.start(task))

        run_span = exporter.span_by_name(SPAN_HARNESS_RUN)
        assert run_span.attributes[ATTR_TASK_ID] == "T-001"
        assert run_span.attributes[ATTR_RUN_ID] == run.run_id

    def test_run_span_resume_false_on_fresh_start(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)
        task = _make_task()

        _run(runtime.start(task))

        run_span = exporter.span_by_name(SPAN_HARNESS_RUN)
        assert run_span.attributes[ATTR_RUN_RESUME] is False

    def test_step_spans_count_matches_steps(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)
        task = _make_task()

        _run(runtime.start(task))

        step_spans = exporter.spans_named(SPAN_HARNESS_STEP)
        assert len(step_spans) == 3  # steps 0, 1, 2

    def test_step_spans_have_indices(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)
        task = _make_task()

        _run(runtime.start(task))

        step_spans = exporter.spans_named(SPAN_HARNESS_STEP)
        indices = [s.attributes[ATTR_STEP_INDEX] for s in step_spans]
        assert indices == [0, 1, 2]

    def test_checkpoint_saved_events(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)
        task = _make_task()

        _run(runtime.start(task))

        step_spans = exporter.spans_named(SPAN_HARNESS_STEP)
        for s in step_spans:
            cp_events = [e for e in s.events if e.name == EVENT_CHECKPOINT_SAVED]
            assert len(cp_events) == 1
            assert ATTR_CHECKPOINT_ID in cp_events[0].attributes

    def test_run_status_completed(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)
        task = _make_task()

        _run(runtime.start(task))

        run_span = exporter.span_by_name(SPAN_HARNESS_RUN)
        assert run_span.attributes[ATTR_RUN_STATUS] == RUN_STATUS_COMPLETED

    def test_run_span_status_ok(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)
        task = _make_task()

        _run(runtime.start(task))

        run_span = exporter.span_by_name(SPAN_HARNESS_RUN)
        assert run_span.status.status_code == StatusCode.OK

    def test_step_spans_status_ok(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)
        task = _make_task()

        _run(runtime.start(task))

        step_spans = exporter.spans_named(SPAN_HARNESS_STEP)
        for s in step_spans:
            assert s.status.status_code == StatusCode.OK


# ---------------------------------------------------------------------------
# Section 52 — Harness Span Parenting
# ---------------------------------------------------------------------------


class TestSpanParenting:
    def test_step_spans_are_children_of_run_span(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)
        task = _make_task()

        _run(runtime.start(task))

        run_span = exporter.span_by_name(SPAN_HARNESS_RUN)
        step_spans = exporter.spans_named(SPAN_HARNESS_STEP)
        for s in step_spans:
            assert s.parent is not None
            assert s.parent.span_id == run_span.context.span_id


# ---------------------------------------------------------------------------
# Section 53 — Harness Failure
# ---------------------------------------------------------------------------


class TestFailedRunTracing:
    def test_run_failed_status(self):
        tracer, exporter = make_recording_tracer()
        executor = FakeStepExecutor(complete_at_step=2, fail_at_step=1)
        runtime, _, _ = _make_traced_runtime(tracer, executor=executor)
        task = _make_task()

        with pytest.raises(RuntimeError):
            _run(runtime.start(task))

        run_span = exporter.span_by_name(SPAN_HARNESS_RUN)
        assert run_span.attributes[ATTR_RUN_STATUS] == RUN_STATUS_FAILED

    def test_run_span_error_status(self):
        tracer, exporter = make_recording_tracer()
        executor = FakeStepExecutor(complete_at_step=2, fail_at_step=1)
        runtime, _, _ = _make_traced_runtime(tracer, executor=executor)
        task = _make_task()

        with pytest.raises(RuntimeError):
            _run(runtime.start(task))

        run_span = exporter.span_by_name(SPAN_HARNESS_RUN)
        assert run_span.status.status_code == StatusCode.ERROR

    def test_failed_step_span_error_status(self):
        tracer, exporter = make_recording_tracer()
        executor = FakeStepExecutor(complete_at_step=2, fail_at_step=1)
        runtime, _, _ = _make_traced_runtime(tracer, executor=executor)
        task = _make_task()

        with pytest.raises(RuntimeError):
            _run(runtime.start(task))

        step_spans = exporter.spans_named(SPAN_HARNESS_STEP)
        # Step 0 succeeded, step 1 failed.
        assert step_spans[0].status.status_code == StatusCode.OK
        assert step_spans[1].status.status_code == StatusCode.ERROR

    def test_exception_recorded_on_run_span(self):
        tracer, exporter = make_recording_tracer()
        executor = FakeStepExecutor(
            complete_at_step=2, fail_at_step=1, fail_exc=RuntimeError("boom")
        )
        runtime, _, _ = _make_traced_runtime(tracer, executor=executor)
        task = _make_task()

        with pytest.raises(RuntimeError):
            _run(runtime.start(task))

        run_span = exporter.span_by_name(SPAN_HARNESS_RUN)
        # Default policy records exception type as attribute, not as event.
        from observability import ATTR_ERROR_EXCEPTION_TYPE
        assert run_span.attributes[ATTR_ERROR_EXCEPTION_TYPE] == "RuntimeError"

    def test_exception_recorded_on_step_span(self):
        tracer, exporter = make_recording_tracer()
        executor = FakeStepExecutor(
            complete_at_step=2, fail_at_step=1, fail_exc=RuntimeError("boom")
        )
        runtime, _, _ = _make_traced_runtime(tracer, executor=executor)
        task = _make_task()

        with pytest.raises(RuntimeError):
            _run(runtime.start(task))

        step_spans = exporter.spans_named(SPAN_HARNESS_STEP)
        # Default policy records exception type as attribute, not as event.
        from observability import ATTR_ERROR_EXCEPTION_TYPE
        assert step_spans[1].attributes[ATTR_ERROR_EXCEPTION_TYPE] == "RuntimeError"

    def test_failed_step_no_checkpoint_event(self):
        tracer, exporter = make_recording_tracer()
        executor = FakeStepExecutor(complete_at_step=2, fail_at_step=1)
        runtime, _, _ = _make_traced_runtime(tracer, executor=executor)
        task = _make_task()

        with pytest.raises(RuntimeError):
            _run(runtime.start(task))

        step_spans = exporter.spans_named(SPAN_HARNESS_STEP)
        cp_events = [
            e for e in step_spans[1].events if e.name == EVENT_CHECKPOINT_SAVED
        ]
        assert len(cp_events) == 0

    def test_original_exception_still_propagates(self):
        tracer, exporter = make_recording_tracer()
        executor = FakeStepExecutor(
            complete_at_step=2, fail_at_step=1, fail_exc=ValueError("custom error")
        )
        runtime, _, _ = _make_traced_runtime(tracer, executor=executor)
        task = _make_task()

        with pytest.raises(ValueError, match="custom error"):
            _run(runtime.start(task))


# ---------------------------------------------------------------------------
# Section 54 — Harness Cancellation
# ---------------------------------------------------------------------------


class TestCancelledRunTracing:
    def test_run_interrupted_status(self):
        tracer, exporter = make_recording_tracer()
        executor = FakeStepExecutor(complete_at_step=2, cancel_at_step=1)
        runtime, _, _ = _make_traced_runtime(tracer, executor=executor)
        task = _make_task()

        with pytest.raises(asyncio.CancelledError):
            _run(runtime.start(task))

        run_span = exporter.span_by_name(SPAN_HARNESS_RUN)
        assert run_span.attributes[ATTR_RUN_STATUS] == RUN_STATUS_INTERRUPTED

    def test_run_span_status_unset_on_cancellation(self):
        tracer, exporter = make_recording_tracer()
        executor = FakeStepExecutor(complete_at_step=2, cancel_at_step=1)
        runtime, _, _ = _make_traced_runtime(tracer, executor=executor)
        task = _make_task()

        with pytest.raises(asyncio.CancelledError):
            _run(runtime.start(task))

        run_span = exporter.span_by_name(SPAN_HARNESS_RUN)
        assert run_span.status.status_code == StatusCode.UNSET

    def test_interrupted_event_on_run_span(self):
        tracer, exporter = make_recording_tracer()
        executor = FakeStepExecutor(complete_at_step=2, cancel_at_step=1)
        runtime, _, _ = _make_traced_runtime(tracer, executor=executor)
        task = _make_task()

        with pytest.raises(asyncio.CancelledError):
            _run(runtime.start(task))

        run_span = exporter.span_by_name(SPAN_HARNESS_RUN)
        interrupted_events = [
            e for e in run_span.events if e.name == EVENT_RUN_INTERRUPTED
        ]
        assert len(interrupted_events) == 1

    def test_cancelled_step_no_checkpoint_event(self):
        tracer, exporter = make_recording_tracer()
        executor = FakeStepExecutor(complete_at_step=2, cancel_at_step=1)
        runtime, _, _ = _make_traced_runtime(tracer, executor=executor)
        task = _make_task()

        with pytest.raises(asyncio.CancelledError):
            _run(runtime.start(task))

        step_spans = exporter.spans_named(SPAN_HARNESS_STEP)
        cp_events = [
            e for e in step_spans[1].events if e.name == EVENT_CHECKPOINT_SAVED
        ]
        assert len(cp_events) == 0

    def test_cancelled_error_still_propagates(self):
        tracer, exporter = make_recording_tracer()
        executor = FakeStepExecutor(complete_at_step=2, cancel_at_step=0)
        runtime, _, _ = _make_traced_runtime(tracer, executor=executor)
        task = _make_task()

        with pytest.raises(asyncio.CancelledError):
            _run(runtime.start(task))


# ---------------------------------------------------------------------------
# Section 55 — Resume
# ---------------------------------------------------------------------------


class TestResumeTracing:
    def _setup_interrupted_run(self, tracer):
        """Helper: run to interruption at step 2, return (runtime2, store, task, old_run)."""
        store = InMemoryCheckpointStore()
        executor1 = FakeStepExecutor(complete_at_step=4, cancel_at_step=2)
        clock = FakeClock(T0)
        run_ids = make_id_factory("R")
        cp_ids = make_id_factory("C")
        runtime1 = HarnessRuntime(
            checkpoint_store=store,
            step_executor=executor1,
            clock=clock,
            run_id_factory=run_ids,
            checkpoint_id_factory=cp_ids,
            tracer=tracer,
        )
        task = _make_task()

        with pytest.raises(asyncio.CancelledError) as exc_info:
            _run(runtime1.start(task))
        old_run = exc_info.value.run

        executor2 = FakeStepExecutor(complete_at_step=4)
        runtime2 = HarnessRuntime(
            checkpoint_store=store,
            step_executor=executor2,
            clock=clock,
            run_id_factory=run_ids,
            checkpoint_id_factory=cp_ids,
            tracer=tracer,
        )
        return runtime2, store, task, old_run

    def test_resume_span_has_resume_true(self):
        tracer, exporter = make_recording_tracer()
        runtime2, _, task, old_run = self._setup_interrupted_run(tracer)

        _run(runtime2.resume(task=task, source_run=old_run, checkpoint_id="C-002"))

        run_spans = exporter.spans_named(SPAN_HARNESS_RUN)
        assert len(run_spans) == 2
        resume_span = run_spans[1]
        assert resume_span.attributes[ATTR_RUN_RESUME] is True

    def test_resume_span_has_from_checkpoint_id(self):
        tracer, exporter = make_recording_tracer()
        runtime2, _, task, old_run = self._setup_interrupted_run(tracer)

        _run(runtime2.resume(task=task, source_run=old_run, checkpoint_id="C-002"))

        run_spans = exporter.spans_named(SPAN_HARNESS_RUN)
        resume_span = run_spans[1]
        assert resume_span.attributes[ATTR_RESUME_FROM_CHECKPOINT_ID] == "C-002"

    def test_resume_event_exists(self):
        tracer, exporter = make_recording_tracer()
        runtime2, _, task, old_run = self._setup_interrupted_run(tracer)

        _run(runtime2.resume(task=task, source_run=old_run, checkpoint_id="C-002"))

        run_spans = exporter.spans_named(SPAN_HARNESS_RUN)
        resume_span = run_spans[1]
        resumed_events = [
            e for e in resume_span.events if e.name == EVENT_RUN_RESUMED
        ]
        assert len(resumed_events) == 1
        assert resumed_events[0].attributes[ATTR_RESUME_FROM_CHECKPOINT_ID] == "C-002"
        assert resumed_events[0].attributes[ATTR_STEP_INDEX] == 2  # checkpoint.step_index + 1


# ---------------------------------------------------------------------------
# Section 50 — No SDK Configured
# ---------------------------------------------------------------------------


class TestNoSDKConfigured:
    def test_business_behavior_unchanged_without_tracer(self):
        """Without a tracer, runtime behaves exactly as before."""
        store = InMemoryCheckpointStore()
        executor = FakeStepExecutor(complete_at_step=2)
        runtime = HarnessRuntime(
            checkpoint_store=store,
            step_executor=executor,
            clock=FakeClock(T0),
            run_id_factory=make_id_factory("R"),
            checkpoint_id_factory=make_id_factory("C"),
        )
        task = _make_task()

        run = _run(runtime.start(task))

        assert executor.executed_steps == [0, 1, 2]
        assert run.current_step == 3
        assert run.last_checkpoint_id == "C-003"
        assert run.status == RunStatus.COMPLETED
        assert task.status == TaskStatus.COMPLETED

    def test_failure_behavior_unchanged_without_tracer(self):
        store = InMemoryCheckpointStore()
        executor = FakeStepExecutor(complete_at_step=2, fail_at_step=1)
        runtime = HarnessRuntime(
            checkpoint_store=store,
            step_executor=executor,
            clock=FakeClock(T0),
            run_id_factory=make_id_factory("R"),
            checkpoint_id_factory=make_id_factory("C"),
        )
        task = _make_task()

        with pytest.raises(RuntimeError):
            _run(runtime.start(task))

        assert task.status == TaskStatus.FAILED


# ---------------------------------------------------------------------------
# Section 68 — Tracer Injection Isolation
# ---------------------------------------------------------------------------


class TestTracerInjectionIsolation:
    def test_two_runtimes_do_not_cross_contaminate(self):
        tracer_a, exporter_a = make_recording_tracer()
        tracer_b, exporter_b = make_recording_tracer()

        runtime_a, _, _ = _make_traced_runtime(tracer_a)
        runtime_b, _, _ = _make_traced_runtime(tracer_b)

        _run(runtime_a.start(_make_task("T-A")))
        _run(runtime_b.start(_make_task("T-B")))

        # Exporter A only has T-A.
        run_spans_a = exporter_a.spans_named(SPAN_HARNESS_RUN)
        assert len(run_spans_a) == 1
        assert run_spans_a[0].attributes[ATTR_TASK_ID] == "T-A"

        # Exporter B only has T-B.
        run_spans_b = exporter_b.spans_named(SPAN_HARNESS_RUN)
        assert len(run_spans_b) == 1
        assert run_spans_b[0].attributes[ATTR_TASK_ID] == "T-B"


# ---------------------------------------------------------------------------
# Section 67 — Deterministic Span Names
# ---------------------------------------------------------------------------


class TestDeterministicSpanNames:
    def test_span_names_are_fixed(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)
        task = _make_task("T-XYZ-999")

        _run(runtime.start(task))

        all_names = {s.name for s in exporter.finished_spans}
        assert all_names <= {SPAN_HARNESS_RUN, SPAN_HARNESS_STEP}
        assert SPAN_HARNESS_RUN in all_names
        assert SPAN_HARNESS_STEP in all_names

    def test_no_task_id_in_span_name(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)
        task = _make_task("T-UNIQUE-123")

        _run(runtime.start(task))

        for s in exporter.finished_spans:
            assert "T-UNIQUE-123" not in s.name


# ---------------------------------------------------------------------------
# Section 66 — Checkpoint State Not Recorded
# ---------------------------------------------------------------------------


class TestNoSensitiveCheckpointState:
    def test_checkpoint_state_not_in_telemetry(self):
        tracer, exporter = make_recording_tracer()
        store = InMemoryCheckpointStore()
        executor = FakeStepExecutor(complete_at_step=2)
        runtime, _, _ = _make_traced_runtime(tracer, store=store, executor=executor)
        task = _make_task()

        _run(runtime.start(task))

        # Check all span attributes and event attributes for the marker.
        for s in exporter.finished_spans:
            for key, val in (s.attributes or {}).items():
                assert "SECRET_MARKER" not in str(val)
            for ev in s.events:
                for key, val in (ev.attributes or {}).items():
                    assert "SECRET_MARKER" not in str(val)

    def test_only_checkpoint_id_in_event(self):
        tracer, exporter = make_recording_tracer()
        runtime, _, _ = _make_traced_runtime(tracer)
        task = _make_task()

        _run(runtime.start(task))

        step_spans = exporter.spans_named(SPAN_HARNESS_STEP)
        for s in step_spans:
            cp_events = [e for e in s.events if e.name == EVENT_CHECKPOINT_SAVED]
            for ev in cp_events:
                # Only checkpoint_id and step_index should be in the event.
                assert set(ev.attributes.keys()) <= {ATTR_CHECKPOINT_ID, ATTR_STEP_INDEX}
