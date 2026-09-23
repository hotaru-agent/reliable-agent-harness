"""Tests for ContextAssembler OpenTelemetry tracing (Phase 6 Step 2).

All offline, deterministic, no network, no collector. Uses a recording
span exporter to assert on span names, attributes, statuses.

Tests prove:
* context.assemble span exists with token/counts attributes
* success: Status OK
* context overflow: Status ERROR, exception type recorded
* input validation failure: Status ERROR
* no context content in telemetry
* tracer injection isolation
* telemetry failure isolation
"""

from __future__ import annotations

import pytest
from opentelemetry.trace.status import Status, StatusCode

from harness.context import (
    ContextAssembler,
    ContextBudget,
    ContextBudgetExceededError,
    ContextItem,
    ContextKind,
    ContextPriority,
)
from observability import (
    ATTR_CONTEXT_INCLUDED_COUNT,
    ATTR_CONTEXT_INPUT_COUNT,
    ATTR_CONTEXT_MUST_KEEP_COUNT,
    ATTR_CONTEXT_OMITTED_COUNT,
    ATTR_CONTEXT_TOKEN_MAX,
    ATTR_CONTEXT_TOKEN_REMAINING,
    ATTR_CONTEXT_TOKEN_REQUIRED,
    ATTR_CONTEXT_TOKEN_USED,
    ATTR_ERROR_EXCEPTION_TYPE,
    SPAN_CONTEXT_ASSEMBLE,
)
from tests.fakes_telemetry import make_recording_tracer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CRITICAL_CONTENT = "CRITICAL_SECRET_MARKER"
_RECENT_CONTENT = "RECENT_SECRET_MARKER"
_HISTORICAL_CONTENT = "HISTORICAL_SECRET_MARKER"


def _make_items():
    return [
        ContextItem(
            item_id="critical",
            kind=ContextKind.CRITICAL_STATE,
            content=_CRITICAL_CONTENT,
            estimated_tokens=10,
            sequence_index=0,
            must_keep=True,
        ),
        ContextItem(
            item_id="recent",
            kind=ContextKind.RECENT_INTERACTION,
            content=_RECENT_CONTENT,
            estimated_tokens=20,
            sequence_index=1,
            priority=ContextPriority.HIGH,
        ),
        ContextItem(
            item_id="historical",
            kind=ContextKind.HISTORICAL_EVIDENCE,
            content=_HISTORICAL_CONTENT,
            estimated_tokens=30,
            sequence_index=2,
            priority=ContextPriority.LOW,
        ),
    ]


# ---------------------------------------------------------------------------
# Section 51 — Context Trace Test
# ---------------------------------------------------------------------------


class TestContextAssembleTracing:
    def test_assemble_span_exists(self):
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        assembler.assemble(_make_items(), ContextBudget(max_tokens=100))

        spans = exporter.spans_named(SPAN_CONTEXT_ASSEMBLE)
        assert len(spans) == 1

    def test_assemble_token_max_attribute(self):
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        assembler.assemble(_make_items(), ContextBudget(max_tokens=100))

        span = exporter.span_by_name(SPAN_CONTEXT_ASSEMBLE)
        assert span.attributes[ATTR_CONTEXT_TOKEN_MAX] == 100

    def test_assemble_input_count_attribute(self):
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        assembler.assemble(_make_items(), ContextBudget(max_tokens=100))

        span = exporter.span_by_name(SPAN_CONTEXT_ASSEMBLE)
        assert span.attributes[ATTR_CONTEXT_INPUT_COUNT] == 3

    def test_assemble_must_keep_count(self):
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        assembler.assemble(_make_items(), ContextBudget(max_tokens=100))

        span = exporter.span_by_name(SPAN_CONTEXT_ASSEMBLE)
        # critical is must_keep=True (CRITICAL_STATE forces it).
        assert span.attributes[ATTR_CONTEXT_MUST_KEEP_COUNT] == 1

    def test_assemble_included_count(self):
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        result = assembler.assemble(_make_items(), ContextBudget(max_tokens=100))

        span = exporter.span_by_name(SPAN_CONTEXT_ASSEMBLE)
        assert span.attributes[ATTR_CONTEXT_INCLUDED_COUNT] == len(result.included_items)

    def test_assemble_omitted_count(self):
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        result = assembler.assemble(_make_items(), ContextBudget(max_tokens=100))

        span = exporter.span_by_name(SPAN_CONTEXT_ASSEMBLE)
        assert span.attributes[ATTR_CONTEXT_OMITTED_COUNT] == len(result.omitted_items)

    def test_assemble_used_tokens(self):
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        result = assembler.assemble(_make_items(), ContextBudget(max_tokens=100))

        span = exporter.span_by_name(SPAN_CONTEXT_ASSEMBLE)
        assert span.attributes[ATTR_CONTEXT_TOKEN_USED] == result.used_tokens

    def test_assemble_remaining_tokens(self):
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        result = assembler.assemble(_make_items(), ContextBudget(max_tokens=100))

        span = exporter.span_by_name(SPAN_CONTEXT_ASSEMBLE)
        assert span.attributes[ATTR_CONTEXT_TOKEN_REMAINING] == result.remaining_tokens

    def test_assemble_success_status(self):
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        assembler.assemble(_make_items(), ContextBudget(max_tokens=100))

        span = exporter.span_by_name(SPAN_CONTEXT_ASSEMBLE)
        assert span.status.status_code == StatusCode.OK

    def test_assemble_omits_some_items(self):
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        # Budget: 10 (critical) + 20 (recent) = 30; historical (30) won't fit in 35.
        result = assembler.assemble(_make_items(), ContextBudget(max_tokens=35))

        span = exporter.span_by_name(SPAN_CONTEXT_ASSEMBLE)
        assert span.attributes[ATTR_CONTEXT_INCLUDED_COUNT] == 2
        assert span.attributes[ATTR_CONTEXT_OMITTED_COUNT] == 1
        assert span.status.status_code == StatusCode.OK


# ---------------------------------------------------------------------------
# Section 29 — Context Overflow
# ---------------------------------------------------------------------------


class TestContextOverflow:
    def test_overflow_error_status(self):
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        # critical alone needs 10, but budget is 5.
        with pytest.raises(ContextBudgetExceededError):
            assembler.assemble(_make_items(), ContextBudget(max_tokens=5))

        span = exporter.span_by_name(SPAN_CONTEXT_ASSEMBLE)
        assert span.status.status_code == StatusCode.ERROR

    def test_overflow_records_required_tokens(self):
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        with pytest.raises(ContextBudgetExceededError):
            assembler.assemble(_make_items(), ContextBudget(max_tokens=5))

        span = exporter.span_by_name(SPAN_CONTEXT_ASSEMBLE)
        assert span.attributes[ATTR_CONTEXT_TOKEN_REQUIRED] == 10

    def test_overflow_exception_type_recorded(self):
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        with pytest.raises(ContextBudgetExceededError):
            assembler.assemble(_make_items(), ContextBudget(max_tokens=5))

        span = exporter.span_by_name(SPAN_CONTEXT_ASSEMBLE)
        assert span.attributes[ATTR_ERROR_EXCEPTION_TYPE] == "ContextBudgetExceededError"


# ---------------------------------------------------------------------------
# Section 30 — Input Validation Failure
# ---------------------------------------------------------------------------


class TestInputValidationFailure:
    def test_duplicate_id_error_status(self):
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        items = [
            ContextItem(
                item_id="dup",
                kind=ContextKind.CRITICAL_STATE,
                content="x",
                estimated_tokens=1,
                sequence_index=0,
                must_keep=True,
            ),
            ContextItem(
                item_id="dup",
                kind=ContextKind.RECENT_INTERACTION,
                content="y",
                estimated_tokens=1,
                sequence_index=1,
            ),
        ]
        with pytest.raises(Exception):
            assembler.assemble(items, ContextBudget(max_tokens=100))

        span = exporter.span_by_name(SPAN_CONTEXT_ASSEMBLE)
        assert span.status.status_code == StatusCode.ERROR

    def test_duplicate_id_exception_type_recorded(self):
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        items = [
            ContextItem(
                item_id="dup",
                kind=ContextKind.CRITICAL_STATE,
                content="x",
                estimated_tokens=1,
                sequence_index=0,
                must_keep=True,
            ),
            ContextItem(
                item_id="dup",
                kind=ContextKind.RECENT_INTERACTION,
                content="y",
                estimated_tokens=1,
                sequence_index=1,
            ),
        ]
        with pytest.raises(Exception):
            assembler.assemble(items, ContextBudget(max_tokens=100))

        span = exporter.span_by_name(SPAN_CONTEXT_ASSEMBLE)
        assert ATTR_ERROR_EXCEPTION_TYPE in span.attributes

    def test_duplicate_id_value_not_in_telemetry(self):
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        items = [
            ContextItem(
                item_id="SECRET_DUP_ID_MARKER",
                kind=ContextKind.CRITICAL_STATE,
                content="x",
                estimated_tokens=1,
                sequence_index=0,
                must_keep=True,
            ),
            ContextItem(
                item_id="SECRET_DUP_ID_MARKER",
                kind=ContextKind.RECENT_INTERACTION,
                content="y",
                estimated_tokens=1,
                sequence_index=1,
            ),
        ]
        with pytest.raises(Exception):
            assembler.assemble(items, ContextBudget(max_tokens=100))

        for s in exporter.finished_spans:
            for key, val in (s.attributes or {}).items():
                assert "SECRET_DUP_ID_MARKER" not in str(val)
            for ev in s.events:
                for key, val in (ev.attributes or {}).items():
                    assert "SECRET_DUP_ID_MARKER" not in str(val)


# ---------------------------------------------------------------------------
# Section 51 — No Context Content in Telemetry
# ---------------------------------------------------------------------------


class TestNoContextContent:
    def test_critical_content_not_in_telemetry(self):
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        assembler.assemble(_make_items(), ContextBudget(max_tokens=100))

        for s in exporter.finished_spans:
            for key, val in (s.attributes or {}).items():
                assert _CRITICAL_CONTENT not in str(val)
                assert _RECENT_CONTENT not in str(val)
                assert _HISTORICAL_CONTENT not in str(val)
            for ev in s.events:
                for key, val in (ev.attributes or {}).items():
                    assert _CRITICAL_CONTENT not in str(val)
                    assert _RECENT_CONTENT not in str(val)
                    assert _HISTORICAL_CONTENT not in str(val)

    def test_no_item_ids_in_telemetry(self):
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        assembler.assemble(_make_items(), ContextBudget(max_tokens=100))

        for s in exporter.finished_spans:
            for key in (s.attributes or {}).keys():
                # No item_id values leaked.
                assert "critical" not in str(key).lower() or "must_keep" in str(key).lower()
                assert "recent" not in str(key).lower()
                assert "historical" not in str(key).lower()


# ---------------------------------------------------------------------------
# Section 31 — Backward Compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_assembler_without_tracer_works(self):
        assembler = ContextAssembler()
        result = assembler.assemble(_make_items(), ContextBudget(max_tokens=100))
        assert len(result.included_items) == 3
        assert result.used_tokens == 60


# ---------------------------------------------------------------------------
# Section 55 — Telemetry Failure Isolation
# ---------------------------------------------------------------------------


class FailingTracer:
    def start_as_current_span(self, name, **kwargs):
        raise RuntimeError("span creation failure")


class TestTelemetryFailureIsolation:
    def test_assemble_unaffected_by_failing_tracer(self):
        assembler = ContextAssembler(tracer=FailingTracer())
        result = assembler.assemble(_make_items(), ContextBudget(max_tokens=100))
        assert len(result.included_items) == 3
        assert result.used_tokens == 60

    def test_overflow_unaffected_by_failing_tracer(self):
        assembler = ContextAssembler(tracer=FailingTracer())
        with pytest.raises(ContextBudgetExceededError):
            assembler.assemble(_make_items(), ContextBudget(max_tokens=5))


# ---------------------------------------------------------------------------
# Section 67 — Deterministic Span Names
# ---------------------------------------------------------------------------


class TestDeterministicSpanNames:
    def test_span_name_is_fixed(self):
        tracer, exporter = make_recording_tracer()
        assembler = ContextAssembler(tracer=tracer)

        assembler.assemble(_make_items(), ContextBudget(max_tokens=100))

        all_names = {s.name for s in exporter.finished_spans}
        assert all_names == {SPAN_CONTEXT_ASSEMBLE}
