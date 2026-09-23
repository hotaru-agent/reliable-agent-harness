"""Tests for ToolOutputProcessor OpenTelemetry tracing (Phase 6 Step 2).

All offline, deterministic, no network, no collector. Uses a recording
span exporter to assert on span names, attributes, statuses.

Tests prove:
* artifact.process_output span exists with source tool
* inline: externalized=False, token attributes, OK status
* externalized: externalized=True, artifact.id, size/tokens, OK status
* store save failure: ERROR status, exception type recorded
* no preview/full output in telemetry
* no content hash in telemetry
* tracer injection isolation
* telemetry failure isolation
"""

from __future__ import annotations

import asyncio

import pytest
from opentelemetry.trace.status import Status, StatusCode

from harness.output_externalization import (
    DeterministicToolOutputRenderer,
    OutputExternalizationPolicy,
    TokenEstimator,
    ToolOutputProcessor,
)
from observability import (
    ATTR_ARTIFACT_CONTEXT_ESTIMATED_TOKENS,
    ATTR_ARTIFACT_EXTERNALIZED,
    ATTR_ARTIFACT_ID,
    ATTR_ARTIFACT_ORIGINAL_ESTIMATED_TOKENS,
    ATTR_ARTIFACT_ORIGINAL_SIZE_BYTES,
    ATTR_ARTIFACT_SOURCE_TOOL,
    ATTR_ERROR_EXCEPTION_TYPE,
    SPAN_ARTIFACT_PROCESS_OUTPUT,
)
from storage.artifact_store import (
    Artifact,
    ArtifactStore,
    InMemoryArtifactStore,
)
from tests.fakes_telemetry import make_recording_tracer


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

_SECRET_MARKER = "VERY_SECRET_ARTIFACT_MARKER"


class FakeTokenEstimator:
    def __init__(self, mode: str = "char_div_4"):
        self.mode = mode
        self.call_count = 0

    def estimate(self, text: str) -> int:
        self.call_count += 1
        if not text:
            return 0
        if self.mode == "char_div_4":
            return max(1, len(text) // 4)
        raise ValueError(f"unknown mode: {self.mode}")


class FailingArtifactStore:
    async def save(self, artifact: Artifact) -> None:
        raise RuntimeError("store save failed")

    async def get(self, artifact_id: str) -> Artifact | None:
        return None


class _IdFactory:
    def __init__(self, prefix: str, start: int = 1):
        self.prefix = prefix
        self.counter = start
        self.consumed: list[str] = []

    def __call__(self) -> str:
        id_str = f"{self.prefix}-{self.counter:03d}"
        self.counter += 1
        self.consumed.append(id_str)
        return id_str


def _utc_now():
    from datetime import datetime, timezone
    return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_traced_processor(
    tracer,
    *,
    store: ArtifactStore | None = None,
    policy: OutputExternalizationPolicy | None = None,
    estimator: TokenEstimator | None = None,
) -> ToolOutputProcessor:
    return ToolOutputProcessor(
        artifact_store=store or InMemoryArtifactStore(),
        token_estimator=estimator or FakeTokenEstimator(),
        output_renderer=DeterministicToolOutputRenderer(),
        externalization_policy=policy or OutputExternalizationPolicy(
            inline_token_limit=100, preview_chars=50,
        ),
        artifact_id_factory=_IdFactory("A"),
        context_item_id_factory=_IdFactory("CTX"),
        clock=_utc_now,
        tracer=tracer,
    )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Section 36 — Inline Output
# ---------------------------------------------------------------------------


class TestInlineOutputTracing:
    def test_span_exists(self):
        tracer, exporter = make_recording_tracer()
        proc = _make_traced_processor(tracer)

        _run(
            proc.process_success_output(
                tool_name="echo",
                output="small output",
                sequence_index=0,
            )
        )

        spans = exporter.spans_named(SPAN_ARTIFACT_PROCESS_OUTPUT)
        assert len(spans) == 1

    def test_source_tool_attribute(self):
        tracer, exporter = make_recording_tracer()
        proc = _make_traced_processor(tracer)

        _run(
            proc.process_success_output(
                tool_name="echo",
                output="small output",
                sequence_index=0,
            )
        )

        span = exporter.span_by_name(SPAN_ARTIFACT_PROCESS_OUTPUT)
        assert span.attributes[ATTR_ARTIFACT_SOURCE_TOOL] == "echo"

    def test_externalized_false(self):
        tracer, exporter = make_recording_tracer()
        proc = _make_traced_processor(tracer)

        _run(
            proc.process_success_output(
                tool_name="echo",
                output="small output",
                sequence_index=0,
            )
        )

        span = exporter.span_by_name(SPAN_ARTIFACT_PROCESS_OUTPUT)
        assert span.attributes[ATTR_ARTIFACT_EXTERNALIZED] is False

    def test_token_attributes(self):
        tracer, exporter = make_recording_tracer()
        proc = _make_traced_processor(tracer)

        _run(
            proc.process_success_output(
                tool_name="echo",
                output="small output",
                sequence_index=0,
            )
        )

        span = exporter.span_by_name(SPAN_ARTIFACT_PROCESS_OUTPUT)
        assert ATTR_ARTIFACT_ORIGINAL_ESTIMATED_TOKENS in span.attributes
        assert ATTR_ARTIFACT_CONTEXT_ESTIMATED_TOKENS in span.attributes

    def test_no_artifact_id_for_inline(self):
        tracer, exporter = make_recording_tracer()
        proc = _make_traced_processor(tracer)

        _run(
            proc.process_success_output(
                tool_name="echo",
                output="small output",
                sequence_index=0,
            )
        )

        span = exporter.span_by_name(SPAN_ARTIFACT_PROCESS_OUTPUT)
        assert ATTR_ARTIFACT_ID not in span.attributes

    def test_ok_status(self):
        tracer, exporter = make_recording_tracer()
        proc = _make_traced_processor(tracer)

        _run(
            proc.process_success_output(
                tool_name="echo",
                output="small output",
                sequence_index=0,
            )
        )

        span = exporter.span_by_name(SPAN_ARTIFACT_PROCESS_OUTPUT)
        assert span.status.status_code == StatusCode.OK


# ---------------------------------------------------------------------------
# Section 37 — Externalized Output
# ---------------------------------------------------------------------------


class TestExternalizedOutputTracing:
    def test_externalized_true(self):
        tracer, exporter = make_recording_tracer()
        proc = _make_traced_processor(
            tracer,
            policy=OutputExternalizationPolicy(inline_token_limit=2, preview_chars=10),
        )

        _run(
            proc.process_success_output(
                tool_name="big_tool",
                output=_SECRET_MARKER * 20,
                sequence_index=0,
            )
        )

        span = exporter.span_by_name(SPAN_ARTIFACT_PROCESS_OUTPUT)
        assert span.attributes[ATTR_ARTIFACT_EXTERNALIZED] is True

    def test_artifact_id_recorded(self):
        tracer, exporter = make_recording_tracer()
        proc = _make_traced_processor(
            tracer,
            policy=OutputExternalizationPolicy(inline_token_limit=2, preview_chars=10),
        )

        _run(
            proc.process_success_output(
                tool_name="big_tool",
                output=_SECRET_MARKER * 20,
                sequence_index=0,
            )
        )

        span = exporter.span_by_name(SPAN_ARTIFACT_PROCESS_OUTPUT)
        assert span.attributes[ATTR_ARTIFACT_ID] == "A-001"

    def test_size_bytes_recorded(self):
        tracer, exporter = make_recording_tracer()
        proc = _make_traced_processor(
            tracer,
            policy=OutputExternalizationPolicy(inline_token_limit=2, preview_chars=10),
        )

        _run(
            proc.process_success_output(
                tool_name="big_tool",
                output=_SECRET_MARKER * 20,
                sequence_index=0,
            )
        )

        span = exporter.span_by_name(SPAN_ARTIFACT_PROCESS_OUTPUT)
        assert ATTR_ARTIFACT_ORIGINAL_SIZE_BYTES in span.attributes

    def test_token_attributes(self):
        tracer, exporter = make_recording_tracer()
        proc = _make_traced_processor(
            tracer,
            policy=OutputExternalizationPolicy(inline_token_limit=2, preview_chars=10),
        )

        _run(
            proc.process_success_output(
                tool_name="big_tool",
                output=_SECRET_MARKER * 20,
                sequence_index=0,
            )
        )

        span = exporter.span_by_name(SPAN_ARTIFACT_PROCESS_OUTPUT)
        assert ATTR_ARTIFACT_ORIGINAL_ESTIMATED_TOKENS in span.attributes
        assert ATTR_ARTIFACT_CONTEXT_ESTIMATED_TOKENS in span.attributes

    def test_ok_status(self):
        tracer, exporter = make_recording_tracer()
        proc = _make_traced_processor(
            tracer,
            policy=OutputExternalizationPolicy(inline_token_limit=2, preview_chars=10),
        )

        _run(
            proc.process_success_output(
                tool_name="big_tool",
                output=_SECRET_MARKER * 20,
                sequence_index=0,
            )
        )

        span = exporter.span_by_name(SPAN_ARTIFACT_PROCESS_OUTPUT)
        assert span.status.status_code == StatusCode.OK


# ---------------------------------------------------------------------------
# Section 38 — Store Save Failure
# ---------------------------------------------------------------------------


class TestStoreSaveFailure:
    def test_error_status(self):
        tracer, exporter = make_recording_tracer()
        proc = _make_traced_processor(
            tracer,
            store=FailingArtifactStore(),
            policy=OutputExternalizationPolicy(inline_token_limit=2, preview_chars=10),
        )

        with pytest.raises(RuntimeError):
            _run(
                proc.process_success_output(
                    tool_name="big_tool",
                    output=_SECRET_MARKER * 20,
                    sequence_index=0,
                )
            )

        span = exporter.span_by_name(SPAN_ARTIFACT_PROCESS_OUTPUT)
        assert span.status.status_code == StatusCode.ERROR

    def test_exception_type_recorded(self):
        tracer, exporter = make_recording_tracer()
        proc = _make_traced_processor(
            tracer,
            store=FailingArtifactStore(),
            policy=OutputExternalizationPolicy(inline_token_limit=2, preview_chars=10),
        )

        with pytest.raises(RuntimeError):
            _run(
                proc.process_success_output(
                    tool_name="big_tool",
                    output=_SECRET_MARKER * 20,
                    sequence_index=0,
                )
            )

        span = exporter.span_by_name(SPAN_ARTIFACT_PROCESS_OUTPUT)
        assert span.attributes[ATTR_ERROR_EXCEPTION_TYPE] == "RuntimeError"


# ---------------------------------------------------------------------------
# Section 52 — No Sensitive Output / Preview / Hash
# ---------------------------------------------------------------------------


class TestNoSensitiveData:
    def test_secret_marker_not_in_telemetry(self):
        tracer, exporter = make_recording_tracer()
        proc = _make_traced_processor(
            tracer,
            policy=OutputExternalizationPolicy(inline_token_limit=2, preview_chars=10),
        )

        _run(
            proc.process_success_output(
                tool_name="big_tool",
                output=_SECRET_MARKER * 20,
                sequence_index=0,
            )
        )

        for s in exporter.finished_spans:
            for key, val in (s.attributes or {}).items():
                assert _SECRET_MARKER not in str(val)
            for ev in s.events:
                for key, val in (ev.attributes or {}).items():
                    assert _SECRET_MARKER not in str(val)

    def test_no_content_hash_in_telemetry(self):
        tracer, exporter = make_recording_tracer()
        proc = _make_traced_processor(
            tracer,
            policy=OutputExternalizationPolicy(inline_token_limit=2, preview_chars=10),
        )

        _run(
            proc.process_success_output(
                tool_name="big_tool",
                output=_SECRET_MARKER * 20,
                sequence_index=0,
            )
        )

        for s in exporter.finished_spans:
            for key in (s.attributes or {}).keys():
                assert "hash" not in str(key).lower()
                assert "content_hash" not in str(key).lower()

    def test_no_preview_in_telemetry(self):
        tracer, exporter = make_recording_tracer()
        proc = _make_traced_processor(
            tracer,
            policy=OutputExternalizationPolicy(inline_token_limit=2, preview_chars=10),
        )

        _run(
            proc.process_success_output(
                tool_name="big_tool",
                output=_SECRET_MARKER * 20,
                sequence_index=0,
            )
        )

        for s in exporter.finished_spans:
            for key in (s.attributes or {}).keys():
                assert "preview" not in str(key).lower()
            for ev in s.events:
                for key in (ev.attributes or {}).keys():
                    assert "preview" not in str(key).lower()

    def test_business_store_still_has_content(self):
        tracer, exporter = make_recording_tracer()
        store = InMemoryArtifactStore()
        proc = _make_traced_processor(
            tracer,
            store=store,
            policy=OutputExternalizationPolicy(inline_token_limit=2, preview_chars=10),
        )

        result = _run(
            proc.process_success_output(
                tool_name="big_tool",
                output=_SECRET_MARKER * 20,
                sequence_index=0,
            )
        )

        # Business store has the full content.
        artifact = _run(store.get(result.artifact_reference.artifact_id))
        assert artifact is not None
        assert _SECRET_MARKER in artifact.content


# ---------------------------------------------------------------------------
# Section 55 — Telemetry Failure Isolation
# ---------------------------------------------------------------------------


class FailingTracer:
    def start_as_current_span(self, name, **kwargs):
        raise RuntimeError("span creation failure")


class TestTelemetryFailureIsolation:
    def test_inline_unaffected_by_failing_tracer(self):
        proc = _make_traced_processor(FailingTracer())
        result = _run(
            proc.process_success_output(
                tool_name="echo",
                output="small output",
                sequence_index=0,
            )
        )
        assert result.externalized is False

    def test_externalized_unaffected_by_failing_tracer(self):
        store = InMemoryArtifactStore()
        proc = _make_traced_processor(
            FailingTracer(),
            store=store,
            policy=OutputExternalizationPolicy(inline_token_limit=2, preview_chars=10),
        )
        result = _run(
            proc.process_success_output(
                tool_name="big_tool",
                output=_SECRET_MARKER * 20,
                sequence_index=0,
            )
        )
        assert result.externalized is True
        artifact = _run(store.get(result.artifact_reference.artifact_id))
        assert artifact is not None


# ---------------------------------------------------------------------------
# Section 67 — Deterministic Span Names
# ---------------------------------------------------------------------------


class TestDeterministicSpanNames:
    def test_span_name_is_fixed(self):
        tracer, exporter = make_recording_tracer()
        proc = _make_traced_processor(tracer)

        _run(
            proc.process_success_output(
                tool_name="echo",
                output="small output",
                sequence_index=0,
            )
        )

        all_names = {s.name for s in exporter.finished_spans}
        assert all_names == {SPAN_ARTIFACT_PROCESS_OUTPUT}


# ---------------------------------------------------------------------------
# No SDK Configured
# ---------------------------------------------------------------------------


class TestNoSDKConfigured:
    def test_business_behavior_without_tracer(self):
        store = InMemoryArtifactStore()
        proc = ToolOutputProcessor(
            artifact_store=store,
            token_estimator=FakeTokenEstimator(),
            output_renderer=DeterministicToolOutputRenderer(),
            externalization_policy=OutputExternalizationPolicy(
                inline_token_limit=2, preview_chars=10,
            ),
            artifact_id_factory=_IdFactory("A"),
            context_item_id_factory=_IdFactory("CTX"),
            clock=_utc_now,
        )

        result = _run(
            proc.process_success_output(
                tool_name="big_tool",
                output=_SECRET_MARKER * 20,
                sequence_index=0,
            )
        )
        assert result.externalized is True
        assert result.artifact_reference.artifact_id == "A-001"
