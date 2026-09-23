"""Deterministic in-memory telemetry fakes for Phase 6 Step 1 tests.

Provides a ``RecordingSpanExporter`` that collects finished spans in
memory, and a helper ``make_recording_tracer`` that builds a
``TracerProvider`` + ``SimpleSpanProcessor`` + exporter and returns the
tracer + exporter pair.

Tests use these to assert on span names, attributes, events, statuses
and parent-child relationships without any network, collector, or
background timing.

No OTLP, no Jaeger, no Zipkin, no Docker.
"""

from __future__ import annotations

from typing import Optional

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Tracer


class RecordingSpanExporter(SpanExporter):
    """In-memory span exporter that collects finished spans.

    Spans are stored in ``finished_spans`` in the order they are
    exported (i.e. finished). Each entry is the SDK ``Span`` object,
    which exposes ``name``, ``attributes``, ``events``, ``status``,
    ``parent`` and ``context``.
    """

    def __init__(self) -> None:
        self.finished_spans: list = []

    def export(self, spans) -> SpanExportResult:
        self.finished_spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def clear(self) -> None:
        self.finished_spans.clear()

    # ------------------------------------------------------------------
    # Convenience query helpers
    # ------------------------------------------------------------------

    def spans_named(self, name: str) -> list:
        """Return all finished spans with the given name."""
        return [s for s in self.finished_spans if s.name == name]

    def span_by_name(self, name: str) -> Optional[object]:
        """Return the first finished span with the given name, or None."""
        for s in self.finished_spans:
            if s.name == name:
                return s
        return None


def make_recording_tracer(
    *,
    scope: str = "reliable_agent_harness",
) -> tuple[Tracer, RecordingSpanExporter]:
    """Build a recording tracer + exporter pair for tests.

    Returns ``(tracer, exporter)``. The tracer is obtained from a fresh
    ``TracerProvider`` (NOT set as the global provider) so tests do not
    pollute each other.
    """
    provider = TracerProvider()
    exporter = RecordingSpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(scope)
    return tracer, exporter
