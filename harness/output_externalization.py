"""Large tool output externalization & recoverable references (Phase 4 Step 2).

The ``ToolOutputProcessor`` takes a successful tool output, estimates
its token cost, and decides:

* **Inline** — if the output fits within ``inline_token_limit``, it
  becomes a ``RECENT_INTERACTION`` ``ContextItem`` with the full
  rendered content.
* **Externalized** — if the output exceeds the limit, the full content
  is saved to an ``ArtifactStore``, and a small ``RECOVERABLE_REFERENCE``
  ``ContextItem`` (with an ``ArtifactReference``) enters the context
  instead.

Key principle:

    Tool Output != Conversation Context.

The full output is preserved in the artifact store and can always be
read back via ``artifact_id``, even if the reference is omitted by the
``ContextAssembler`` due to budget pressure.

    Omitted != Deleted.

This module does NOT:

* call an LLM for summarization;
* compress content;
* deduplicate by hash;
* integrate with ``ToolRuntime`` / ``AgentActionController`` /
  ``HarnessRuntime``;
* automatically re-inject artifacts into context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Protocol, runtime_checkable

from opentelemetry.trace import Tracer
from opentelemetry.trace.status import Status, StatusCode

from harness.context import (
    ContextItem,
    ContextKind,
    ContextPriority,
)
from observability.tracing import (
    ATTR_ARTIFACT_CONTEXT_ESTIMATED_TOKENS,
    ATTR_ARTIFACT_EXTERNALIZED,
    ATTR_ARTIFACT_ID,
    ATTR_ARTIFACT_ORIGINAL_ESTIMATED_TOKENS,
    ATTR_ARTIFACT_ORIGINAL_SIZE_BYTES,
    ATTR_ARTIFACT_SOURCE_TOOL,
    SPAN_ARTIFACT_PROCESS_OUTPUT,
    resolve_tracer,
    safe_record_exception,
    safe_set_attribute,
    safe_set_status,
    safe_span,
)
from storage.artifact_store import Artifact, ArtifactKind, ArtifactStore


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ToolOutputSerializationError(Exception):
    """Raised when a tool output cannot be deterministically rendered.

    For example: custom objects, NaN, Infinity, or non-JSON-like types.
    ``repr()`` is never used because it can be non-deterministic.
    """


# ---------------------------------------------------------------------------
# Token estimator
# ---------------------------------------------------------------------------


@runtime_checkable
class TokenEstimator(Protocol):
    """Estimates the token cost of a text string.

    This is NOT a model tokenizer. It is a deterministic cost estimate
    for context budgeting. The estimator must return an ``int >= 0``;
    non-empty text must estimate to ``>= 1`` (consistent with the
    ``ContextItem`` zero-token invariant).
    """

    def estimate(self, text: str) -> int:
        ...


def _validate_token_estimate(estimate: int, text: str) -> int:
    """Validate a token estimate and enforce the zero-token invariant."""
    if (
        not isinstance(estimate, int)
        or isinstance(estimate, bool)
    ):
        raise ValueError(
            f"token estimate must be an int, got {type(estimate).__name__}"
        )
    if estimate < 0:
        raise ValueError(f"token estimate must be >= 0, got {estimate}")
    if estimate == 0 and text != "":
        raise ValueError(
            "token estimate=0 requires empty text; non-empty text must "
            "estimate to >= 1"
        )
    return estimate


# ---------------------------------------------------------------------------
# Tool output renderer
# ---------------------------------------------------------------------------


@runtime_checkable
class ToolOutputRenderer(Protocol):
    """Renders a tool output (``Any``) into a deterministic ``str``.

    ``ToolExecutionResult.output`` is typed as ``Any``, so the renderer
    must handle common types (str, None, bool, int, float, list, dict)
    and explicitly reject unsupported types.
    """

    def render(self, output: Any) -> str:
        ...


class DeterministicToolOutputRenderer:
    """Deterministic renderer for JSON-like tool outputs.

    * ``str`` → returned as-is (no JSON quoting).
    * ``None`` → ``"null"``.
    * ``bool`` → ``"true"`` / ``"false"``.
    * ``int`` / ``float`` → JSON number representation.
    * ``list`` / ``dict`` → ``json.dumps(sort_keys=True, ...)``.
    * Unsupported types (custom objects, NaN, Infinity) →
      ``ToolOutputSerializationError``.
    """

    def render(self, output: Any) -> str:
        return _render_output(output)


def _render_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    if output is None:
        return "null"
    if isinstance(output, bool):
        return "true" if output else "false"
    if isinstance(output, (int, float)):
        # json.dumps handles int/float; reject NaN/Inf via allow_nan=False.
        try:
            return json.dumps(output, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ToolOutputSerializationError(
                f"cannot serialize numeric output: {exc}"
            ) from exc
    if isinstance(output, (list, dict)):
        try:
            return json.dumps(
                output,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (ValueError, TypeError) as exc:
            raise ToolOutputSerializationError(
                f"cannot serialize JSON-like output: {exc}"
            ) from exc
    raise ToolOutputSerializationError(
        f"unsupported output type {type(output).__name__}; "
        "only str/None/bool/int/float/list/dict are allowed"
    )


# ---------------------------------------------------------------------------
# Externalization policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputExternalizationPolicy:
    """Policy for deciding inline vs externalized tool output.

    Attributes:
        inline_token_limit: outputs with ``estimated_tokens <= limit``
            stay inline. Outputs exceeding the limit are externalized.
            Must be ``> 0``.
        preview_chars: number of characters from the start of the
            output to include in the artifact reference preview.
            Must be ``>= 0``. ``0`` means empty preview.
    """

    inline_token_limit: int
    preview_chars: int = 200

    def __post_init__(self) -> None:
        if (
            not isinstance(self.inline_token_limit, int)
            or isinstance(self.inline_token_limit, bool)
            or self.inline_token_limit <= 0
        ):
            raise ValueError("inline_token_limit must be an int > 0")
        if (
            not isinstance(self.preview_chars, int)
            or isinstance(self.preview_chars, bool)
            or self.preview_chars < 0
        ):
            raise ValueError("preview_chars must be an int >= 0")


# ---------------------------------------------------------------------------
# Artifact reference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactReference:
    """Small recoverable reference to an externalized artifact.

    This is what enters the model context instead of the full output.
    The full content is always retrievable via ``artifact_id`` from the
    ``ArtifactStore``.

    Attributes:
        artifact_id: the artifact's stable ID.
        content_hash: SHA-256 hex digest of the original content.
        source_tool_name: the tool that produced the output.
        original_size_bytes: UTF-8 byte size of the original content.
        original_estimated_tokens: token estimate of the full content.
        preview: deterministic prefix of the original content
            (``content[:preview_chars]``).
        summary: deterministic descriptive summary (NOT an LLM summary).
    """

    artifact_id: str
    content_hash: str
    source_tool_name: str
    original_size_bytes: int
    original_estimated_tokens: int
    preview: str
    summary: str

    def render(self) -> str:
        """Render this reference as a stable, human-readable string.

        The output is deterministic: the same ``ArtifactReference``
        always produces the same rendered text.
        """
        lines = [
            f"[Artifact {self.artifact_id}]",
            f"Tool: {self.source_tool_name}",
            "Full output externalized.",
            f"Original size: {self.original_size_bytes} bytes",
            f"Original estimated tokens: {self.original_estimated_tokens}",
            f"SHA256: {self.content_hash}",
        ]
        if self.preview:
            lines.append("Preview:")
            lines.append(self.preview)
        return "\n".join(lines)


def _build_summary(
    *,
    tool_name: str,
    size_bytes: int,
    estimated_tokens: int,
) -> str:
    """Build a deterministic descriptive summary.

    This is NOT an LLM semantic summary. It is a fixed-format descriptor
    that is fully determined by its inputs.
    """
    return (
        f"Tool output from {tool_name!r}; "
        f"{size_bytes} bytes; "
        f"estimated {estimated_tokens} tokens."
    )


# ---------------------------------------------------------------------------
# Processing result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolOutputProcessingResult:
    """Result of processing one successful tool output.

    Attributes:
        context_item: the ``ContextItem`` to enter the context (either
            inline full output or a recoverable reference).
        artifact_reference: the ``ArtifactReference`` if externalized,
            or ``None`` if inline.
        externalized: ``True`` if the output was externalized to the
            artifact store.
    """

    context_item: ContextItem
    artifact_reference: ArtifactReference | None
    externalized: bool

    def __post_init__(self) -> None:
        if not isinstance(self.context_item, ContextItem):
            raise ValueError("context_item must be a ContextItem")
        if self.artifact_reference is not None and not isinstance(
            self.artifact_reference, ArtifactReference
        ):
            raise ValueError(
                "artifact_reference must be an ArtifactReference or None"
            )
        if not isinstance(self.externalized, bool):
            raise ValueError("externalized must be a bool")


# ---------------------------------------------------------------------------
# Tool output processor
# ---------------------------------------------------------------------------


# Type aliases for injected factories.
ArtifactIdFactory = Callable[[], str]
ContextItemIdFactory = Callable[[], str]
Clock = Callable[[], datetime]


class ToolOutputProcessor:
    """Processes successful tool outputs for context budgeting.

    Decides whether a tool output stays inline in the context or is
    externalized to an ``ArtifactStore`` with a recoverable reference.

    Dependencies are injected for deterministic testing:

    * ``artifact_store`` — where full outputs are saved.
    * ``token_estimator`` — estimates token cost of text.
    * ``output_renderer`` — renders ``Any`` output to ``str``.
    * ``externalization_policy`` — inline vs externalize threshold.
    * ``artifact_id_factory`` — produces deterministic artifact IDs.
    * ``context_item_id_factory`` — produces deterministic context item IDs.
    * ``clock`` — produces timezone-aware datetimes.
    """

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        token_estimator: TokenEstimator,
        output_renderer: ToolOutputRenderer,
        externalization_policy: OutputExternalizationPolicy,
        artifact_id_factory: ArtifactIdFactory,
        context_item_id_factory: ContextItemIdFactory,
        clock: Clock,
        tracer: Tracer | None = None,
    ) -> None:
        self._store = artifact_store
        self._estimator = token_estimator
        self._renderer = output_renderer
        self._policy = externalization_policy
        self._artifact_id_factory = artifact_id_factory
        self._context_item_id_factory = context_item_id_factory
        self._clock = clock
        self._tracer = resolve_tracer(tracer)

    async def process_success_output(
        self,
        *,
        tool_name: str,
        output: Any,
        sequence_index: int,
        priority: ContextPriority = ContextPriority.NORMAL,
    ) -> ToolOutputProcessingResult:
        """Process one successful tool output.

        Returns a ``ToolOutputProcessingResult`` containing either an
        inline ``ContextItem`` (full output) or an externalized
        ``ContextItem`` (recoverable reference).

        If the output is externalized, the full content is saved to the
        artifact store BEFORE the reference is returned. If the save
        fails, the error propagates — no dangling references.
        """
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("tool_name must be a non-empty str")
        if (
            not isinstance(sequence_index, int)
            or isinstance(sequence_index, bool)
            or sequence_index < 0
        ):
            raise ValueError("sequence_index must be an int >= 0")
        if not isinstance(priority, ContextPriority):
            raise ValueError("priority must be a ContextPriority")

        with safe_span(self._tracer, SPAN_ARTIFACT_PROCESS_OUTPUT) as span:
            safe_set_attribute(span, ATTR_ARTIFACT_SOURCE_TOOL, tool_name)
            return await self._process_with_span(
                tool_name=tool_name,
                output=output,
                sequence_index=sequence_index,
                priority=priority,
                span=span,
            )

    async def _process_with_span(
        self,
        *,
        tool_name: str,
        output: Any,
        sequence_index: int,
        priority: ContextPriority,
        span: object,
    ) -> ToolOutputProcessingResult:
        """Internal: process with an active span for telemetry."""
        # Render the output to a deterministic string.
        rendered = self._renderer.render(output)

        # Estimate token cost of the full rendered output.
        full_tokens = _validate_token_estimate(
            self._estimator.estimate(rendered), rendered
        )

        # Decide: inline or externalize.
        if full_tokens <= self._policy.inline_token_limit:
            result = self._process_inline(
                rendered=rendered,
                full_tokens=full_tokens,
                tool_name=tool_name,
                sequence_index=sequence_index,
                priority=priority,
            )
            safe_set_attribute(span, ATTR_ARTIFACT_EXTERNALIZED, False)
            safe_set_attribute(
                span, ATTR_ARTIFACT_ORIGINAL_ESTIMATED_TOKENS, full_tokens
            )
            safe_set_attribute(
                span, ATTR_ARTIFACT_CONTEXT_ESTIMATED_TOKENS, full_tokens
            )
            safe_set_status(span, Status(StatusCode.OK))
            return result

        # Externalized path.
        try:
            result = await self._process_externalized(
                rendered=rendered,
                full_tokens=full_tokens,
                tool_name=tool_name,
                sequence_index=sequence_index,
                priority=priority,
            )
        except Exception as exc:
            safe_record_exception(span, exc)
            safe_set_status(span, Status(StatusCode.ERROR))
            raise

        safe_set_attribute(span, ATTR_ARTIFACT_EXTERNALIZED, True)
        safe_set_attribute(span, ATTR_ARTIFACT_ID, result.artifact_reference.artifact_id)
        safe_set_attribute(
            span,
            ATTR_ARTIFACT_ORIGINAL_SIZE_BYTES,
            result.artifact_reference.original_size_bytes,
        )
        safe_set_attribute(
            span,
            ATTR_ARTIFACT_ORIGINAL_ESTIMATED_TOKENS,
            result.artifact_reference.original_estimated_tokens,
        )
        safe_set_attribute(
            span,
            ATTR_ARTIFACT_CONTEXT_ESTIMATED_TOKENS,
            result.context_item.estimated_tokens,
        )
        safe_set_status(span, Status(StatusCode.OK))
        return result

    # ------------------------------------------------------------------
    # Inline flow
    # ------------------------------------------------------------------

    def _process_inline(
        self,
        *,
        rendered: str,
        full_tokens: int,
        tool_name: str,
        sequence_index: int,
        priority: ContextPriority,
    ) -> ToolOutputProcessingResult:
        """Inline: full output enters context, no artifact created."""
        context_item = ContextItem(
            item_id=self._context_item_id_factory(),
            kind=ContextKind.RECENT_INTERACTION,
            content=rendered,
            estimated_tokens=full_tokens,
            sequence_index=sequence_index,
            priority=priority,
            must_keep=False,
            compressible=False,
            externalizable=True,
        )
        return ToolOutputProcessingResult(
            context_item=context_item,
            artifact_reference=None,
            externalized=False,
        )

    # ------------------------------------------------------------------
    # Externalized flow
    # ------------------------------------------------------------------

    async def _process_externalized(
        self,
        *,
        rendered: str,
        full_tokens: int,
        tool_name: str,
        sequence_index: int,
        priority: ContextPriority,
    ) -> ToolOutputProcessingResult:
        """Externalize: save full output to store, return reference."""
        artifact_id = self._artifact_id_factory()
        created_at = self._clock()

        # Create the artifact with derived fields computed internally.
        artifact = Artifact.create(
            artifact_id=artifact_id,
            kind=ArtifactKind.TOOL_OUTPUT,
            content=rendered,
            estimated_tokens=full_tokens,
            created_at=created_at,
            source_tool_name=tool_name,
        )

        # Save BEFORE returning the reference. If save fails, the error
        # propagates — no dangling references.
        await self._store.save(artifact)

        # Build the reference.
        preview = rendered[: self._policy.preview_chars]
        summary = _build_summary(
            tool_name=tool_name,
            size_bytes=artifact.size_bytes,
            estimated_tokens=full_tokens,
        )
        reference = ArtifactReference(
            artifact_id=artifact_id,
            content_hash=artifact.content_hash,
            source_tool_name=tool_name,
            original_size_bytes=artifact.size_bytes,
            original_estimated_tokens=full_tokens,
            preview=preview,
            summary=summary,
        )

        # Render the reference text for the context item.
        reference_text = reference.render()
        reference_tokens = _validate_token_estimate(
            self._estimator.estimate(reference_text), reference_text
        )

        context_item = ContextItem(
            item_id=self._context_item_id_factory(),
            kind=ContextKind.RECOVERABLE_REFERENCE,
            content=reference_text,
            estimated_tokens=reference_tokens,
            sequence_index=sequence_index,
            priority=priority,
            must_keep=False,
            compressible=False,
            externalizable=False,  # already externalized
        )

        return ToolOutputProcessingResult(
            context_item=context_item,
            artifact_reference=reference,
            externalized=True,
        )
