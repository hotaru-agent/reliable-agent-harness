"""Context items, budget & deterministic assembly (Phase 4 Step 1).

This module implements deterministic, rule-based context budgeting for
long-horizon tool-using agents. The core idea is:

    Context Is A Budget.

Many ``ContextItem`` candidates exist; a ``ContextBudget`` limits how
many tokens can be assembled; ``ContextAssembler`` deterministically
selects which items survive within budget.

Selection policy (priority DESC, then recency DESC, then item_id ASC)
is decoupled from render order (sequence_index ASC, chronological).

The assembler does NOT:

* call an LLM for summarization;
* compress content;
* externalize large tool output to an artifact store;
* modify items;
* integrate with ``HarnessRuntime`` / ``AgentActionController`` /
  ``ToolRuntime``.

It is a pure, stateless, deterministic selection function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Sequence, Tuple

from opentelemetry.trace import Tracer
from opentelemetry.trace.status import Status, StatusCode

from observability.tracing import (
    ATTR_CONTEXT_INCLUDED_COUNT,
    ATTR_CONTEXT_INPUT_COUNT,
    ATTR_CONTEXT_MUST_KEEP_COUNT,
    ATTR_CONTEXT_OMITTED_COUNT,
    ATTR_CONTEXT_TOKEN_MAX,
    ATTR_CONTEXT_TOKEN_REMAINING,
    ATTR_CONTEXT_TOKEN_REQUIRED,
    ATTR_CONTEXT_TOKEN_USED,
    SPAN_CONTEXT_ASSEMBLE,
    resolve_tracer,
    safe_record_exception,
    safe_set_attribute,
    safe_set_status,
    safe_span,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ContextAssemblyError(Exception):
    """Raised when context assembly input is invalid.

    For example: duplicate ``item_id`` or duplicate ``sequence_index``.
    """


class ContextBudgetExceededError(Exception):
    """Raised when must-keep items alone exceed the budget.

    The runtime must NOT silently drop critical state to fit a budget.
    The error carries ``required_tokens`` and ``available_tokens`` so
    callers can diagnose the overflow.

    Attributes:
        required_tokens: total tokens of all must-keep items.
        available_tokens: the budget's ``max_tokens``.
    """

    def __init__(self, required_tokens: int, available_tokens: int) -> None:
        self.required_tokens = required_tokens
        self.available_tokens = available_tokens
        super().__init__(
            f"Critical context requires {required_tokens} tokens, "
            f"but budget is {available_tokens}."
        )


# ---------------------------------------------------------------------------
# Context kind & priority
# ---------------------------------------------------------------------------


class ContextKind(Enum):
    """What a ``ContextItem`` is.

    * ``CRITICAL_STATE`` — must be known to resume / continue reasoning
      (task goal, constraints, current plan, important decisions,
      replan requirement).
    * ``RECENT_INTERACTION`` — recently happened (agent action, tool
      observation, tool result, recent reasoning evidence).
    * ``HISTORICAL_EVIDENCE`` — old logs / observations / tool results
      that may still have value but can be dropped first when budget
      is tight.
    * ``RECOVERABLE_REFERENCE`` — a reference to content that can be
      re-fetched later (e.g. ``artifact:A-102``). Modeled here but the
      artifact store itself is Phase 4 Step 2.
    """

    CRITICAL_STATE = "critical_state"
    RECENT_INTERACTION = "recent_interaction"
    HISTORICAL_EVIDENCE = "historical_evidence"
    RECOVERABLE_REFERENCE = "recoverable_reference"


class ContextPriority(IntEnum):
    """How important an item is when budget is tight.

    Higher value = higher priority. Selection sorts priority DESC.
    """

    LOW = 10
    NORMAL = 20
    HIGH = 30
    CRITICAL = 40


# ---------------------------------------------------------------------------
# Context item
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextItem:
    """One candidate for the assembled context.

    Attributes:
        item_id: stable identity; must be non-empty and unique within
            an assembly input.
        kind: what this item is (see ``ContextKind``).
        content: the text content prepared for the agent / model.
            This phase only supports ``str``.
        estimated_tokens: budget cost. Caller-supplied estimate; not a
            model tokenizer result. Must be ``>= 0``.
        sequence_index: position in the context timeline (0, 1, 2, ...).
            Used for recency ranking and final render order. Must be
            ``>= 0`` and unique within an assembly input.
        priority: budget-tightness importance. Defaults to ``NORMAL``.
        must_keep: if ``True``, the assembler must never silently drop
            this item. ``CRITICAL_STATE`` items are forced to
            ``must_keep=True``.
        compressible: metadata flag — future compression capability.
            This phase does NOT compress.
        externalizable: metadata flag — future artifact externalization
            capability. This phase does NOT externalize.
    """

    item_id: str
    kind: ContextKind
    content: str
    estimated_tokens: int
    sequence_index: int
    priority: ContextPriority = ContextPriority.NORMAL
    must_keep: bool = False
    compressible: bool = False
    externalizable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id:
            raise ValueError("item_id must be a non-empty str")
        if not isinstance(self.kind, ContextKind):
            raise ValueError("kind must be a ContextKind")
        if not isinstance(self.content, str):
            raise ValueError("content must be a str")
        if (
            not isinstance(self.estimated_tokens, int)
            or isinstance(self.estimated_tokens, bool)
            or self.estimated_tokens < 0
        ):
            raise ValueError("estimated_tokens must be an int >= 0")
        # Phase 4 Step 2: zero-token items must have empty content.
        # This prevents a non-empty item from bypassing the budget by
        # claiming 0 tokens.
        if self.estimated_tokens == 0 and self.content != "":
            raise ValueError(
                "estimated_tokens=0 requires empty content; non-empty "
                "content must have estimated_tokens >= 1"
            )
        if (
            not isinstance(self.sequence_index, int)
            or isinstance(self.sequence_index, bool)
            or self.sequence_index < 0
        ):
            raise ValueError("sequence_index must be an int >= 0")
        if not isinstance(self.priority, ContextPriority):
            raise ValueError("priority must be a ContextPriority")
        if not isinstance(self.must_keep, bool):
            raise ValueError("must_keep must be a bool")
        if not isinstance(self.compressible, bool):
            raise ValueError("compressible must be a bool")
        if not isinstance(self.externalizable, bool):
            raise ValueError("externalizable must be a bool")
        # Critical-state invariant: CRITICAL_STATE must be must_keep.
        if self.kind is ContextKind.CRITICAL_STATE and not self.must_keep:
            raise ValueError(
                "CRITICAL_STATE items must have must_keep=True; "
                "critical state must never be silently droppable"
            )


# ---------------------------------------------------------------------------
# Context budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextBudget:
    """Token budget for one context assembly.

    Attributes:
        max_tokens: maximum total ``estimated_tokens`` the assembler may
            include. Must be ``> 0``.
    """

    max_tokens: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_tokens, int)
            or isinstance(self.max_tokens, bool)
            or self.max_tokens <= 0
        ):
            raise ValueError("max_tokens must be an int > 0")


# ---------------------------------------------------------------------------
# Context selection result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextSelectionResult:
    """Outcome of one ``ContextAssembler.assemble()`` call.

    Attributes:
        included_items: items selected within budget, ordered by
            ``sequence_index`` ASC (chronological render order).
        omitted_items: items NOT selected, ordered by ``sequence_index``
            ASC for deterministic debugging.
        used_tokens: total ``estimated_tokens`` of included items.
        max_tokens: the budget's ``max_tokens``.

    Invariants:
        * ``used_tokens >= 0``
        * ``used_tokens <= max_tokens``
        * included ``item_id``s are unique
        * ``included + omitted == all input items`` (assembler guarantees)
    """

    included_items: Tuple[ContextItem, ...]
    omitted_items: Tuple[ContextItem, ...]
    used_tokens: int
    max_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.included_items, tuple):
            raise ValueError("included_items must be a tuple")
        if not isinstance(self.omitted_items, tuple):
            raise ValueError("omitted_items must be a tuple")
        if not isinstance(self.used_tokens, int) or self.used_tokens < 0:
            raise ValueError("used_tokens must be an int >= 0")
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError("max_tokens must be an int > 0")
        if self.used_tokens > self.max_tokens:
            raise ValueError(
                f"used_tokens ({self.used_tokens}) must not exceed "
                f"max_tokens ({self.max_tokens})"
            )
        # Included item_ids must be unique.
        ids = [it.item_id for it in self.included_items]
        if len(ids) != len(set(ids)):
            raise ValueError("included_items has duplicate item_id")

    @property
    def remaining_tokens(self) -> int:
        """Tokens left in the budget after selection."""
        return self.max_tokens - self.used_tokens


# ---------------------------------------------------------------------------
# Context assembler
# ---------------------------------------------------------------------------


class ContextAssembler:
    """Deterministic, stateless context budget assembler.

    ``assemble(items, budget)`` is a pure function: same inputs always
    produce the same ``ContextSelectionResult``. The assembler holds no
    mutable state.

    Selection policy:

    1. Select all ``must_keep=True`` items. If their total exceeds
       ``budget.max_tokens``, raise ``ContextBudgetExceededError``.
    2. Rank remaining (optional) items by:
       a. ``priority`` DESC (higher priority first)
       b. ``sequence_index`` DESC (more recent first)
       c. ``item_id`` ASC (stable tie-break)
    3. Greedily fit optional items in ranking order. An item that does
       not fit is omitted; selection continues to the next candidate
       (a later smaller item may still fit).

    Render order:

    * ``included_items`` and ``omitted_items`` are both ordered by
      ``sequence_index`` ASC (chronological), NOT by selection ranking.

    An optional ``tracer`` may be injected for OpenTelemetry tracing.
    When ``None`` (default), a no-op tracer is used so behaviour is
    unaffected. Telemetry failure never becomes business failure.
    """

    def __init__(
        self,
        *,
        tracer: Tracer | None = None,
    ) -> None:
        self._tracer = resolve_tracer(tracer)

    def assemble(
        self,
        items: Sequence[ContextItem],
        budget: ContextBudget,
    ) -> ContextSelectionResult:
        """Assemble ``items`` within ``budget`` deterministically."""
        with safe_span(self._tracer, SPAN_CONTEXT_ASSEMBLE) as span:
            safe_set_attribute(span, ATTR_CONTEXT_TOKEN_MAX, budget.max_tokens)
            safe_set_attribute(span, ATTR_CONTEXT_INPUT_COUNT, len(items))
            return self._assemble_with_span(items, budget, span)

    def _assemble_with_span(
        self,
        items: Sequence[ContextItem],
        budget: ContextBudget,
        span: object,
    ) -> ContextSelectionResult:
        """Internal: assemble with an active span for telemetry."""
        # --- Validate input ---
        try:
            validated = self._validate(items)
        except Exception as exc:
            safe_record_exception(span, exc)
            safe_set_status(span, Status(StatusCode.ERROR))
            raise

        # --- Partition must_keep vs optional ---
        must_keep = [it for it in validated if it.must_keep]
        optional = [it for it in validated if not it.must_keep]

        safe_set_attribute(span, ATTR_CONTEXT_MUST_KEEP_COUNT, len(must_keep))

        # --- Step 1: must-keep overflow check ---
        required_tokens = sum(it.estimated_tokens for it in must_keep)
        if required_tokens > budget.max_tokens:
            safe_set_attribute(span, ATTR_CONTEXT_TOKEN_REQUIRED, required_tokens)
            safe_record_exception(span, ContextBudgetExceededError(
                required_tokens=required_tokens,
                available_tokens=budget.max_tokens,
            ))
            safe_set_status(span, Status(StatusCode.ERROR))
            raise ContextBudgetExceededError(
                required_tokens=required_tokens,
                available_tokens=budget.max_tokens,
            )

        # --- Step 2: rank optional candidates ---
        ranked = self._rank(optionals=optional)

        # --- Step 3: greedy fit ---
        remaining = budget.max_tokens - required_tokens
        included_optional: list[ContextItem] = []
        omitted_optional: list[ContextItem] = []
        for item in ranked:
            if item.estimated_tokens <= remaining:
                included_optional.append(item)
                remaining -= item.estimated_tokens
            else:
                omitted_optional.append(item)

        # --- Combine & sort by sequence_index ASC (render order) ---
        all_included = sorted(
            must_keep + included_optional,
            key=lambda it: it.sequence_index,
        )
        all_omitted = sorted(
            omitted_optional,
            key=lambda it: it.sequence_index,
        )
        used = sum(it.estimated_tokens for it in all_included)

        # Record final counts/tokens on the span.
        safe_set_attribute(span, ATTR_CONTEXT_INCLUDED_COUNT, len(all_included))
        safe_set_attribute(span, ATTR_CONTEXT_OMITTED_COUNT, len(all_omitted))
        safe_set_attribute(span, ATTR_CONTEXT_TOKEN_USED, used)
        safe_set_attribute(
            span, ATTR_CONTEXT_TOKEN_REMAINING, budget.max_tokens - used
        )
        safe_set_status(span, Status(StatusCode.OK))

        return ContextSelectionResult(
            included_items=tuple(all_included),
            omitted_items=tuple(all_omitted),
            used_tokens=used,
            max_tokens=budget.max_tokens,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(items: Sequence[ContextItem]) -> list[ContextItem]:
        """Validate input items: types, uniqueness of id / sequence."""
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            raise ContextAssemblyError("items must be a sequence of ContextItem")
        validated: list[ContextItem] = []
        seen_ids: set[str] = set()
        seen_seq: set[int] = set()
        for it in items:
            if not isinstance(it, ContextItem):
                raise ContextAssemblyError(
                    "all items must be ContextItem instances"
                )
            if it.item_id in seen_ids:
                raise ContextAssemblyError(
                    f"duplicate item_id: {it.item_id!r}"
                )
            if it.sequence_index in seen_seq:
                raise ContextAssemblyError(
                    f"duplicate sequence_index: {it.sequence_index}"
                )
            seen_ids.add(it.item_id)
            seen_seq.add(it.sequence_index)
            validated.append(it)
        return validated

    @staticmethod
    def _rank(optionals: list[ContextItem]) -> list[ContextItem]:
        """Rank optional candidates by priority DESC, recency DESC, id ASC."""
        return sorted(
            optionals,
            key=lambda it: (
                -int(it.priority),       # priority DESC
                -it.sequence_index,      # recency DESC
                it.item_id,              # stable tie-break ASC
            ),
        )
