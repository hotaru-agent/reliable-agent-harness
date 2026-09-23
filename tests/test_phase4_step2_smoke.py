"""Smoke test for Phase 4 Step 2 — artifact store & output externalization.

Simulates a long-horizon scenario:

    run_tests Tool returns 5000 estimated tokens of pytest output.
    inline_token_limit = 200.

Flow:
    large pytest output
        ↓
    ToolOutputProcessor
        ↓
    Artifact A-001 (full output preserved)
        ↓
    ArtifactReference
        ↓
    RECOVERABLE_REFERENCE ContextItem
        ↓
    ContextAssembler with:
        - Task goal (CRITICAL, must_keep)
        - constraints (CRITICAL, must_keep)
        - reference (NORMAL)
        - recent small observation (NORMAL)

Proves:
    used_tokens <= budget
    full pytest output NOT in assembled context
    reference can be selected / omitted
    regardless of omission, A-001 full content is readable

Fully offline, deterministic, no LLM, no network, no MCP.
"""

from __future__ import annotations

import asyncio

from harness.context import (
    ContextAssembler,
    ContextBudget,
    ContextItem,
    ContextKind,
    ContextPriority,
)
from harness.output_externalization import (
    DeterministicToolOutputRenderer,
    OutputExternalizationPolicy,
    ToolOutputProcessor,
)
from storage.artifact_store import InMemoryArtifactStore


class CharDiv4Estimator:
    """Simple deterministic estimator: len(text) // 4, min 1 for non-empty."""

    def __init__(self):
        self.call_count = 0

    def estimate(self, text: str) -> int:
        self.call_count += 1
        if not text:
            return 0
        return max(1, len(text) // 4)


class IdFactory:
    def __init__(self, prefix: str, start: int = 1):
        self.prefix = prefix
        self.counter = start

    def __call__(self) -> str:
        id_str = f"{self.prefix}-{self.counter:03d}"
        self.counter += 1
        return id_str


def _utc_now():
    from datetime import datetime, timezone
    return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_phase4_step2_smoke():
    store = InMemoryArtifactStore()
    estimator = CharDiv4Estimator()
    processor = ToolOutputProcessor(
        artifact_store=store,
        token_estimator=estimator,
        output_renderer=DeterministicToolOutputRenderer(),
        externalization_policy=OutputExternalizationPolicy(
            inline_token_limit=200,
            preview_chars=100,
        ),
        artifact_id_factory=IdFactory("A"),
        context_item_id_factory=IdFactory("CTX"),
        clock=_utc_now,
    )

    # Simulate run_tests returning a large pytest output.
    large_output = "FAILED test_a\nFAILED test_b\n" * 500  # ~10000 chars
    result = asyncio.run(
        processor.process_success_output(
            tool_name="run_tests",
            output=large_output,
            sequence_index=2,
            priority=ContextPriority.NORMAL,
        )
    )

    # --- Externalization happened ---
    assert result.externalized is True
    assert result.artifact_reference is not None
    assert result.artifact_reference.artifact_id == "A-001"
    assert result.context_item.kind is ContextKind.RECOVERABLE_REFERENCE

    # --- Full output NOT in context ---
    assert large_output not in result.context_item.content
    assert "FAILED test_a\nFAILED test_b\n" * 10 not in result.context_item.content

    # --- Context token cost is small (reference, not full) ---
    assert result.context_item.estimated_tokens < 200  # reference is small

    # --- Build context for assembly ---
    items = [
        ContextItem(
            item_id="goal",
            kind=ContextKind.CRITICAL_STATE,
            content="Fix all failing tests in module X.",
            estimated_tokens=20,
            sequence_index=0,
            must_keep=True,
        ),
        ContextItem(
            item_id="constraints",
            kind=ContextKind.CRITICAL_STATE,
            content="Do not modify public API.",
            estimated_tokens=10,
            sequence_index=1,
            must_keep=True,
        ),
        result.context_item,  # the reference, sequence_index=2
        ContextItem(
            item_id="recent_obs",
            kind=ContextKind.RECENT_INTERACTION,
            content="Read file src/main.py: 42 lines.",
            estimated_tokens=15,
            sequence_index=3,
            priority=ContextPriority.NORMAL,
        ),
    ]

    # --- Assembly with generous budget (reference included) ---
    assembled = ContextAssembler().assemble(items, ContextBudget(max_tokens=200))
    assert assembled.used_tokens <= 200

    included_ids = [it.item_id for it in assembled.included_items]
    assert "goal" in included_ids
    assert "constraints" in included_ids
    assert result.context_item.item_id in included_ids  # reference included

    # Full output still NOT in assembled context.
    for item in assembled.included_items:
        assert large_output not in item.content

    # --- Readback works ---
    stored = asyncio.run(store.get("A-001"))
    assert stored is not None
    assert stored.content == large_output
    assert stored.source_tool_name == "run_tests"

    # --- Assembly with tight budget (reference omitted) ---
    assembled_tight = ContextAssembler().assemble(
        items, ContextBudget(max_tokens=35),  # only fits goal + constraints
    )
    assert assembled_tight.used_tokens <= 35
    omitted_ids = [it.item_id for it in assembled_tight.omitted_items]
    assert result.context_item.item_id in omitted_ids  # reference omitted

    # --- Omitted reference still recoverable! ---
    stored_again = asyncio.run(store.get("A-001"))
    assert stored_again is not None
    assert stored_again.content == large_output
