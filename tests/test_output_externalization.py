"""Tests for ToolOutputProcessor & externalization (Phase 4 Step 2).

Covers:

* TokenEstimator validation (Section 24)
* ToolOutputRenderer rendering (Sections 27-30)
* OutputExternalizationPolicy validation (Section 25)
* Threshold boundary (Section 63)
* Inline flow (Sections 35, 64)
* Externalized flow (Sections 36, 65)
* Full readback (Section 66)
* No artifact ID for inline (Section 67)
* Unsupported output (Section 68)
* Store failure propagates (Section 69)
* Reference integrity fields (Section 70)
* Reference rendering stable (Section 71)
* Context token cost (Section 72)
* Priority preserved (Section 73)
* Reference not must-keep (Section 74)
* Omitted reference recoverable (Section 75)
* Full output not in reference context (Section 76)
* Step 1 token invariant (Section 77)

All offline, deterministic, no LLM, no network, no MCP.
"""

from __future__ import annotations

import asyncio

import pytest

from harness.context import (
    ContextAssembler,
    ContextBudget,
    ContextItem,
    ContextKind,
    ContextPriority,
)
from harness.output_externalization import (
    ArtifactReference,
    DeterministicToolOutputRenderer,
    OutputExternalizationPolicy,
    TokenEstimator,
    ToolOutputProcessor,
    ToolOutputProcessingResult,
    ToolOutputSerializationError,
)
from storage.artifact_store import (
    Artifact,
    ArtifactKind,
    ArtifactStore,
    DuplicateArtifactError,
    InMemoryArtifactStore,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeTokenEstimator:
    """Deterministic token estimator for tests.

    NOT a model tokenizer. Uses a simple character-count heuristic
    or a scripted mapping.
    """

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


class ScriptedTokenEstimator:
    """Returns scripted token counts for specific texts."""

    def __init__(self, mapping: dict[str, int] | None = None, default: int = 10):
        self.mapping = mapping or {}
        self.default = default
        self.call_count = 0

    def estimate(self, text: str) -> int:
        self.call_count += 1
        if text in self.mapping:
            return self.mapping[text]
        if not text:
            return 0
        return self.default


class FailingArtifactStore:
    """Artifact store that always raises on save."""

    async def save(self, artifact: Artifact) -> None:
        raise RuntimeError("store save failed")

    async def get(self, artifact_id: str) -> Artifact | None:
        return None


class _IdFactory:
    """Produces sequential IDs like A-001, A-002, ..."""

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


def _make_processor(
    *,
    store: ArtifactStore | None = None,
    estimator: TokenEstimator | None = None,
    policy: OutputExternalizationPolicy | None = None,
    artifact_id_factory=None,
    context_item_id_factory=None,
    clock=None,
    renderer: DeterministicToolOutputRenderer | None = None,
) -> ToolOutputProcessor:
    return ToolOutputProcessor(
        artifact_store=store or InMemoryArtifactStore(),
        token_estimator=estimator or FakeTokenEstimator(),
        output_renderer=renderer or DeterministicToolOutputRenderer(),
        externalization_policy=policy or OutputExternalizationPolicy(
            inline_token_limit=100, preview_chars=50,
        ),
        artifact_id_factory=artifact_id_factory or _IdFactory("A"),
        context_item_id_factory=context_item_id_factory or _IdFactory("CTX"),
        clock=clock or _utc_now,
    )


# ===========================================================================
# TokenEstimator validation (Section 24)
# ===========================================================================


class TestTokenEstimatorValidation:
    def test_negative_estimate_rejected(self):
        processor = _make_processor(
            estimator=ScriptedTokenEstimator(default=-1),
        )
        with pytest.raises(ValueError, match=">= 0"):
            asyncio.run(
                processor.process_success_output(
                    tool_name="t", output="hello", sequence_index=0,
                )
            )

    def test_zero_estimate_non_empty_text_rejected(self):
        processor = _make_processor(
            estimator=ScriptedTokenEstimator(default=0),
        )
        with pytest.raises(ValueError, match="estimate=0"):
            asyncio.run(
                processor.process_success_output(
                    tool_name="t", output="hello", sequence_index=0,
                )
            )


# ===========================================================================
# ToolOutputRenderer (Sections 27-30)
# ===========================================================================


class TestToolOutputRenderer:
    def test_str_passthrough(self):
        assert DeterministicToolOutputRenderer().render("hello") == "hello"

    def test_none(self):
        assert DeterministicToolOutputRenderer().render(None) == "null"

    def test_bool_true(self):
        assert DeterministicToolOutputRenderer().render(True) == "true"

    def test_bool_false(self):
        assert DeterministicToolOutputRenderer().render(False) == "false"

    def test_int(self):
        assert DeterministicToolOutputRenderer().render(42) == "42"

    def test_float(self):
        assert DeterministicToolOutputRenderer().render(3.14) == "3.14"

    def test_list(self):
        result = DeterministicToolOutputRenderer().render([1, 2, 3])
        assert result == "[1,2,3]"

    def test_dict(self):
        result = DeterministicToolOutputRenderer().render({"b": 2, "a": 1})
        assert result == '{"a":1,"b":2}'

    def test_dict_key_ordering_irrelevant(self):
        r1 = DeterministicToolOutputRenderer().render({"a": 1, "b": 2})
        r2 = DeterministicToolOutputRenderer().render({"b": 2, "a": 1})
        assert r1 == r2

    def test_nested(self):
        result = DeterministicToolOutputRenderer().render(
            {"outer": {"b": 2, "a": 1}, "list": [3, 2, 1]}
        )
        assert '"a":1' in result
        assert '"b":2' in result
        assert "[3,2,1]" in result

    def test_str_not_json_quoted(self):
        """str output should NOT be JSON-quoted."""
        result = DeterministicToolOutputRenderer().render("hello")
        assert result == "hello"
        assert result != '"hello"'

    def test_nan_rejected(self):
        with pytest.raises(ToolOutputSerializationError):
            DeterministicToolOutputRenderer().render(float("nan"))

    def test_infinity_rejected(self):
        with pytest.raises(ToolOutputSerializationError):
            DeterministicToolOutputRenderer().render(float("inf"))

    def test_custom_object_rejected(self):
        class Custom:
            pass
        with pytest.raises(ToolOutputSerializationError, match="unsupported"):
            DeterministicToolOutputRenderer().render(Custom())

    def test_nested_custom_object_rejected(self):
        class Custom:
            pass
        with pytest.raises(ToolOutputSerializationError):
            DeterministicToolOutputRenderer().render({"ok": 1, "bad": Custom()})


# ===========================================================================
# OutputExternalizationPolicy (Section 25)
# ===========================================================================


class TestExternalizationPolicy:
    def test_valid(self):
        p = OutputExternalizationPolicy(
            inline_token_limit=100, preview_chars=200,
        )
        assert p.inline_token_limit == 100
        assert p.preview_chars == 200

    def test_default_preview(self):
        p = OutputExternalizationPolicy(inline_token_limit=50)
        assert p.preview_chars == 200

    def test_zero_limit_rejected(self):
        with pytest.raises(ValueError, match="inline_token_limit"):
            OutputExternalizationPolicy(inline_token_limit=0)

    def test_negative_limit_rejected(self):
        with pytest.raises(ValueError, match="inline_token_limit"):
            OutputExternalizationPolicy(inline_token_limit=-1)

    def test_negative_preview_rejected(self):
        with pytest.raises(ValueError, match="preview_chars"):
            OutputExternalizationPolicy(
                inline_token_limit=100, preview_chars=-1,
            )

    def test_zero_preview_accepted(self):
        p = OutputExternalizationPolicy(
            inline_token_limit=100, preview_chars=0,
        )
        assert p.preview_chars == 0


# ===========================================================================
# Threshold boundary (Section 63)
# ===========================================================================


class TestThresholdBoundary:
    def test_at_limit_inline(self):
        """estimated_tokens == limit → inline."""
        processor = _make_processor(
            estimator=ScriptedTokenEstimator(default=100),
            policy=OutputExternalizationPolicy(inline_token_limit=100),
        )
        result = asyncio.run(
            processor.process_success_output(
                tool_name="t", output="x" * 400, sequence_index=0,
            )
        )
        assert result.externalized is False

    def test_below_limit_inline(self):
        """estimated_tokens == limit-1 → inline."""
        processor = _make_processor(
            estimator=ScriptedTokenEstimator(default=99),
            policy=OutputExternalizationPolicy(inline_token_limit=100),
        )
        result = asyncio.run(
            processor.process_success_output(
                tool_name="t", output="x" * 400, sequence_index=0,
            )
        )
        assert result.externalized is False

    def test_above_limit_externalized(self):
        """estimated_tokens == limit+1 → externalized."""
        processor = _make_processor(
            estimator=ScriptedTokenEstimator(default=101),
            policy=OutputExternalizationPolicy(inline_token_limit=100),
        )
        result = asyncio.run(
            processor.process_success_output(
                tool_name="t", output="x" * 400, sequence_index=0,
            )
        )
        assert result.externalized is True


# ===========================================================================
# Inline flow (Sections 35, 64)
# ===========================================================================


class TestInlineFlow:
    def test_inline_no_artifact_saved(self):
        store = InMemoryArtifactStore()
        processor = _make_processor(
            store=store,
            estimator=ScriptedTokenEstimator(default=50),
            policy=OutputExternalizationPolicy(inline_token_limit=100),
        )
        result = asyncio.run(
            processor.process_success_output(
                tool_name="read_file", output="small output",
                sequence_index=0,
            )
        )
        assert result.externalized is False
        assert result.artifact_reference is None
        # No artifact saved.
        assert asyncio.run(store.get("A-001")) is None

    def test_inline_context_item_is_recent_interaction(self):
        processor = _make_processor(
            estimator=ScriptedTokenEstimator(default=50),
        )
        result = asyncio.run(
            processor.process_success_output(
                tool_name="t", output="hello", sequence_index=0,
            )
        )
        assert result.context_item.kind is ContextKind.RECENT_INTERACTION

    def test_inline_full_output_in_content(self):
        processor = _make_processor(
            estimator=ScriptedTokenEstimator(default=50),
        )
        result = asyncio.run(
            processor.process_success_output(
                tool_name="t", output="full inline content", sequence_index=0,
            )
        )
        assert result.context_item.content == "full inline content"

    def test_inline_externalizable_true(self):
        processor = _make_processor(
            estimator=ScriptedTokenEstimator(default=50),
        )
        result = asyncio.run(
            processor.process_success_output(
                tool_name="t", output="x", sequence_index=0,
            )
        )
        assert result.context_item.externalizable is True


# ===========================================================================
# Externalized flow (Sections 36, 65)
# ===========================================================================


class TestExternalizedFlow:
    def test_externalized_artifact_saved(self):
        store = InMemoryArtifactStore()
        processor = _make_processor(
            store=store,
            estimator=ScriptedTokenEstimator(default=5000),
            policy=OutputExternalizationPolicy(inline_token_limit=100),
        )
        result = asyncio.run(
            processor.process_success_output(
                tool_name="run_tests", output="x" * 20000, sequence_index=0,
            )
        )
        assert result.externalized is True
        assert result.artifact_reference is not None
        # Artifact saved.
        stored = asyncio.run(store.get(result.artifact_reference.artifact_id))
        assert stored is not None

    def test_externalized_context_is_recoverable_reference(self):
        processor = _make_processor(
            estimator=ScriptedTokenEstimator(default=5000),
        )
        result = asyncio.run(
            processor.process_success_output(
                tool_name="t", output="x" * 200, sequence_index=0,
            )
        )
        assert result.context_item.kind is ContextKind.RECOVERABLE_REFERENCE

    def test_externalized_context_not_full_output(self):
        """ContextItem.content must NOT contain the full original output."""
        marker = "SECRET_MARKER_AT_END_" + "x" * 200
        processor = _make_processor(
            estimator=ScriptedTokenEstimator(default=5000),
            policy=OutputExternalizationPolicy(
                inline_token_limit=100, preview_chars=10,
            ),
        )
        result = asyncio.run(
            processor.process_success_output(
                tool_name="t", output=marker, sequence_index=0,
            )
        )
        # The marker is at the end, beyond the 10-char preview.
        assert "SECRET_MARKER_AT_END_" not in result.context_item.content
        # But the artifact has the full content.
        assert result.artifact_reference is not None

    def test_externalized_externalizable_false(self):
        processor = _make_processor(
            estimator=ScriptedTokenEstimator(default=5000),
        )
        result = asyncio.run(
            processor.process_success_output(
                tool_name="t", output="x" * 200, sequence_index=0,
            )
        )
        assert result.context_item.externalizable is False


# ===========================================================================
# Full readback (Section 66)
# ===========================================================================


def test_full_readback():
    """Externalized content can be fully read back from the store."""
    store = InMemoryArtifactStore()
    processor = _make_processor(
        store=store,
        estimator=ScriptedTokenEstimator(default=5000),
    )
    original = "line 1\nline 2\nline 3\n" * 100
    result = asyncio.run(
        processor.process_success_output(
            tool_name="run_tests", output=original, sequence_index=0,
        )
    )
    assert result.artifact_reference is not None
    stored = asyncio.run(store.get(result.artifact_reference.artifact_id))
    assert stored is not None
    assert stored.content == original


# ===========================================================================
# No artifact ID for inline (Section 67)
# ===========================================================================


def test_inline_does_not_consume_artifact_id():
    """Inline output does not consume an artifact ID."""
    store = InMemoryArtifactStore()
    artifact_ids = _IdFactory("A")
    processor = _make_processor(
        store=store,
        artifact_id_factory=artifact_ids,
        estimator=ScriptedTokenEstimator(default=50),  # inline
    )
    # Process inline first.
    asyncio.run(
        processor.process_success_output(
            tool_name="t", output="small", sequence_index=0,
        )
    )
    assert artifact_ids.consumed == []  # no artifact ID consumed

    # Now process a large output.
    processor_large = _make_processor(
        store=store,
        artifact_id_factory=artifact_ids,
        estimator=ScriptedTokenEstimator(default=5000),  # externalized
    )
    result = asyncio.run(
        processor_large.process_success_output(
            tool_name="t", output="x" * 200, sequence_index=1,
        )
    )
    # The large output gets A-001, not A-002.
    assert result.artifact_reference.artifact_id == "A-001"


# ===========================================================================
# Unsupported output (Section 68)
# ===========================================================================


def test_unsupported_output_raises():
    """Custom object output → ToolOutputSerializationError, no artifact."""
    store = InMemoryArtifactStore()
    processor = _make_processor(store=store)

    class Custom:
        pass

    with pytest.raises(ToolOutputSerializationError):
        asyncio.run(
            processor.process_success_output(
                tool_name="t", output=Custom(), sequence_index=0,
            )
        )
    # No artifact saved.
    assert asyncio.run(store.get("A-001")) is None


# ===========================================================================
# Store failure propagates (Section 69)
# ===========================================================================


def test_store_failure_propagates():
    """If store.save() fails, the error propagates — no dangling reference."""
    processor = _make_processor(
        store=FailingArtifactStore(),
        estimator=ScriptedTokenEstimator(default=5000),
    )
    with pytest.raises(RuntimeError, match="store save failed"):
        asyncio.run(
            processor.process_success_output(
                tool_name="t", output="x" * 200, sequence_index=0,
            )
        )


# ===========================================================================
# Reference integrity fields (Section 70)
# ===========================================================================


def test_reference_integrity_fields():
    """ArtifactReference fields match the Artifact."""
    store = InMemoryArtifactStore()
    processor = _make_processor(
        store=store,
        estimator=ScriptedTokenEstimator(default=4200),
    )
    result = asyncio.run(
        processor.process_success_output(
            tool_name="run_tests", output="test output content",
            sequence_index=0,
        )
    )
    ref = result.artifact_reference
    assert ref is not None
    artifact = asyncio.run(store.get(ref.artifact_id))
    assert artifact is not None

    assert ref.artifact_id == artifact.artifact_id
    assert ref.content_hash == artifact.content_hash
    assert ref.source_tool_name == artifact.source_tool_name
    assert ref.original_size_bytes == artifact.size_bytes
    assert ref.original_estimated_tokens == artifact.estimated_tokens


# ===========================================================================
# Reference rendering stable (Section 71)
# ===========================================================================


def test_reference_rendering_stable():
    """render() produces the same output every time."""
    processor = _make_processor(
        estimator=ScriptedTokenEstimator(default=5000),
    )
    result = asyncio.run(
        processor.process_success_output(
            tool_name="run_tests", output="x" * 200, sequence_index=0,
        )
    )
    ref = result.artifact_reference
    assert ref is not None
    text1 = ref.render()
    text2 = ref.render()
    assert text1 == text2
    assert "A-001" in text1
    assert "run_tests" in text1
    assert "SHA256:" in text1


# ===========================================================================
# Context token cost (Section 72)
# ===========================================================================


def test_context_token_cost_based_on_reference():
    """ContextItem.estimated_tokens is based on reference text, not original."""
    # Use a mapping estimator to distinguish original vs reference text.
    class MappedEstimator:
        def __init__(self):
            self.call_count = 0
            self.last_text = ""

        def estimate(self, text: str) -> int:
            self.call_count += 1
            self.last_text = text
            if text.startswith("x"):  # original output is "x" * 500
                return 5000
            return 25  # reference text

    processor = _make_processor(
        estimator=MappedEstimator(),
        policy=OutputExternalizationPolicy(inline_token_limit=100),
    )
    result = asyncio.run(
        processor.process_success_output(
            tool_name="t", output="x" * 500, sequence_index=0,
        )
    )
    assert result.externalized is True
    assert result.context_item.estimated_tokens == 25
    assert result.artifact_reference.original_estimated_tokens == 5000


# ===========================================================================
# Externalization reduces context cost (Section 47)
# ===========================================================================


def test_externalization_reduces_context_cost():
    """Context carries reference cost, not original output cost."""
    store = InMemoryArtifactStore()

    class LargeThenSmall:
        def __init__(self):
            self.calls = 0

        def estimate(self, text: str) -> int:
            self.calls += 1
            if self.calls == 1:
                return 5000  # original
            return 30  # reference

    processor = _make_processor(
        store=store,
        estimator=LargeThenSmall(),
    )
    result = asyncio.run(
        processor.process_success_output(
            tool_name="t", output="x" * 500, sequence_index=0,
        )
    )
    assert result.context_item.estimated_tokens == 30
    assert result.artifact_reference.original_estimated_tokens == 5000


# ===========================================================================
# Priority preserved (Section 73)
# ===========================================================================


class TestPriorityPreserved:
    def test_inline_priority_preserved(self):
        processor = _make_processor(
            estimator=ScriptedTokenEstimator(default=50),
        )
        result = asyncio.run(
            processor.process_success_output(
                tool_name="t", output="x", sequence_index=0,
                priority=ContextPriority.HIGH,
            )
        )
        assert result.context_item.priority is ContextPriority.HIGH

    def test_externalized_priority_preserved(self):
        processor = _make_processor(
            estimator=ScriptedTokenEstimator(default=5000),
        )
        result = asyncio.run(
            processor.process_success_output(
                tool_name="t", output="x" * 200, sequence_index=0,
                priority=ContextPriority.HIGH,
            )
        )
        assert result.context_item.priority is ContextPriority.HIGH


# ===========================================================================
# Reference not must-keep (Section 74)
# ===========================================================================


class TestNotMustKeep:
    def test_inline_not_must_keep(self):
        processor = _make_processor(
            estimator=ScriptedTokenEstimator(default=50),
        )
        result = asyncio.run(
            processor.process_success_output(
                tool_name="t", output="x", sequence_index=0,
            )
        )
        assert result.context_item.must_keep is False

    def test_externalized_not_must_keep(self):
        processor = _make_processor(
            estimator=ScriptedTokenEstimator(default=5000),
        )
        result = asyncio.run(
            processor.process_success_output(
                tool_name="t", output="x" * 200, sequence_index=0,
            )
        )
        assert result.context_item.must_keep is False


# ===========================================================================
# Omitted reference recoverable (Section 75)
# ===========================================================================


def test_omitted_reference_recoverable():
    """Even if ContextAssembler omits the reference, the artifact is
    still fully readable from the store."""
    store = InMemoryArtifactStore()
    processor = _make_processor(
        store=store,
        estimator=ScriptedTokenEstimator(default=5000),
        policy=OutputExternalizationPolicy(inline_token_limit=100),
    )
    original = "important large output " * 100
    result = asyncio.run(
        processor.process_success_output(
            tool_name="run_tests", output=original, sequence_index=5,
        )
    )
    ref_item = result.context_item
    artifact_id = result.artifact_reference.artifact_id

    # Now assemble with a very tight budget so the reference is omitted.
    critical = ContextItem(
        item_id="goal",
        kind=ContextKind.CRITICAL_STATE,
        content="task goal",
        estimated_tokens=95,
        sequence_index=0,
        must_keep=True,
    )
    assembler = ContextAssembler()
    assembled = assembler.assemble(
        [critical, ref_item],
        ContextBudget(max_tokens=100),
    )
    # The reference must be omitted (95 + ref_tokens > 100).
    assert ref_item.item_id not in [it.item_id for it in assembled.included_items]
    assert ref_item.item_id in [it.item_id for it in assembled.omitted_items]

    # But the full content is still recoverable!
    stored = asyncio.run(store.get(artifact_id))
    assert stored is not None
    assert stored.content == original


# ===========================================================================
# Full output not in reference context (Section 76)
# ===========================================================================


def test_full_output_not_in_reference_context():
    """The full original output must not appear in the reference context."""
    marker = "UNIQUE_END_MARKER_" + "z" * 300
    processor = _make_processor(
        estimator=ScriptedTokenEstimator(default=5000),
        policy=OutputExternalizationPolicy(
            inline_token_limit=100, preview_chars=20,
        ),
    )
    result = asyncio.run(
        processor.process_success_output(
            tool_name="t", output=marker, sequence_index=0,
        )
    )
    # The marker is at the beginning but the preview is only 20 chars.
    # The full marker string "UNIQUE_END_MARKER_zzz..." is > 20 chars.
    # The reference content should contain the preview but not the full marker.
    assert "UNIQUE_END_MARKER_" in result.context_item.content  # in preview
    # But the "z" * 300 tail should NOT be in the context.
    assert "z" * 100 not in result.context_item.content

    # The artifact has the full content.
    assert result.artifact_reference is not None


# ===========================================================================
# Preview test (Section 44)
# ===========================================================================


def test_preview_is_exact_prefix():
    """Preview is content[:preview_chars], no ellipsis added."""
    processor = _make_processor(
        estimator=ScriptedTokenEstimator(default=5000),
        policy=OutputExternalizationPolicy(
            inline_token_limit=100, preview_chars=5,
        ),
    )
    result = asyncio.run(
        processor.process_success_output(
            tool_name="t", output="abcdefgh", sequence_index=0,
        )
    )
    ref = result.artifact_reference
    assert ref is not None
    assert ref.preview == "abcde"  # exact prefix, no "..."


def test_preview_zero_empty():
    """preview_chars=0 → empty preview."""
    processor = _make_processor(
        estimator=ScriptedTokenEstimator(default=5000),
        policy=OutputExternalizationPolicy(
            inline_token_limit=100, preview_chars=0,
        ),
    )
    result = asyncio.run(
        processor.process_success_output(
            tool_name="t", output="abcdefgh", sequence_index=0,
        )
    )
    ref = result.artifact_reference
    assert ref is not None
    assert ref.preview == ""


# ===========================================================================
# Summary test (Section 45)
# ===========================================================================


def test_summary_deterministic():
    """Same inputs → same summary."""
    processor = _make_processor(
        estimator=ScriptedTokenEstimator(default=4200),
    )
    result1 = asyncio.run(
        processor.process_success_output(
            tool_name="run_tests", output="x" * 200, sequence_index=0,
        )
    )
    # Build a second processor with same config.
    processor2 = _make_processor(
        estimator=ScriptedTokenEstimator(default=4200),
    )
    result2 = asyncio.run(
        processor2.process_success_output(
            tool_name="run_tests", output="x" * 200, sequence_index=0,
        )
    )
    assert result1.artifact_reference.summary == result2.artifact_reference.summary
    assert "run_tests" in result1.artifact_reference.summary
    assert "4200" in result1.artifact_reference.summary


# ===========================================================================
# Source tool name preserved (Section 40)
# ===========================================================================


def test_source_tool_name_preserved():
    processor = _make_processor(
        estimator=ScriptedTokenEstimator(default=5000),
    )
    result = asyncio.run(
        processor.process_success_output(
            tool_name="custom_tool", output="x" * 200, sequence_index=0,
        )
    )
    assert result.artifact_reference.source_tool_name == "custom_tool"


# ===========================================================================
# Unicode integrity (Section 43)
# ===========================================================================


def test_unicode_artifact_integrity():
    """Unicode content is correctly hashed and sized."""
    store = InMemoryArtifactStore()
    processor = _make_processor(
        store=store,
        estimator=ScriptedTokenEstimator(default=5000),
    )
    unicode_content = "测试输出 🚀"
    result = asyncio.run(
        processor.process_success_output(
            tool_name="t", output=unicode_content, sequence_index=0,
        )
    )
    stored = asyncio.run(store.get(result.artifact_reference.artifact_id))
    assert stored is not None
    assert stored.content == unicode_content
    assert stored.size_bytes == len(unicode_content.encode("utf-8"))


# ===========================================================================
# ContextAssembler integration (Section 51)
# ===========================================================================


def test_context_assembler_unaware_of_artifact_store():
    """ContextAssembler only sees ContextItems, not ArtifactStore."""
    store = InMemoryArtifactStore()
    processor = _make_processor(
        store=store,
        estimator=ScriptedTokenEstimator(default=5000),
    )
    result = asyncio.run(
        processor.process_success_output(
            tool_name="run_tests", output="x" * 200, sequence_index=2,
        )
    )

    items = [
        ContextItem(
            item_id="goal",
            kind=ContextKind.CRITICAL_STATE,
            content="task goal",
            estimated_tokens=20,
            sequence_index=0,
            must_keep=True,
        ),
        ContextItem(
            item_id="obs",
            kind=ContextKind.RECENT_INTERACTION,
            content="small obs",
            estimated_tokens=10,
            sequence_index=1,
        ),
        result.context_item,
    ]
    assembled = ContextAssembler().assemble(items, ContextBudget(max_tokens=50))
    # goal(20) + obs(10) = 30; remaining 20.
    # reference item has some token cost; if it fits, included.
    # Either way, the assembler works with ContextItems only.
    assert assembled.used_tokens <= 50
    # The full output is NOT in any included item's content.
    for item in assembled.included_items:
        assert "x" * 200 not in item.content
