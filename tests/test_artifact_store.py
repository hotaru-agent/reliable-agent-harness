"""Tests for the Artifact model and InMemoryArtifactStore (Phase 4 Step 2).

Covers:

* Artifact model invariants (Section 61)
* Artifact.create derived fields (Section 14)
* SHA-256 content hash (Section 13)
* UTF-8 size_bytes (Sections 13, 43)
* Unicode content (Section 43)
* Timezone-aware created_at (Section 13)
* Frozen semantics (Section 20)
* InMemoryArtifactStore save/get (Section 62)
* Missing artifact → None (Section 62)
* Duplicate artifact ID rejected (Sections 19, 21)
* Multiple artifacts isolated (Section 62)

All offline, deterministic, no LLM, no network, no MCP.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone, timedelta

import pytest

from storage.artifact_store import (
    Artifact,
    ArtifactKind,
    ArtifactStore,
    DuplicateArtifactError,
    InMemoryArtifactStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_artifact(
    artifact_id: str = "A-001",
    content: str = "hello world",
    estimated_tokens: int = 100,
    source_tool_name: str = "run_tests",
    kind: ArtifactKind = ArtifactKind.TOOL_OUTPUT,
    created_at: datetime | None = None,
) -> Artifact:
    return Artifact.create(
        artifact_id=artifact_id,
        kind=kind,
        content=content,
        estimated_tokens=estimated_tokens,
        created_at=created_at or _utc_now(),
        source_tool_name=source_tool_name,
    )


# ===========================================================================
# Artifact model invariants (Section 61)
# ===========================================================================


class TestArtifactModel:
    def test_valid_creation(self):
        a = _make_artifact()
        assert a.artifact_id == "A-001"
        assert a.kind is ArtifactKind.TOOL_OUTPUT
        assert a.content == "hello world"
        assert a.source_tool_name == "run_tests"
        assert a.estimated_tokens == 100

    def test_empty_artifact_id_rejected(self):
        with pytest.raises(ValueError, match="artifact_id"):
            _make_artifact(artifact_id="")

    def test_non_str_artifact_id_rejected(self):
        with pytest.raises(ValueError, match="artifact_id"):
            Artifact.create(
                artifact_id=123,  # type: ignore[arg-type]
                kind=ArtifactKind.TOOL_OUTPUT,
                content="x",
                estimated_tokens=1,
                created_at=_utc_now(),
                source_tool_name="t",
            )

    def test_empty_source_tool_name_rejected(self):
        with pytest.raises(ValueError, match="source_tool_name"):
            _make_artifact(source_tool_name="")

    def test_non_str_source_tool_name_rejected(self):
        with pytest.raises(ValueError, match="source_tool_name"):
            Artifact.create(
                artifact_id="A-001",
                kind=ArtifactKind.TOOL_OUTPUT,
                content="x",
                estimated_tokens=1,
                created_at=_utc_now(),
                source_tool_name=123,  # type: ignore[arg-type]
            )

    def test_negative_estimated_tokens_rejected(self):
        with pytest.raises(ValueError, match="estimated_tokens"):
            _make_artifact(estimated_tokens=-1)

    def test_bool_estimated_tokens_rejected(self):
        with pytest.raises(ValueError, match="estimated_tokens"):
            _make_artifact(estimated_tokens=True)  # type: ignore[arg-type]

    def test_zero_estimated_tokens_accepted(self):
        """Zero-token artifact with empty content is valid."""
        a = Artifact.create(
            artifact_id="A-001",
            kind=ArtifactKind.TOOL_OUTPUT,
            content="",
            estimated_tokens=0,
            created_at=_utc_now(),
            source_tool_name="t",
        )
        assert a.estimated_tokens == 0

    def test_non_enum_kind_rejected(self):
        with pytest.raises(ValueError, match="kind"):
            Artifact.create(
                artifact_id="A-001",
                kind="tool_output",  # type: ignore[arg-type]
                content="x",
                estimated_tokens=1,
                created_at=_utc_now(),
                source_tool_name="t",
            )

    # --- created_at timezone (Section 13) ---

    def test_timezone_aware_created_at_accepted(self):
        a = _make_artifact()
        assert a.created_at.tzinfo is not None

    def test_naive_created_at_rejected(self):
        naive = datetime(2024, 1, 1, 12, 0, 0)  # no tzinfo
        with pytest.raises(ValueError, match="timezone-aware"):
            Artifact.create(
                artifact_id="A-001",
                kind=ArtifactKind.TOOL_OUTPUT,
                content="x",
                estimated_tokens=1,
                created_at=naive,
                source_tool_name="t",
            )

    def test_non_utc_timezone_accepted(self):
        """Any timezone-aware datetime is accepted, not just UTC."""
        tz = timezone(timedelta(hours=8))
        a = Artifact.create(
            artifact_id="A-001",
            kind=ArtifactKind.TOOL_OUTPUT,
            content="x",
            estimated_tokens=1,
            created_at=datetime(2024, 1, 1, 20, 0, 0, tzinfo=tz),
            source_tool_name="t",
        )
        assert a.created_at.tzinfo is not None

    # --- frozen semantics (Section 20) ---

    def test_frozen(self):
        a = _make_artifact()
        with pytest.raises(Exception):
            a.artifact_id = "other"  # type: ignore[misc]

    # --- SHA-256 hash (Section 13) ---

    def test_content_hash_is_sha256(self):
        content = "hello world"
        a = _make_artifact(content=content)
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert a.content_hash == expected

    def test_content_hash_deterministic(self):
        a1 = _make_artifact(content="test")
        a2 = _make_artifact(artifact_id="A-002", content="test")
        assert a1.content_hash == a2.content_hash

    def test_different_content_different_hash(self):
        a1 = _make_artifact(content="aaa")
        a2 = _make_artifact(artifact_id="A-002", content="bbb")
        assert a1.content_hash != a2.content_hash

    # --- UTF-8 size_bytes (Sections 13, 43) ---

    def test_size_bytes_ascii(self):
        a = _make_artifact(content="hello")
        assert a.size_bytes == 5

    def test_size_bytes_unicode(self):
        """UTF-8 byte size != character count for unicode."""
        content = "测试输出 🚀"
        a = _make_artifact(content=content)
        expected = len(content.encode("utf-8"))
        assert a.size_bytes == expected
        # Verify it's NOT just len(content).
        assert a.size_bytes != len(content)

    def test_size_bytes_empty_content(self):
        a = _make_artifact(content="")
        assert a.size_bytes == 0

    def test_size_bytes_japanese(self):
        content = "テスト出力"
        a = _make_artifact(content=content)
        assert a.size_bytes == len(content.encode("utf-8"))

    # --- Artifact.create computes derived fields (Section 14) ---

    def test_create_does_not_accept_content_hash(self):
        """Artifact.create does not accept content_hash — it computes it."""
        # The create() signature does not include content_hash.
        import inspect
        sig = inspect.signature(Artifact.create)
        assert "content_hash" not in sig.parameters
        assert "size_bytes" not in sig.parameters


# ===========================================================================
# InMemoryArtifactStore (Section 62)
# ===========================================================================


class TestInMemoryArtifactStore:
    def test_save_and_get(self):
        store = InMemoryArtifactStore()
        a = _make_artifact()
        asyncio.run(store.save(a))
        retrieved = asyncio.run(store.get("A-001"))
        assert retrieved is not None
        assert retrieved.artifact_id == "A-001"
        assert retrieved.content == "hello world"

    def test_missing_returns_none(self):
        store = InMemoryArtifactStore()
        result = asyncio.run(store.get("nonexistent"))
        assert result is None

    def test_duplicate_id_rejected(self):
        store = InMemoryArtifactStore()
        a1 = _make_artifact(artifact_id="A-001")
        a2 = _make_artifact(artifact_id="A-001", content="different")
        asyncio.run(store.save(a1))
        with pytest.raises(DuplicateArtifactError, match="A-001"):
            asyncio.run(store.save(a2))

    def test_multiple_artifacts_isolated(self):
        store = InMemoryArtifactStore()
        a1 = _make_artifact(artifact_id="A-001", content="first")
        a2 = _make_artifact(artifact_id="A-002", content="second")
        asyncio.run(store.save(a1))
        asyncio.run(store.save(a2))
        r1 = asyncio.run(store.get("A-001"))
        r2 = asyncio.run(store.get("A-002"))
        assert r1 is not None and r2 is not None
        assert r1.content == "first"
        assert r2.content == "second"

    def test_save_non_artifact_rejected(self):
        store = InMemoryArtifactStore()
        with pytest.raises(ValueError, match="Artifact"):
            asyncio.run(store.save("not an artifact"))  # type: ignore[arg-type]

    def test_artifact_immutable_in_store(self):
        """Frozen dataclass cannot be mutated after save."""
        store = InMemoryArtifactStore()
        a = _make_artifact()
        asyncio.run(store.save(a))
        retrieved = asyncio.run(store.get("A-001"))
        assert retrieved is not None
        with pytest.raises(Exception):
            retrieved.content = "tampered"  # type: ignore[misc]

    def test_implements_protocol(self):
        store = InMemoryArtifactStore()
        assert isinstance(store, ArtifactStore)


# ===========================================================================
# No deduplication (Section 58)
# ===========================================================================


def test_no_deduplication():
    """Two identical outputs produce two artifacts with different IDs."""
    store = InMemoryArtifactStore()
    a1 = _make_artifact(artifact_id="A-001", content="same")
    a2 = _make_artifact(artifact_id="A-002", content="same")
    asyncio.run(store.save(a1))
    asyncio.run(store.save(a2))
    r1 = asyncio.run(store.get("A-001"))
    r2 = asyncio.run(store.get("A-002"))
    assert r1 is not None and r2 is not None
    assert r1.content_hash == r2.content_hash  # same hash
    assert r1.artifact_id != r2.artifact_id  # different IDs
