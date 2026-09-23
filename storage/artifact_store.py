"""Artifact store for large tool output externalization (Phase 4 Step 2).

The ``ArtifactStore`` protocol defines the async surface for storing
and retrieving full tool outputs that are too large for the model
context. ``InMemoryArtifactStore`` is a deterministic, offline
implementation suitable for tests and local development.

Key principles:

* **Externalized != Deleted.** An artifact's full content is preserved
  even when only a small reference enters the model context.
* **No deduplication.** Two identical outputs produce two artifacts
  with different ``artifact_id``s. Hash-based dedup is a future concern.
* **No compression.** Artifacts store the original content verbatim.
* **In-memory only.** Artifacts do not survive process crashes.

The store does NOT integrate with ``HarnessRuntime``,
``AgentActionController``, or ``ToolRuntime``. It is an independent
storage layer that the ``ToolOutputProcessor`` uses.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DuplicateArtifactError(Exception):
    """Raised when an artifact_id already exists in the store.

    The store never silently overwrites historical evidence.
    """


# ---------------------------------------------------------------------------
# Artifact kind
# ---------------------------------------------------------------------------


class ArtifactKind(Enum):
    """What kind of content an artifact holds.

    Phase 4 Step 2 only needs ``TOOL_OUTPUT`` for large tool results.
    The taxonomy can grow later without breaking existing artifacts.
    """

    TOOL_OUTPUT = "tool_output"


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Artifact:
    """Immutable record of a fully externalized tool output.

    All derived fields (``content_hash``, ``size_bytes``) are computed
    by ``Artifact.create()``, NOT trusted from the caller. This ensures
    integrity and consistency.

    Attributes:
        artifact_id: stable unique identifier (non-empty str).
        kind: what this artifact is (see ``ArtifactKind``).
        content: the complete original text content.
        content_hash: SHA-256 hex digest of ``content.encode("utf-8")``.
        size_bytes: ``len(content.encode("utf-8"))`` (NOT ``len(content)``).
        estimated_tokens: caller-supplied token estimate for the full
            content. NOT a model tokenizer result.
        created_at: timezone-aware datetime when the artifact was created.
        source_tool_name: the tool that produced this output (non-empty).
    """

    artifact_id: str
    kind: ArtifactKind
    content: str
    content_hash: str
    size_bytes: int
    estimated_tokens: int
    created_at: datetime
    source_tool_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not self.artifact_id:
            raise ValueError("artifact_id must be a non-empty str")
        if not isinstance(self.kind, ArtifactKind):
            raise ValueError("kind must be an ArtifactKind")
        if not isinstance(self.content, str):
            raise ValueError("content must be a str")
        if not isinstance(self.content_hash, str) or not self.content_hash:
            raise ValueError("content_hash must be a non-empty str")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise ValueError("size_bytes must be an int >= 0")
        if (
            not isinstance(self.estimated_tokens, int)
            or isinstance(self.estimated_tokens, bool)
            or self.estimated_tokens < 0
        ):
            raise ValueError("estimated_tokens must be an int >= 0")
        # Phase 5 Step 1 invariant (Section 68): non-empty content must not
        # claim zero tokens, consistent with ToolOutputProcessor and
        # ContextItem semantics. Empty content may legitimately be 0.
        if self.content and self.estimated_tokens == 0:
            raise ValueError(
                "estimated_tokens must be >= 1 for non-empty content"
            )
        if not isinstance(self.created_at, datetime):
            raise ValueError("created_at must be a datetime")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if (
            not isinstance(self.source_tool_name, str)
            or not self.source_tool_name
        ):
            raise ValueError("source_tool_name must be a non-empty str")

    @classmethod
    def create(
        cls,
        *,
        artifact_id: str,
        kind: ArtifactKind,
        content: str,
        estimated_tokens: int,
        created_at: datetime,
        source_tool_name: str,
    ) -> "Artifact":
        """Create an artifact, computing ``content_hash`` and ``size_bytes``.

        Callers must NOT supply derived fields directly. This factory
        computes them from ``content`` to ensure integrity.
        """
        encoded = content.encode("utf-8")
        return cls(
            artifact_id=artifact_id,
            kind=kind,
            content=content,
            content_hash=hashlib.sha256(encoded).hexdigest(),
            size_bytes=len(encoded),
            estimated_tokens=estimated_tokens,
            created_at=created_at,
            source_tool_name=source_tool_name,
        )


# ---------------------------------------------------------------------------
# Artifact store protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ArtifactStore(Protocol):
    """Async storage contract for externalized artifacts."""

    async def save(self, artifact: Artifact) -> None:
        """Persist an artifact. Rejects duplicate ``artifact_id``."""
        ...

    async def get(self, artifact_id: str) -> Artifact | None:
        """Return the artifact for ``artifact_id`` or ``None`` if missing."""
        ...


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------


class InMemoryArtifactStore:
    """Deterministic in-memory ``ArtifactStore``.

    Artifacts are immutable frozen dataclasses with only scalar fields,
    so no deep-copy is needed — the stored object cannot be mutated by
    callers.

    Lifetime: current Python process only. Artifacts do NOT survive
    process crashes.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, Artifact] = {}

    async def save(self, artifact: Artifact) -> None:
        """Store ``artifact``. Raises ``DuplicateArtifactError`` on conflict."""
        if not isinstance(artifact, Artifact):
            raise ValueError("artifact must be an Artifact")
        if artifact.artifact_id in self._by_id:
            raise DuplicateArtifactError(
                f"artifact_id {artifact.artifact_id!r} already exists"
            )
        self._by_id[artifact.artifact_id] = artifact

    async def get(self, artifact_id: str) -> Artifact | None:
        """Return the artifact for ``artifact_id`` or ``None`` if missing."""
        return self._by_id.get(artifact_id)
