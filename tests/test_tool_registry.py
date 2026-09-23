"""Tests for the tool registry (Phase 2 Step 1)."""

from __future__ import annotations

import pytest

from tests.fake_tools import EchoTool
from tools import (
    DuplicateToolError,
    ToolRegistry,
    ToolSideEffect,
    ToolSpec,
)

_ECHO_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


def _echo_spec(name: str = "echo") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="echo the text argument",
        input_schema=_ECHO_SCHEMA,
        required_permissions=frozenset(),
        timeout_seconds=1.0,
        side_effect=ToolSideEffect.READ_ONLY,
    )


# ---------------------------------------------------------------------------
# register + get
# ---------------------------------------------------------------------------


def test_register_and_get():
    registry = ToolRegistry()
    handler = EchoTool()
    registry.register(_echo_spec(), handler)

    got = registry.get("echo")
    assert got is not None
    assert got.spec.name == "echo"
    assert got.handler is handler


# ---------------------------------------------------------------------------
# contains
# ---------------------------------------------------------------------------


def test_contains():
    registry = ToolRegistry()
    assert not registry.contains("echo")
    registry.register(_echo_spec(), EchoTool())
    assert registry.contains("echo")
    assert not registry.contains("nope")


# ---------------------------------------------------------------------------
# duplicate name rejected
# ---------------------------------------------------------------------------


def test_duplicate_registration_rejected():
    registry = ToolRegistry()
    registry.register(_echo_spec(), EchoTool())
    with pytest.raises(DuplicateToolError, match="already registered"):
        registry.register(_echo_spec(), EchoTool())


# ---------------------------------------------------------------------------
# different names coexist
# ---------------------------------------------------------------------------


def test_different_names_coexist():
    registry = ToolRegistry()
    registry.register(_echo_spec("echo"), EchoTool())
    registry.register(_echo_spec("echo2"), EchoTool())
    assert registry.contains("echo")
    assert registry.contains("echo2")
    assert registry.get("echo").spec.name == "echo"
    assert registry.get("echo2").spec.name == "echo2"


# ---------------------------------------------------------------------------
# missing lookup returns None
# ---------------------------------------------------------------------------


def test_missing_lookup_returns_none():
    registry = ToolRegistry()
    assert registry.get("missing") is None


# ---------------------------------------------------------------------------
# list_specs
# ---------------------------------------------------------------------------


def test_list_specs_snapshot():
    registry = ToolRegistry()
    registry.register(_echo_spec("a"), EchoTool())
    registry.register(_echo_spec("b"), EchoTool())
    specs = registry.list_specs()
    assert [s.name for s in specs] == ["a", "b"]
    # Mutating the returned list does not affect the registry.
    specs.clear()
    assert registry.contains("a") and registry.contains("b")


# ---------------------------------------------------------------------------
# Phase 2 Step 2 — Registry metadata snapshot semantics
# ---------------------------------------------------------------------------


def test_register_snapshots_input_schema():
    """Mutating the original schema after register does not affect the registry."""
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    spec = ToolSpec(
        name="echo",
        description="echo",
        input_schema=schema,
        required_permissions=frozenset(),
        timeout_seconds=1.0,
        side_effect=ToolSideEffect.READ_ONLY,
    )
    registry = ToolRegistry()
    registry.register(spec, EchoTool())

    # Mutate the original schema dict.
    schema["properties"].clear()
    schema["required"].clear()

    # Registry's internal spec is unchanged.
    registered = registry.get("echo")
    assert registered is not None
    assert "text" in registered.spec.input_schema["properties"]
    assert registered.spec.input_schema["required"] == ["text"]


def test_retrieved_spec_cannot_corrupt_registry():
    """Mutating a retrieved spec's input_schema does not corrupt the registry."""
    registry = ToolRegistry()
    registry.register(_echo_spec(), EchoTool())

    registered = registry.get("echo")
    assert registered is not None
    # Mutate the retrieved spec's schema.
    registered.spec.input_schema["properties"].clear()

    # A subsequent get still sees the original schema.
    again = registry.get("echo")
    assert again is not None
    assert "text" in again.spec.input_schema["properties"]


def test_list_specs_snapshots_input_schema():
    """Specs returned by list_specs are isolated from the registry."""
    registry = ToolRegistry()
    registry.register(_echo_spec(), EchoTool())

    specs = registry.list_specs()
    specs[0].input_schema["properties"].clear()

    again = registry.list_specs()
    assert "text" in again[0].input_schema["properties"]


def test_handler_identity_preserved_across_get():
    """Handler object identity is preserved (only metadata is snapshotted)."""
    registry = ToolRegistry()
    handler = EchoTool()
    registry.register(_echo_spec(), handler)

    registered = registry.get("echo")
    assert registered is not None
    assert registered.handler is handler
