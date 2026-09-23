"""Tool registry (Phase 2 Step 1 + Step 2).

A minimal, explicit registry mapping tool name -> ``RegisteredTool``.

Behaviour:

* duplicate registration -> ``DuplicateToolError`` (never silent overwrite)
* missing lookup -> returns ``None`` (a lookup miss, not an exception)
* ``contains(name)`` -> bool

Snapshot semantics (Phase 2 Step 2):

* On ``register``, the spec's mutable metadata (``input_schema``) is
  deep-copied so later mutation of the caller's original dict cannot
  corrupt the registry.
* On ``get`` / ``list_specs``, the returned spec has a deep-copied
  ``input_schema`` so external mutation of a retrieved spec cannot
  corrupt the registry's internal copy.

Handler identity is preserved (the same handler object is returned) —
only metadata is snapshotted.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Optional

from tools.errors import DuplicateToolError
from tools.models import RegisteredTool, ToolHandler, ToolSpec


def _freeze_spec(spec: ToolSpec) -> ToolSpec:
    """Return a ToolSpec with a deep-copied ``input_schema``."""
    return dataclasses.replace(
        spec, input_schema=copy.deepcopy(spec.input_schema)
    )


class ToolRegistry:
    """In-memory registry of ``RegisteredTool`` keyed by tool name."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        """Register a tool.

        Raises ``DuplicateToolError`` if ``spec.name`` is already
        registered. The spec's ``input_schema`` is deep-copied so the
        caller's original dict cannot later corrupt the registry.
        """
        if spec.name in self._tools:
            raise DuplicateToolError(f"tool {spec.name!r} is already registered")
        self._tools[spec.name] = RegisteredTool(
            spec=_freeze_spec(spec), handler=handler
        )

    def get(self, name: str) -> Optional[RegisteredTool]:
        """Return the registered tool for ``name`` or ``None`` if missing.

        The returned spec has a deep-copied ``input_schema`` so mutating
        it cannot corrupt the registry. Handler identity is preserved.
        """
        rt = self._tools.get(name)
        if rt is None:
            return None
        return RegisteredTool(
            spec=_freeze_spec(rt.spec),
            handler=rt.handler,
        )

    def contains(self, name: str) -> bool:
        """Return True if a tool with ``name`` is registered."""
        return name in self._tools

    def list_specs(self) -> list[ToolSpec]:
        """Return all registered specs (snapshot, insertion order).

        Each spec has a deep-copied ``input_schema``. Not a discovery API.
        """
        return [_freeze_spec(rt.spec) for rt in self._tools.values()]
