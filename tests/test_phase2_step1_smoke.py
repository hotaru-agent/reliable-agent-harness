"""Smoke test for Phase 2 Step 1 — unified tool runtime boundary.

Builds a single ``ToolRuntime`` over a ``ToolRegistry`` and exercises
the success + every structured-failure path through the same boundary,
proving the runtime gives consistent structured results regardless of
the failure reason, while executing the handler at most once.

Fully offline, deterministic, no LLM, no network, no MCP.
"""

from __future__ import annotations

import asyncio

from tests.fake_tools import FieldEchoTool, SlowTool
from tools import (
    ToolCall,
    ToolErrorType,
    ToolExecutionContext,
    ToolRegistry,
    ToolRuntime,
    ToolSideEffect,
    ToolSpec,
)

_READ_REPO_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}


def test_phase2_step1_smoke():
    registry = ToolRegistry()

    # A permission-protected, read-only tool with a short timeout.
    read_repo = ToolSpec(
        name="read_repo",
        description="read a repo path",
        input_schema=_READ_REPO_SCHEMA,
        required_permissions=frozenset({"repo.read"}),
        timeout_seconds=0.05,
        side_effect=ToolSideEffect.READ_ONLY,
    )
    read_handler = FieldEchoTool(field="path")  # echoes arguments["path"]
    registry.register(read_repo, read_handler)

    # A slow tool that will exceed its timeout.
    slow_spec = ToolSpec(
        name="slow_repo",
        description="slow tool",
        input_schema={"type": "object"},
        required_permissions=frozenset(),
        timeout_seconds=0.05,
        side_effect=ToolSideEffect.READ_ONLY,
    )
    slow_handler = SlowTool(delay=1.0)
    registry.register(slow_spec, slow_handler)

    runtime = ToolRuntime(registry=registry)

    # 1. Valid call -> success.
    ok = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="read_repo", arguments={"path": "src/main.py"}),
            ToolExecutionContext(granted_permissions=frozenset({"repo.read"})),
        )
    )
    assert ok.success is True
    assert ok.output == "src/main.py"
    assert ok.error is None
    assert read_handler.call_count == 1

    # 2. Invalid args -> VALIDATION (handler not called for this attempt).
    bad_args = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="read_repo", arguments={}),  # missing path
            ToolExecutionContext(granted_permissions=frozenset({"repo.read"})),
        )
    )
    assert bad_args.success is False
    assert bad_args.error.error_type is ToolErrorType.VALIDATION
    assert bad_args.error.retryable is False

    # 3. Missing permission -> PERMISSION.
    no_perm = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="read_repo", arguments={"path": "x"}),
            ToolExecutionContext(granted_permissions=frozenset()),  # no repo.read
        )
    )
    assert no_perm.success is False
    assert no_perm.error.error_type is ToolErrorType.PERMISSION
    assert no_perm.error.retryable is False

    # 4. Slow tool -> TIMEOUT (retryable=True, single attempt).
    timeout = asyncio.run(
        runtime.execute(
            ToolCall(tool_name="slow_repo", arguments={}),
            ToolExecutionContext(),
        )
    )
    assert timeout.success is False
    assert timeout.error.error_type is ToolErrorType.TIMEOUT
    assert timeout.error.retryable is True
    assert slow_handler.call_count == 1  # no retry

    # All four results came from the same runtime boundary with a
    # consistent structured shape.
    for r in (ok, bad_args, no_perm, timeout):
        assert isinstance(r.tool_name, str) and r.tool_name
        if r.success:
            assert r.error is None
        else:
            assert r.error is not None
            assert isinstance(r.error.error_type, ToolErrorType)
            assert isinstance(r.error.retryable, bool)
