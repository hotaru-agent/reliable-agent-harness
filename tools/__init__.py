"""Tool runtime layer (Phase 2 Step 1 + Step 2).

Exposes the tool contract, registry, validator, retry policy and the
async tool runtime with deterministic retry. No MCP, no OpenTelemetry.
"""

from tools.errors import DuplicateToolError, ToolReportedFailure
from tools.models import (
    DEFAULT_RETRYABLE,
    RegisteredTool,
    RetryPolicy,
    ToolCall,
    ToolError,
    ToolErrorType,
    ToolExecutionContext,
    ToolExecutionResult,
    ToolHandler,
    ToolSideEffect,
    ToolSource,
    ToolSpec,
)
from tools.registry import ToolRegistry
from tools.retry import compute_backoff, should_retry
from tools.runtime import ToolRuntime
from tools.validator import (
    SchemaError,
    SimpleToolArgumentValidator,
    ToolArgumentValidator,
    ValidationError,
)

__all__ = [
    "DEFAULT_RETRYABLE",
    "DuplicateToolError",
    "RegisteredTool",
    "RetryPolicy",
    "SchemaError",
    "SimpleToolArgumentValidator",
    "ToolArgumentValidator",
    "ToolCall",
    "ToolError",
    "ToolErrorType",
    "ToolExecutionContext",
    "ToolExecutionResult",
    "ToolHandler",
    "ToolRegistry",
    "ToolReportedFailure",
    "ToolRuntime",
    "ToolSideEffect",
    "ToolSource",
    "ToolSpec",
    "ValidationError",
    "compute_backoff",
    "should_retry",
]
