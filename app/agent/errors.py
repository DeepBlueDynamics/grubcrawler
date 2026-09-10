"""Typed error hierarchy for the agent module.

Every error carries a machine-readable `code` so loop callers never need to
parse exception messages.  Each code maps to the error semantics defined in
plan/MASTER_PLAN.md.
"""

from __future__ import annotations

from typing import Optional


class AgentError(Exception):
    """Base for all agent errors."""
    code: str = "agent_error"
    retriable: bool = False

    def __init__(self, message: str, *, code: Optional[str] = None, retriable: Optional[bool] = None):
        super().__init__(message)
        if code is not None:
            self.code = code
        if retriable is not None:
            self.retriable = retriable


class ValidationError(AgentError):
    """Request or argument validation failed."""
    code = "validation_error"
    retriable = False


class PolicyDeniedError(AgentError):
    """A policy gate rejected the action."""
    code = "policy_denied"
    retriable = False


class ToolTimeoutError(AgentError):
    """Tool execution exceeded its deadline."""
    code = "tool_timeout"
    retriable = True


class ToolUnavailableError(AgentError):
    """Requested tool does not exist or is not allowed."""
    code = "tool_unavailable"
    retriable = False


class ExecutionError(AgentError):
    """Catch-all for runtime failures during tool execution."""
    code = "execution_error"
    retriable = False


class ProviderError(AgentError):
    """LLM provider returned an error or was unreachable."""
    code = "provider_error"
    retriable = True


class StopConditionError(AgentError):
    """A stop condition was hit (max steps, wall time, etc.)."""
    code = "stop_condition"
    retriable = False
