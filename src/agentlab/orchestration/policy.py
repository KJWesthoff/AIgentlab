"""Policy enforcement point.

The model proposes; this module authorizes. A model output is never
authorization — every proposed tool call passes through here before
execution, regardless of which agent produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agents.definitions import AgentSpec
from ..tools.definitions import Tool


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str
    requires_approval: bool = False


def authorize_tool_call(
    agent: AgentSpec,
    tool_name: str,
    tool: Tool | None,
) -> PolicyDecision:
    if not tool_name:
        return PolicyDecision(False, "No tool was specified.")

    if tool_name not in agent.allowed_tools:
        return PolicyDecision(
            False, f"Tool {tool_name!r} is not allowed for agent {agent.name!r}."
        )

    if tool is None:
        return PolicyDecision(False, f"Requested tool {tool_name!r} does not exist.")

    if not tool.definition.read_only:
        return PolicyDecision(
            allowed=False,
            reason="Write-capable tools require human approval.",
            requires_approval=True,
        )

    return PolicyDecision(True, "Allowed by policy.")
