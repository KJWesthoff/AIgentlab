"""Policy enforcement point.

The model proposes; this module authorizes. A model output is never
authorization — every proposed tool call passes through here before
execution, regardless of which agent produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agents.definitions import AgentSpec
from ..tools.definitions import Tool
from .principal import Principal


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str
    requires_approval: bool = False


def authorize_tool_call(
    agent: AgentSpec,
    tool_name: str,
    tool: Tool | None,
    principal: Principal | None = None,
) -> PolicyDecision:
    """Authorize one proposed call.

    Order matters. The agent's allowlist says *what this stage may do*;
    the principal's scopes say *on whose authority it may do it*. A gate
    that only checks the first approves anything a trusted peer asks for,
    which is how a request gets laundered through a delegation chain.
    """
    if not tool_name:
        return PolicyDecision(False, "No tool was specified.")

    if tool_name not in agent.allowed_tools:
        return PolicyDecision(
            False, f"Tool {tool_name!r} is not allowed for agent {agent.name!r}."
        )

    if tool is None:
        return PolicyDecision(False, f"Requested tool {tool_name!r} does not exist.")

    scope = tool.definition.required_scope
    if scope is not None:
        if principal is None:
            return PolicyDecision(
                False,
                f"Tool {tool_name!r} needs scope {scope!r}, but the call "
                "carries no principal — nobody's authority to act on.",
            )
        if not principal.authorizes(scope):
            return PolicyDecision(
                False,
                f"Principal {principal.name!r} does not hold {scope!r}, "
                f"which {tool_name!r} requires. Approval cannot substitute "
                "for authority.",
            )

    if not tool.definition.read_only:
        return PolicyDecision(
            allowed=False,
            reason="Write-capable tools require human approval.",
            requires_approval=True,
        )

    return PolicyDecision(True, "Allowed by policy.")
