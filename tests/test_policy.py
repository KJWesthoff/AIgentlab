from pydantic import BaseModel

from agentlab.agents.definitions import AgentSpec
from agentlab.orchestration.policy import authorize_tool_call
from agentlab.tools.definitions import Tool, ToolDefinition


class NoInput(BaseModel):
    pass


def make_agent(allowed_tools: set[str]) -> AgentSpec:
    return AgentSpec(
        name="researcher",
        description="test",
        model_profile="economical",
        system_prompt="test",
        allowed_tools=allowed_tools,
    )


def make_tool(name: str, read_only: bool = True) -> Tool:
    return Tool(
        ToolDefinition(name=name, description="test", read_only=read_only),
        NoInput,
        lambda: None,
    )


def test_allows_permitted_read_only_tool():
    decision = authorize_tool_call(
        make_agent({"search"}), "search", make_tool("search")
    )
    assert decision.allowed


def test_denies_tool_not_in_allowlist():
    decision = authorize_tool_call(
        make_agent(set()), "shell_execute", make_tool("shell_execute")
    )
    assert not decision.allowed
    assert "not allowed" in decision.reason


def test_denies_nonexistent_tool():
    decision = authorize_tool_call(make_agent({"ghost"}), "ghost", None)
    assert not decision.allowed


def test_write_tool_requires_approval():
    decision = authorize_tool_call(
        make_agent({"email_send"}),
        "email_send",
        make_tool("email_send", read_only=False),
    )
    assert not decision.allowed
    assert decision.requires_approval
