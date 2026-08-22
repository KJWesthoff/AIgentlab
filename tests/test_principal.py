"""The principal, and what the gate does with it.

One principal, many agents: authorization derives from the human who
started the run, not from whichever agent is asking. These tests cover
the two properties that makes real — the gate checks scopes rather than
the caller, and the identity never enters a context window.
"""

import asyncio

import pytest
import yaml
from pydantic import BaseModel

from agentlab.agents.definitions import AgentSpec
from agentlab.agents.runtime import AgentRuntime
from agentlab.llm.registry import ModelProfile, ModelRegistry
from agentlab.llm.service import LLMService
from agentlab.llm.types import ToolCall
from agentlab.orchestration.policy import authorize_tool_call
from agentlab.orchestration.principal import Principal, load_principal
from agentlab.orchestration.state import BudgetTracker, TaskState
from agentlab.tools.definitions import Tool, ToolDefinition


class NoInput(BaseModel):
    pass


def make_tool(name: str, scope: str | None, read_only: bool = True) -> Tool:
    return Tool(
        ToolDefinition(
            name=name,
            description="test",
            read_only=read_only,
            required_scope=scope,
        ),
        NoInput,
        lambda: {"ok": True},
    )


def make_agent(tools: set[str]) -> AgentSpec:
    return AgentSpec(
        name="writer",
        description="test",
        model_profile="economical",
        system_prompt="test",
        allowed_tools=tools,
    )


HOLDER = Principal(name="alice", scopes=frozenset({"read:corpus"}))


# --- The gate asks on whose authority ------------------------------------


def test_a_held_scope_authorizes_the_call():
    decision = authorize_tool_call(
        make_agent({"search"}), "search", make_tool("search", "read:corpus"),
        HOLDER,
    )
    assert decision.allowed


def test_a_scope_the_principal_lacks_is_refused():
    """The allowlist says yes; authority says no. Authority wins."""
    decision = authorize_tool_call(
        make_agent({"publish"}), "publish",
        make_tool("publish", "write:reports"), HOLDER,
    )
    assert not decision.allowed
    assert "does not hold" in decision.reason
    # And it is refused on authority, not deferred to a human.
    assert not decision.requires_approval


def test_no_principal_means_no_authority_to_act_on():
    """A call with no principal has nobody's say-so behind it."""
    decision = authorize_tool_call(
        make_agent({"search"}), "search",
        make_tool("search", "read:corpus"), None,
    )
    assert not decision.allowed
    assert "no principal" in decision.reason


def test_authority_is_checked_before_the_approval_gate():
    """Approval cannot substitute for authority.

    A write tool the principal cannot authorize must be refused outright
    rather than presented to a human — otherwise the human is asked to
    approve something nobody was authorized to request.
    """
    decision = authorize_tool_call(
        make_agent({"publish"}), "publish",
        make_tool("publish", "write:reports", read_only=False), HOLDER,
    )
    assert not decision.allowed
    assert not decision.requires_approval
    assert "Approval cannot substitute for authority" in decision.reason


def test_holding_the_scope_still_requires_approval_to_write():
    """The two controls are independent: authority does not imply consent."""
    authorized = Principal(name="alice", scopes=frozenset({"write:reports"}))
    decision = authorize_tool_call(
        make_agent({"publish"}), "publish",
        make_tool("publish", "write:reports", read_only=False), authorized,
    )
    assert not decision.allowed
    assert decision.requires_approval


def test_an_unscoped_tool_is_never_authorized_by_scope():
    """Declaring no scope must not read as 'authorized by default'."""
    assert not HOLDER.authorizes(None)


# --- Propagation ---------------------------------------------------------


def build_runtime(tools, tracker=None):
    from scripted_provider import ScriptedProvider, scripted_text

    provider = ScriptedProvider([
        scripted_text("", tool_calls=[
            ToolCall(id="c1", name="search", arguments={}),
        ]),
        scripted_text("done"),
    ])
    registry = ModelRegistry({
        "economical": ModelProfile(
            provider="scripted", model="scripted/model",
            capabilities={"text", "tool_calling"},
        )
    })
    return AgentRuntime(
        service=LLMService(
            model_registry=registry, providers={"scripted": provider}
        ),
        tools=tools,
        tracker=tracker or BudgetTracker(),
    ), provider


def run_with(principal):
    agent = make_agent({"search"})
    runtime, provider = build_runtime(
        {"search": make_tool("search", "read:corpus")}
    )
    state = TaskState(task_id="t", objective="o", principal=principal)
    asyncio.run(
        runtime._tool_loop(agent, state, runtime._initial_messages(agent, "go"))
    )
    return state, provider


def test_the_principal_reaches_the_gate_through_task_state():
    state, _ = run_with(HOLDER)
    assert any(entry["event"] == "tool_call" for entry in state.history)


def test_a_run_without_the_scope_is_denied_at_the_gate():
    state, _ = run_with(Principal(name="bob", scopes=frozenset()))
    denials = [e for e in state.history if e["event"] == "policy_denial"]
    assert denials and "does not hold" in denials[0]["reason"]


def test_the_principal_never_enters_the_context_window():
    """An identity the model can read is one it can be steered to rewrite.

    The principal is a parameter, not a message. If it ever appears in a
    context window this test fails, because at that point the model could
    be persuaded to restate it.
    """
    _, provider = run_with(HOLDER)

    for request in provider.requests:
        for message in request.messages:
            assert HOLDER.name not in message.content
            assert "read:corpus" not in message.content


# --- Configuration -------------------------------------------------------


def test_load_principal_reads_name_and_scopes(tmp_path):
    path = tmp_path / "principal.yaml"
    path.write_text(yaml.safe_dump(
        {"principal": {"name": "carol", "scopes": ["read:corpus"]}}
    ))
    principal = load_principal(path)
    assert principal.name == "carol"
    assert principal.scopes == frozenset({"read:corpus"})


def test_local_user_resolves_to_the_real_account(tmp_path):
    """'local-user' must name a real account, not an invented one."""
    import getpass

    path = tmp_path / "principal.yaml"
    path.write_text(yaml.safe_dump(
        {"principal": {"name": "local-user", "scopes": []}}
    ))
    assert load_principal(path).name == getpass.getuser()


def test_scopes_can_be_narrowed_for_one_run(tmp_path):
    path = tmp_path / "principal.yaml"
    path.write_text(yaml.safe_dump(
        {"principal": {"name": "carol",
                       "scopes": ["read:corpus", "write:reports"]}}
    ))
    narrowed = load_principal(path, scopes=["read:corpus"])
    assert narrowed.scopes == frozenset({"read:corpus"})


def test_the_shipped_principal_covers_the_bundled_tools():
    """Config drift here breaks every run, so catch it in the suite."""
    from pathlib import Path

    from agentlab.tools.registry import build_default_tools
    from agentlab.tools.write_report import build_write_tools

    root = Path(__file__).resolve().parents[1]
    principal = load_principal(root / "config" / "principal.yaml")
    tools = {**build_default_tools(), **build_write_tools()}

    for name, tool in tools.items():
        scope = tool.definition.required_scope
        assert scope, f"{name} declares no scope"
        assert principal.authorizes(scope), f"{name} needs {scope}"
