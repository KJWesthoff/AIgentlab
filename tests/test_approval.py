"""The write tool and the human approval gate in front of it.

``save_report`` is the only tool here that changes state, so these tests
cover both halves of what protects it: the path validation inside the
tool, and the approval gate that decides whether it runs at all.

The fatigue tests are the point of the exercise. A gate that offers
"don't ask again" — and they all do, because prompting every call is
unusable — stops being a per-call control the moment someone takes it.
"""

import io
import json

import pytest

from agentlab.orchestration.approval import (
    ApprovalScope,
    ConsoleApprover,
    DenyingApprover,
)
from agentlab.tools.write_report import (
    UnsafePathError,
    build_write_tools,
    resolve_report_path,
)


def approver(answers: str) -> ConsoleApprover:
    return ConsoleApprover(
        input_stream=io.StringIO(answers), output_stream=io.StringIO()
    )


def request(instance: ConsoleApprover, tool: str = "save_report"):
    return instance.request(
        agent="writer",
        tool=tool,
        arguments={"filename": "summary.md", "content": "# Summary"},
        preview="# Summary",
    )


# --- The tool actually writes -------------------------------------------


def test_save_report_writes_a_real_file(tmp_path):
    tool = build_write_tools(tmp_path)["save_report"]
    result = tool.execute({"filename": "summary.md", "content": "# Hello"})

    written = tmp_path / "summary.md"
    assert written.read_text() == "# Hello"
    assert result["bytes"] == len("# Hello")
    assert result["overwrote_existing"] is False


def test_save_report_reports_an_overwrite(tmp_path):
    """Overwriting is destructive, so the result says so explicitly."""
    tool = build_write_tools(tmp_path)["save_report"]
    tool.execute({"filename": "summary.md", "content": "first"})
    result = tool.execute({"filename": "summary.md", "content": "second"})

    assert result["overwrote_existing"] is True
    assert (tmp_path / "summary.md").read_text() == "second"


def test_save_report_is_not_read_only():
    """What makes it the sink at the end of an attack path."""
    definition = build_write_tools()["save_report"].definition
    assert definition.read_only is False
    assert definition.writes == ["reports"]


@pytest.mark.parametrize(
    "filename",
    [
        "../escape.md",
        "/etc/passwd.md",
        "nested/report.md",
        "report.txt",
        "",
        "   ",
    ],
)
def test_save_report_refuses_paths_outside_its_directory(tmp_path, filename):
    """The filename comes from a model acting on attacker-influenced text."""
    with pytest.raises(UnsafePathError):
        resolve_report_path(tmp_path, filename)


def test_save_report_rejects_traversal_via_the_resolved_path(tmp_path):
    """Checking the resolved path is what defeats a symlinked escape."""
    outside = tmp_path / "outside"
    outside.mkdir()
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "link.md").symlink_to(outside / "link.md")

    with pytest.raises(UnsafePathError):
        resolve_report_path(reports, "link.md")


def test_save_report_refuses_an_oversized_payload(tmp_path):
    tool = build_write_tools(tmp_path)["save_report"]
    with pytest.raises(ValueError, match="limit"):
        tool.execute({"filename": "big.md", "content": "x" * 300_000})


# --- The gate -----------------------------------------------------------


def test_no_approver_means_no_writes():
    """Unattended runs must not silently gain write access."""
    decision = DenyingApprover().request(
        agent="writer", tool="save_report", arguments={}, preview=""
    )
    assert decision.approved is False
    assert decision.scope is ApprovalScope.NONE
    assert "--approve-writes" in decision.reason


def test_approving_once_does_not_approve_the_next_call():
    instance = approver("y\nn\n")

    first = request(instance)
    assert first.approved is True
    assert first.scope is ApprovalScope.ONCE

    # The grant did not persist, so the human is asked again.
    assert request(instance).approved is False
    assert instance.session_grants == set()


@pytest.mark.parametrize("answer", ["n", "", "no", "q"])
def test_anything_but_yes_denies(answer):
    assert request(approver(f"{answer}\n")).approved is False


def test_the_prompt_shows_what_is_being_approved():
    output = io.StringIO()
    instance = ConsoleApprover(
        input_stream=io.StringIO("n\n"), output_stream=output
    )
    request(instance)
    shown = output.getvalue()

    assert "save_report" in shown
    assert "writer" in shown
    assert "summary.md" in shown
    assert "# Summary" in shown


# --- Approval fatigue ---------------------------------------------------


def test_approving_for_the_session_stops_asking():
    """One 'a' and the gate never troubles anyone again this run.

    This is the fatigue vector, reproduced faithfully rather than
    caricatured: the option exists in every real implementation because
    per-call prompting is unusable.
    """
    instance = approver("a\n")

    first = request(instance)
    assert first.approved is True
    assert first.scope is ApprovalScope.SESSION

    # No further input is available — a second read would raise or block
    # if the gate asked again. It does not ask.
    for _ in range(5):
        later = request(instance)
        assert later.approved is True
        assert later.scope is ApprovalScope.SESSION


def test_a_session_grant_covers_only_the_tool_it_was_given_for():
    """Fatigue should not spread to tools nobody approved."""
    instance = approver("a\nn\n")
    assert request(instance, "save_report").approved is True
    assert request(instance, "delete_everything").approved is False


def test_auto_approved_calls_are_still_announced():
    """The transcript must show what happened, even unattended.

    Nobody reads it in time to object — that is the finding — but a run
    that hid the calls entirely would be indefensible.
    """
    output = io.StringIO()
    instance = ConsoleApprover(
        input_stream=io.StringIO("a\n"), output_stream=output
    )
    request(instance)
    output.truncate(0)
    output.seek(0)

    request(instance)
    assert "auto-approved" in output.getvalue()
    assert "session grant" in output.getvalue()


# --- End to end through the agent runtime -------------------------------


def run_writer_proposing_a_save(tmp_path, approver_instance):
    """Drive one agent through a proposed save_report call.

    Uses the scripted provider so the tool-call path is exercised for
    real — policy check, approval, execution — without a model call.
    """
    import asyncio

    from scripted_provider import ScriptedProvider, scripted_text

    from agentlab.agents.definitions import AgentSpec
    from agentlab.agents.runtime import AgentRuntime
    from agentlab.llm.registry import ModelProfile, ModelRegistry
    from agentlab.llm.service import LLMService
    from agentlab.llm.types import ToolCall
    from agentlab.observability.trace import TraceWriter
    from agentlab.orchestration.principal import Principal
    from agentlab.orchestration.state import BudgetTracker, TaskState

    agent = AgentSpec(
        name="writer",
        description="test",
        model_profile="economical",
        system_prompt="test",
        allowed_tools={"save_report"},
        max_calls=2,
    )
    provider = ScriptedProvider(
        [
            scripted_text(
                "",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="save_report",
                        arguments={
                            "filename": "summary.md",
                            "content": "# Summary",
                        },
                    )
                ],
            ),
            scripted_text("done"),
        ]
    )
    registry = ModelRegistry(
        {
            "economical": ModelProfile(
                provider="scripted",
                model="scripted/model",
                capabilities={"text", "tool_calling", "structured_output"},
            )
        }
    )
    trace_path = tmp_path / "trace.jsonl"
    tracer = TraceWriter(trace_path)
    runtime = AgentRuntime(
        service=LLMService(
            model_registry=registry, providers={"scripted": provider}
        ),
        tools=build_write_tools(tmp_path / "reports"),
        tracker=BudgetTracker(),
        tracer=tracer,
        approver=approver_instance,
    )
    state = TaskState(
        task_id="t",
        objective="save it",
        principal=Principal(
            name="test-user", scopes=frozenset({"write:reports"})
        ),
    )
    asyncio.run(
        runtime._tool_loop(agent, state, runtime._initial_messages(agent, "go"))
    )
    tracer.close()

    events = [
        json.loads(line)
        for line in trace_path.read_text().splitlines()
        if line.strip()
    ]
    return state, events, tmp_path / "reports" / "summary.md"


def test_runtime_denies_the_write_with_no_approver(tmp_path):
    state, events, written = run_writer_proposing_a_save(tmp_path, None)

    assert not written.exists()
    assert any(e["event"] == "policy_decision" and not e["allowed"] for e in events)
    assert any(e["event"] == "approval_decision" and not e["approved"] for e in events)


def test_the_escalation_is_traced_as_an_escalation_not_a_denial(tmp_path):
    """`policy_decision` must say it is asking, not refusing.

    Policy returns allowed=False for a write-capable call because it
    will not decide alone — the human decides next. A trace that only
    carries the flag showed the live viewer a denial for a call that
    was then approved and executed, with the write's own tool_result
    printed underneath it.
    """
    _, events, written = run_writer_proposing_a_save(tmp_path, approver("y\n"))
    assert written.exists()

    decision = next(e for e in events if e["event"] == "policy_decision")
    assert decision["allowed"] is False
    assert decision["requires_approval"] is True

    approval = next(e for e in events if e["event"] == "approval_decision")
    assert approval["approved"] is True
    # Nothing in the trace claims this call was denied.
    assert not any(
        e["event"] == "policy_decision" and not e.get("requires_approval")
        for e in events
    )


def test_runtime_writes_the_file_once_a_human_approves(tmp_path):
    state, events, written = run_writer_proposing_a_save(tmp_path, approver("y\n"))

    assert written.read_text() == "# Summary"
    decision = next(e for e in events if e["event"] == "approval_decision")
    assert decision["approved"] is True
    assert decision["scope"] == "once"


def test_a_session_grant_is_traced_as_such(tmp_path):
    """The trace has to distinguish the two, or the graph cannot report it."""
    _, events, written = run_writer_proposing_a_save(tmp_path, approver("a\n"))

    assert written.exists()
    decision = next(e for e in events if e["event"] == "approval_decision")
    assert decision["scope"] == "session"


def test_an_agent_with_tools_can_use_them_in_a_prose_stage(tmp_path):
    """Regression: the writer's draft stage had no tool loop.

    save_report was in the writer's allowlist and reachable in the
    permission graph, but workflow.py calls run_text, which generated
    prose and never offered the tools — so the grant was a promise the
    runtime did not keep, and a live run silently wrote nothing.
    """
    import asyncio

    from scripted_provider import ScriptedProvider, scripted_text

    from agentlab.agents.definitions import AgentSpec
    from agentlab.agents.runtime import AgentRuntime
    from agentlab.llm.registry import ModelProfile, ModelRegistry
    from agentlab.llm.service import LLMService
    from agentlab.llm.types import ToolCall
    from agentlab.orchestration.principal import Principal
    from agentlab.orchestration.state import BudgetTracker, TaskState

    agent = AgentSpec(
        name="writer",
        description="test",
        model_profile="economical",
        system_prompt="test",
        allowed_tools={"save_report"},
        max_calls=2,
    )
    provider = ScriptedProvider(
        [
            scripted_text(
                "",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="save_report",
                        arguments={"filename": "d.md", "content": "# Draft"},
                    )
                ],
            ),
            scripted_text("# Draft"),
        ]
    )
    runtime = AgentRuntime(
        service=LLMService(
            model_registry=ModelRegistry(
                {
                    "economical": ModelProfile(
                        provider="scripted",
                        model="scripted/model",
                        capabilities={"text", "tool_calling"},
                    )
                }
            ),
            providers={"scripted": provider},
        ),
        tools=build_write_tools(tmp_path),
        tracker=BudgetTracker(),
        approver=approver("y\n"),
    )

    draft = asyncio.run(
        runtime.run_text(
            agent=agent,
            state=TaskState(
                task_id="t",
                objective="o",
                principal=Principal(
                    name="test-user", scopes=frozenset({"write:reports"})
                ),
            ),
            task_input="write and save it",
        )
    )

    assert draft == "# Draft"
    assert (tmp_path / "d.md").read_text() == "# Draft"


def test_every_request_carries_the_agents_output_cap():
    """Regression: an uncapped writer emitted 65,535 tokens in one call."""
    import asyncio

    from scripted_provider import ScriptedProvider, scripted_text

    from agentlab.agents.definitions import AgentSpec
    from agentlab.agents.runtime import AgentRuntime
    from agentlab.llm.registry import ModelProfile, ModelRegistry
    from agentlab.llm.service import LLMService
    from agentlab.orchestration.principal import Principal
    from agentlab.orchestration.state import BudgetTracker, TaskState

    agent = AgentSpec(
        name="writer",
        description="test",
        model_profile="economical",
        system_prompt="test",
        max_output_tokens=1234,
    )
    provider = ScriptedProvider([scripted_text("draft")])
    runtime = AgentRuntime(
        service=LLMService(
            model_registry=ModelRegistry(
                {
                    "economical": ModelProfile(
                        provider="scripted",
                        model="scripted/model",
                        capabilities={"text"},
                    )
                }
            ),
            providers={"scripted": provider},
        ),
        tools={},
        tracker=BudgetTracker(),
    )
    asyncio.run(
        runtime.run_text(
            agent=agent,
            state=TaskState(task_id="t", objective="o"),
            task_input="go",
        )
    )

    assert provider.requests[0].max_output_tokens == 1234
