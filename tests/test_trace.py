"""Trace and live-viewer tests — offline, against the scripted provider.

The trace is the ground truth for the live viewer: these tests assert
that a run emits the events the viewer renders, in particular the exact
context window entering each model call and every policy decision.
"""

import json
import urllib.request
from pathlib import Path

from scripted_provider import ScriptedProvider, scripted_text
from test_workflow import (
    ANALYSIS,
    APPROVED,
    DRAFT,
    RESEARCH,
    TEST_PRINCIPAL,
    make_registry,
)

from agentlab.agents.definitions import load_agents
from agentlab.agents.runtime import AgentRuntime
from agentlab.llm.service import LLMService
from agentlab.llm.types import ToolCall, Usage
from agentlab.observability.server import TraceServer
from agentlab.observability.trace import TraceWriter
from agentlab.orchestration.state import BudgetTracker, ExecutionBudget
from agentlab.orchestration.workflow import Workflow
from agentlab.tools.registry import build_default_tools

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_traced_workflow(
    provider: ScriptedProvider, tracer: TraceWriter
) -> Workflow:
    service = LLMService(
        model_registry=make_registry(),
        providers={"openrouter": provider},
    )
    tracker = BudgetTracker(budget=ExecutionBudget())
    runtime = AgentRuntime(
        service=service,
        tools=build_default_tools(PROJECT_ROOT / "data" / "corpus"),
        tracker=tracker,
        tracer=tracer,
    )
    agents = load_agents(PROJECT_ROOT / "config" / "agents.yaml")
    return Workflow(
        runtime=runtime, agents=agents, tracker=tracker, tracer=tracer,
        principal=TEST_PRINCIPAL,
    )


def read_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


async def run_scripted(tmp_path: Path, responses: list) -> list[dict]:
    tracer = TraceWriter(tmp_path / "trace.jsonl")
    workflow = build_traced_workflow(ScriptedProvider(responses), tracer)
    await workflow.execute("Explain RAG vs lookup.")
    tracer.close()
    return read_trace(tracer.path)


async def test_trace_captures_context_windows_and_lifecycle(tmp_path):
    events = await run_scripted(
        tmp_path,
        [
            scripted_text(
                "",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="search_documents",
                        arguments={"query": "RAG database lookup"},
                    )
                ],
            ),
            scripted_text(RESEARCH),
            scripted_text(ANALYSIS),
            scripted_text(DRAFT),
            scripted_text(APPROVED),
        ],
    )

    sequences = [e["seq"] for e in events]
    assert sequences == sorted(sequences)

    started = events[0]
    assert started["event"] == "run_started"
    researcher = next(a for a in started["agents"] if a["name"] == "researcher")
    assert researcher["allowed_tools"] == ["search_documents"]
    assert started["budget"]["maximum_model_calls"] == 12

    # The first researcher call carries the full context window: the
    # system prompt (with the untrusted-data instruction) and the task.
    first_call = next(e for e in events if e["event"] == "model_call_started")
    assert first_call["agent"] == "researcher"
    assert first_call["tools_offered"] == ["search_documents"]
    roles = [m["role"] for m in first_call["messages"]]
    assert roles == ["system", "user"]
    assert "untrusted data" in first_call["messages"][0]["content"]

    # The allowed tool call produces a policy decision and an
    # untrusted-labeled result...
    decision = next(e for e in events if e["event"] == "policy_decision")
    assert decision["allowed"] and decision["tool"] == "search_documents"
    result = next(e for e in events if e["event"] == "tool_result")
    assert result["untrusted"]
    assert result["result"].startswith("[UNTRUSTED TOOL OUTPUT")
    assert result["budget"]["tool_calls"] == 1

    # ...and the next context window shows that labeled result re-entering
    # the researcher's context — the thing the viewer exists to show.
    later_calls = [e for e in events if e["event"] == "model_call_started"][1:]
    tool_messages = [
        m
        for call in later_calls
        for m in call["messages"]
        if m["role"] == "tool"
    ]
    assert any(m["untrusted"] for m in tool_messages)

    artifact_types = [
        e["artifact_type"] for e in events if e["event"] == "artifact_produced"
    ]
    assert {"ResearchResult", "AnalysisResult", "ReviewResult"} <= set(
        artifact_types
    )
    assert any(e["event"] == "draft_produced" for e in events)

    finished = events[-1]
    assert finished["event"] == "run_finished"
    assert finished["approved"] is True
    assert finished["final_answer"] == DRAFT


async def test_trace_records_policy_denial(tmp_path):
    events = await run_scripted(
        tmp_path,
        [
            scripted_text(
                "",
                tool_calls=[
                    ToolCall(id="c1", name="shell_execute", arguments={"cmd": "id"})
                ],
            ),
            scripted_text(RESEARCH),
            scripted_text(ANALYSIS),
            scripted_text(DRAFT),
            scripted_text(APPROVED),
        ],
    )

    decision = next(e for e in events if e["event"] == "policy_decision")
    assert decision["allowed"] is False
    assert decision["tool"] == "shell_execute"
    assert decision["reason"]
    # A tool off the allowlist is refused outright — no human is asked,
    # which is what separates this from a write-capable escalation.
    assert decision["requires_approval"] is False
    assert not any(e["event"] == "approval_requested" for e in events)
    assert not any(e["event"] == "tool_result" for e in events)


async def test_every_model_call_reports_usage_for_the_viewers_tally(tmp_path):
    """The viewer tallies tokens per agent, so every call must report.

    The analyst and reviewer only ever make structured calls; when
    those returned the artifact alone their columns read zero, which
    looks like a broken tally rather than the spend it hides.
    """
    usage = Usage(input_tokens=200, output_tokens=40, estimated_cost=0.0015)
    events = await run_scripted(
        tmp_path,
        [
            scripted_text(RESEARCH, usage=usage),
            scripted_text(ANALYSIS, usage=usage),
            scripted_text(DRAFT, usage=usage),
            scripted_text(APPROVED, usage=usage),
        ],
    )

    finished = [e for e in events if e["event"] == "model_call_finished"]
    agents = {e["agent"] for e in finished}
    assert {"researcher", "analyst", "writer", "reviewer"} <= agents
    assert all(e["usage"]["input_tokens"] == 200 for e in finished)

    # Structured calls carry the artifact's own cost, and the running
    # budget the viewer prints beside the tally agrees with the sum.
    structured = [e for e in finished if e.get("call_kind") == "structured"]
    assert {e["agent"] for e in structured} == {"analyst", "reviewer"}
    assert all(e["round_trips"] == 1 for e in structured)

    last = finished[-1]["budget"]
    assert last["input_tokens"] == 200 * len(finished)
    assert last["output_tokens"] == 40 * len(finished)


def test_server_serves_viewer_and_events(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    tracer = TraceWriter(trace_path)
    tracer.emit("run_started", objective="x", agents=[], budget={})
    tracer.emit("stage_started", stage="research", agent="researcher")
    tracer.close()

    server = TraceServer(trace_path=trace_path, port=0)
    server.start()
    try:
        with urllib.request.urlopen(server.url) as response:
            page = response.read().decode()
        assert "agentlab" in page and "Security measures" in page

        with urllib.request.urlopen(server.url + "events?after=0") as response:
            payload = json.load(response)
        assert [e["event"] for e in payload["events"]] == [
            "run_started",
            "stage_started",
        ]

        with urllib.request.urlopen(server.url + "events?after=1") as response:
            payload = json.load(response)
        assert [e["seq"] for e in payload["events"]] == [2]
    finally:
        server.stop()
