"""End-to-end workflow tests against a scripted test provider — no network, no key."""

import json
from pathlib import Path

import pytest
from scripted_provider import ScriptedProvider, scripted_text

from agentlab.agents.definitions import load_agents
from agentlab.agents.runtime import AgentRuntime
from agentlab.llm.registry import ModelProfile, ModelRegistry
from agentlab.llm.types import ToolCall
from agentlab.orchestration.state import BudgetTracker, ExecutionBudget
from agentlab.orchestration.workflow import Workflow
from agentlab.tools.registry import build_default_tools

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESEARCH = json.dumps(
    {
        "evidence": [
            {
                "claim": "RAG synthesizes answers from retrieved chunks.",
                "source": "rag-fundamentals.md",
                "excerpt": "retrieves relevant document chunks",
                "confidence": 0.9,
            }
        ],
        "unanswered_questions": [],
    }
)
ANALYSIS = json.dumps(
    {
        "conclusions": ["RAG adds synthesis on top of retrieval."],
        "contradictions": [],
        "unsupported_claims": [],
        "confidence": 0.8,
    }
)
DRAFT = "RAG retrieves chunks and synthesizes an answer."
APPROVED = json.dumps(
    {"approved": True, "required_changes": [], "unsupported_statements": []}
)
REJECTED = json.dumps(
    {
        "approved": False,
        "required_changes": ["Cite the source document."],
        "unsupported_statements": [],
    }
)


def make_registry() -> ModelRegistry:
    capabilities = {
        "text",
        "tool_calling",
        "structured_output",
        "long_context",
        "reasoning",
    }
    return ModelRegistry(
        {
            name: ModelProfile(
                provider="openrouter",
                model="scripted/model",
                capabilities=capabilities,
            )
            for name in ("economical", "researcher", "analyst", "reviewer")
        }
    )


def build_workflow(provider: ScriptedProvider) -> Workflow:
    from agentlab.llm.service import LLMService

    service = LLMService(
        model_registry=make_registry(),
        providers={"openrouter": provider},
    )
    tracker = BudgetTracker(budget=ExecutionBudget())
    runtime = AgentRuntime(
        service=service,
        tools=build_default_tools(PROJECT_ROOT / "data" / "corpus"),
        tracker=tracker,
    )
    agents = load_agents(PROJECT_ROOT / "config" / "agents.yaml")
    return Workflow(runtime=runtime, agents=agents, tracker=tracker)


async def test_approved_first_pass_with_tool_call():
    provider = ScriptedProvider(
        responses=[
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
        ]
    )

    result = await build_workflow(provider).execute("Explain RAG vs lookup.")

    assert result.approved
    assert result.revisions == 0
    assert result.final_answer == DRAFT
    assert result.tool_calls == 1
    assert any(e["event"] == "tool_call" for e in result.history)


async def test_rejected_draft_triggers_one_revision():
    provider = ScriptedProvider(
        responses=[
            scripted_text(RESEARCH),
            scripted_text(ANALYSIS),
            scripted_text(DRAFT),
            scripted_text(REJECTED),
            scripted_text(DRAFT + " (rag-fundamentals.md)"),
            scripted_text(APPROVED),
        ]
    )

    result = await build_workflow(provider).execute("Explain RAG vs lookup.")

    assert result.approved
    assert result.revisions == 1
    assert "rag-fundamentals.md" in result.final_answer


async def test_unauthorized_tool_is_denied_not_executed():
    """A prompt-injection-style request for a tool outside the allowlist
    must be denied by policy and surfaced to the model as a denial."""
    provider = ScriptedProvider(
        responses=[
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
        ]
    )

    workflow = build_workflow(provider)
    result = await workflow.execute("Explain RAG vs lookup.")

    assert result.tool_calls == 0
    denials = [e for e in result.history if e["event"] == "policy_denial"]
    assert len(denials) == 1
    assert denials[0]["tool"] == "shell_execute"

    # The denial reason was fed back to the model as a tool message.
    tool_messages = [
        m
        for request in provider.requests
        for m in request.messages
        if m.role.value == "tool"
    ]
    assert any("Denied by policy" in m.content for m in tool_messages)


async def test_invalid_structured_output_gets_one_retry():
    provider = ScriptedProvider(
        responses=[
            scripted_text("this is not json at all"),  # researcher, invalid
            scripted_text(RESEARCH),  # researcher retry
            scripted_text(ANALYSIS),
            scripted_text(DRAFT),
            scripted_text(APPROVED),
        ]
    )

    result = await build_workflow(provider).execute("Explain RAG vs lookup.")
    assert result.approved


def test_search_documents_finds_corpus_content():
    tools = build_default_tools(PROJECT_ROOT / "data" / "corpus")
    result = tools["search_documents"].execute(
        {"query": "retrieval augmented generation database lookup"}
    )
    assert result["results"]
    assert result["results"][0]["document"] == "rag-fundamentals.md"


def test_tool_rejects_invalid_arguments():
    tools = build_default_tools(PROJECT_ROOT / "data" / "corpus")
    with pytest.raises(Exception):
        tools["search_documents"].execute({"query": "x", "limit": 999})


EMPTY_RESEARCH = json.dumps({"evidence": [], "unanswered_questions": ["What is print()?"]})


async def test_workflow_stops_when_retrieval_finds_nothing():
    """A question the corpus cannot support must not reach the writer.

    Running the rest of the pipeline spends three more model calls to
    produce a refusal that was knowable after research, and the refusal
    reads like a broken system rather than a corpus that lacks the topic.
    """
    provider = ScriptedProvider([scripted_text(EMPTY_RESEARCH)])
    result = await build_workflow(provider).execute("What does print() do?")

    assert result.approved is False
    assert "No answer" in result.final_answer
    assert "--corpus-dir" in result.final_answer
    # The unanswered question is surfaced rather than swallowed.
    assert "What is print()?" in result.final_answer
    # Only the researcher ran: no analyst, writer or reviewer calls.
    assert result.model_calls == 1
    assert result.research.evidence == []


async def test_workflow_still_runs_when_evidence_exists():
    """The early exit must not fire on a normal run."""
    provider = ScriptedProvider(
        [
            scripted_text(RESEARCH),
            scripted_text(ANALYSIS),
            scripted_text(DRAFT),
            scripted_text(APPROVED),
        ]
    )
    result = await build_workflow(provider).execute("objective")

    assert result.approved is True
    assert result.model_calls > 1
