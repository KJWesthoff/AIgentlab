"""Collectors: turn configuration and run traces into graph edges.

Two collectors, mirroring BloodHound's split between what SharpHound
reads out of the directory and what session collection observes on the
wire:

- :func:`collect_static` reads ``agents.yaml``, ``models.yaml``, the tool
  registry and the corpus. It describes what the configuration *permits*
  — the graph exists before anything runs and costs nothing to build.
- :func:`collect_runtime` replays a JSONL trace from a real run and adds
  what actually *happened*: tool calls made, policy denials, documents
  that genuinely entered an agent's context.

The gap between the two is itself informative, so runtime edges are kept
as distinct kinds (``Called``, ``Denied``) rather than merged into the
permission layer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..agents.definitions import AgentSpec, load_agents
from ..llm.registry import ModelRegistry
from ..orchestration.principal import load_principal
from ..tools.definitions import Tool
from ..tools.registry import build_default_tools
from ..tools.write_report import build_write_tools
from .model import EdgeKind, Graph, NodeKind

#: Documents in one corpus are interchangeable to the analysis — each is
#: untrusted, and each produces the same path shape — so materializing
#: more of them adds nodes without adding findings, and makes the graph
#: unreadable on a projector. A handful is enough to show the shape; the
#: corpus node still records the true count and that it was truncated.
MAX_DOCUMENTS = 5

_DOCUMENT_FIELD = re.compile(r'"document"\s*:\s*"((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Stage:
    """One step of the pipeline and the artifacts it exchanges.

    This mirrors ``Workflow.execute`` in orchestration/workflow.py, which
    is plain Python and cannot be introspected reliably. ``tests/`` asserts
    the two stay in step, so drift shows up as a test failure rather than
    as a silently wrong graph.
    """

    name: str
    agent: str
    produces: str
    consumes: tuple[str, ...]


PIPELINE: tuple[Stage, ...] = (
    Stage("research", "researcher", "ResearchResult", ()),
    Stage("analyze", "analyst", "AnalysisResult", ("ResearchResult",)),
    # The writer also consumes ReviewResult on the revision pass, which is
    # what closes the loop between reviewer and writer.
    Stage(
        "write",
        "writer",
        "Draft",
        ("ResearchResult", "AnalysisResult", "ReviewResult"),
    ),
    Stage("review", "reviewer", "ReviewResult", ("ResearchResult", "Draft")),
)


def node_id(kind: NodeKind, name: str) -> str:
    return f"{kind.value}:{name}"


def collect_static(
    *,
    config_dir: Path,
    corpus_dir: Path,
    tools: dict[str, Tool] | None = None,
    max_documents: int = MAX_DOCUMENTS,
) -> Graph:
    """Build the permission graph from configuration alone.

    ``tools`` defaults to everything main.py wires up. Both search
    backends publish identical tool *definitions* — only the ranking
    differs — so the cheap one is used to avoid downloading an embedding
    model just to draw a graph.
    """
    graph = Graph()
    agents = load_agents(config_dir / "agents.yaml")
    registry = ModelRegistry.from_yaml(config_dir / "models.yaml")
    if tools is None:
        tools = {**build_default_tools(corpus_dir), **build_write_tools()}

    _add_principal(graph, config_dir)
    _add_models(graph, registry)
    _add_tools(graph, tools)
    _add_corpus(graph, corpus_dir, max_documents)
    _add_agents(graph, agents, registry)
    _add_pipeline(graph, agents)
    _derive_taint(graph, agents, tools)

    return graph


def _add_principal(graph: Graph, config_dir: Path) -> None:
    """The human every agent acts for, and the authority they carry.

    One principal, many agents: authorization derives from this identity
    rather than from whichever agent is asking, so the graph can show
    that a delegation chain never reaches further than the person who
    started it.
    """
    path = config_dir / "principal.yaml"
    if not path.is_file():
        return

    principal = load_principal(path)
    principal_id = node_id(NodeKind.PRINCIPAL, principal.name)
    graph.add_node(
        principal_id,
        NodeKind.PRINCIPAL,
        principal.name,
        scopes=sorted(principal.scopes),
    )

    for scope in sorted(principal.scopes):
        scope_id = node_id(NodeKind.SCOPE, scope)
        graph.add_node(scope_id, NodeKind.SCOPE, scope)
        graph.add_edge(principal_id, scope_id, EdgeKind.HOLDS_SCOPE)


def _add_models(graph: Graph, registry: ModelRegistry) -> None:
    for name, profile in registry.profiles.items():
        profile_id = node_id(NodeKind.MODEL_PROFILE, name)
        graph.add_node(
            profile_id,
            NodeKind.MODEL_PROFILE,
            name,
            capabilities=sorted(profile.capabilities),
            maximum_cost_per_call_usd=(
                profile.limits.maximum_cost_per_call_usd
            ),
        )

        model_id = node_id(NodeKind.MODEL, profile.model)
        graph.add_node(model_id, NodeKind.MODEL, profile.model)
        graph.add_edge(profile_id, model_id, EdgeKind.BACKED_BY)

        provider_id = node_id(NodeKind.PROVIDER, profile.provider)
        graph.add_node(provider_id, NodeKind.PROVIDER, profile.provider)
        graph.add_edge(model_id, provider_id, EdgeKind.SERVED_BY)

        for capability in sorted(profile.capabilities):
            capability_id = node_id(NodeKind.CAPABILITY, capability)
            graph.add_node(capability_id, NodeKind.CAPABILITY, capability)
            graph.add_edge(profile_id, capability_id, EdgeKind.GRANTS)


def _add_tools(graph: Graph, tools: dict[str, Tool]) -> None:
    for name, tool in tools.items():
        definition = tool.definition
        tool_id = node_id(NodeKind.TOOL, name)
        graph.add_node(
            tool_id,
            NodeKind.TOOL,
            name,
            description=definition.description,
            risk=definition.risk,
            read_only=definition.read_only,
        )

        # A write-capable tool is gated on a human by policy.py, so the
        # gate is a real node on the path rather than an implicit rule.
        if not definition.read_only:
            gate_id = node_id(NodeKind.APPROVAL_GATE, "human-approval")
            graph.add_node(
                gate_id,
                NodeKind.APPROVAL_GATE,
                "human approval",
                description="Write-capable tools require human approval.",
            )
            graph.add_edge(tool_id, gate_id, EdgeKind.GUARDED_BY)

        if definition.required_scope:
            scope_id = node_id(NodeKind.SCOPE, definition.required_scope)
            graph.add_node(
                scope_id, NodeKind.SCOPE, definition.required_scope
            )
            graph.add_edge(tool_id, scope_id, EdgeKind.REQUIRES_SCOPE)

        for source in definition.reads:
            source_id = node_id(NodeKind.CORPUS, source)
            graph.add_node(source_id, NodeKind.CORPUS, source)
            graph.add_edge(tool_id, source_id, EdgeKind.READS)

        for sink in definition.writes:
            sink_id = node_id(NodeKind.CORPUS, sink)
            graph.add_node(sink_id, NodeKind.CORPUS, sink)
            graph.add_edge(tool_id, sink_id, EdgeKind.WRITES)


def _add_corpus(graph: Graph, corpus_dir: Path, max_documents: int) -> None:
    corpus_id = node_id(NodeKind.CORPUS, "corpus")
    documents = sorted(corpus_dir.rglob("*.md")) if corpus_dir.is_dir() else []

    graph.add_node(
        corpus_id,
        NodeKind.CORPUS,
        "corpus",
        path=str(corpus_dir),
        document_count=len(documents),
        truncated=len(documents) > max_documents,
        trusted=False,
    )

    for path in documents[:max_documents]:
        name = str(path.relative_to(corpus_dir))
        document_id = node_id(NodeKind.DOCUMENT, name)
        graph.add_node(
            document_id,
            NodeKind.DOCUMENT,
            name,
            # Corpus content is attacker-influenced by assumption: this is
            # the equivalent of a share an unprivileged user can write to.
            trusted=False,
        )
        graph.add_edge(corpus_id, document_id, EdgeKind.CONTAINS)


def _add_agents(
    graph: Graph, agents: dict[str, AgentSpec], registry: ModelRegistry
) -> None:
    for name, spec in agents.items():
        agent_id = node_id(NodeKind.AGENT, name)
        graph.add_node(
            agent_id,
            NodeKind.AGENT,
            name,
            description=spec.description,
            model_profile=spec.model_profile,
            allowed_tools=sorted(spec.allowed_tools),
            required_capabilities=sorted(spec.required_capabilities),
            max_calls=spec.max_calls,
        )

        profile_id = node_id(NodeKind.MODEL_PROFILE, spec.model_profile)
        graph.add_edge(agent_id, profile_id, EdgeKind.RUNS_ON)

        # Every agent acts for the one principal — the edge that makes a
        # delegation chain traceable back to a human.
        for principal in graph.of_kind(NodeKind.PRINCIPAL):
            graph.add_edge(agent_id, principal.id, EdgeKind.ACTS_FOR)

        for capability in sorted(spec.required_capabilities):
            capability_id = node_id(NodeKind.CAPABILITY, capability)
            graph.add_node(capability_id, NodeKind.CAPABILITY, capability)
            graph.add_edge(agent_id, capability_id, EdgeKind.REQUIRES)

        for tool_name in sorted(spec.allowed_tools):
            tool_id = node_id(NodeKind.TOOL, tool_name)
            # Deliberately not auto-created: an allowlist entry naming a
            # tool that does not exist is a dangling grant, and the
            # analyzer reports it by noticing the missing edge.
            graph.add_edge(agent_id, tool_id, EdgeKind.ALLOWED_TO_CALL)


def _add_pipeline(graph: Graph, agents: dict[str, AgentSpec]) -> None:
    """Wire the artifacts agents hand each other.

    These are the lateral-movement edges. An artifact is the only thing
    that crosses from one agent's context into another's, which makes it
    the exact analogue of a cached credential.
    """
    for stage in PIPELINE:
        agent_id = node_id(NodeKind.AGENT, stage.agent)
        if not graph.has_node(agent_id):
            continue

        artifact_id = node_id(NodeKind.ARTIFACT, stage.produces)
        graph.add_node(
            artifact_id, NodeKind.ARTIFACT, stage.produces, stage=stage.name
        )
        graph.add_edge(agent_id, artifact_id, EdgeKind.PRODUCES)

    for stage in PIPELINE:
        agent_id = node_id(NodeKind.AGENT, stage.agent)
        for artifact in stage.consumes:
            artifact_id = node_id(NodeKind.ARTIFACT, artifact)
            graph.add_edge(artifact_id, agent_id, EdgeKind.FLOWS_TO)


def _derive_taint(
    graph: Graph, agents: dict[str, AgentSpec], tools: dict[str, Tool]
) -> None:
    """Derive the edges that make composed permissions visible.

    ``CanInject``: a document reaches an agent's context because that
    agent may call a tool that reads the document's corpus. ``CanCoerce``:
    one agent's output reaches another agent's context, directly or
    through a chain of artifacts. Neither is configured anywhere — both
    fall out of the composition, which is the whole point.
    """
    for name, spec in agents.items():
        agent_id = node_id(NodeKind.AGENT, name)
        for tool_name in spec.allowed_tools:
            tool = tools.get(tool_name)
            if tool is None:
                continue
            for source in tool.definition.reads:
                source_id = node_id(NodeKind.CORPUS, source)
                for edge in graph.outgoing(
                    source_id, frozenset({EdgeKind.CONTAINS})
                ):
                    graph.add_edge(
                        edge.target,
                        agent_id,
                        EdgeKind.CAN_INJECT,
                        via_tool=tool_name,
                    )

    artifact_flow = frozenset({EdgeKind.PRODUCES, EdgeKind.FLOWS_TO})
    for node in graph.of_kind(NodeKind.AGENT):
        for target_id, path in graph.reachable_from(
            node.id, artifact_flow
        ).items():
            target = graph.node(target_id)
            if target is None or target.kind is not NodeKind.AGENT:
                continue
            graph.add_edge(
                node.id,
                target_id,
                EdgeKind.CAN_COERCE,
                hops=len(path),
                via=graph.describe_path(path),
            )


def collect_runtime(graph: Graph, trace_path: Path) -> Graph:
    """Overlay a recorded run onto an existing static graph.

    Tolerates a partial final line: a trace may be read while the run that
    writes it is still going.
    """
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        _apply_event(graph, event)

    return graph


def _apply_event(graph: Graph, event: dict) -> None:
    kind = event.get("event")
    agent = event.get("agent")
    if not agent:
        return

    agent_id = node_id(NodeKind.AGENT, agent)
    if not graph.has_node(agent_id):
        return

    # A write-capable call is escalated, not refused: policy defers and the
    # human answers in approval_decision. Recording the escalation as a
    # denial reported "the control held" for writes that were approved and
    # executed — with the APPROVED edge sitting right beside it.
    if (
        kind == "policy_decision"
        and not event.get("allowed", True)
        and not event.get("requires_approval", False)
    ):
        _add_denied_edge(graph, agent_id, event, event.get("reason", ""))

    elif kind == "tool_result":
        tool_id = node_id(NodeKind.TOOL, event.get("tool", ""))
        if graph.has_node(tool_id):
            graph.add_edge(agent_id, tool_id, EdgeKind.CALLED)
        _record_observed_documents(graph, agent_id, event.get("result", ""))

    elif kind == "approval_decision":
        if event.get("approved"):
            tool_id = node_id(NodeKind.TOOL, event.get("tool", ""))
            if graph.has_node(tool_id):
                graph.add_edge(
                    agent_id,
                    tool_id,
                    EdgeKind.APPROVED,
                    scope=event.get("scope", "once"),
                )
        else:
            # The refusal the escalation was deferred to. This is where a
            # write-capable denial actually happens.
            _add_denied_edge(graph, agent_id, event, event.get("reason", ""))

    elif kind == "artifact_produced":
        artifact = event.get("artifact_type", "")
        artifact_id = node_id(NodeKind.ARTIFACT, artifact)
        graph.add_node(artifact_id, NodeKind.ARTIFACT, artifact)
        graph.add_edge(agent_id, artifact_id, EdgeKind.PRODUCES, observed=True)


def _add_denied_edge(
    graph: Graph, agent_id: str, event: dict, reason: str
) -> None:
    tool_id = node_id(NodeKind.TOOL, event.get("tool", ""))
    graph.add_node(
        tool_id, NodeKind.TOOL, event.get("tool", ""), observed_only=True
    )
    graph.add_edge(agent_id, tool_id, EdgeKind.DENIED, reason=reason)


def _record_observed_documents(
    graph: Graph, agent_id: str, result: str
) -> None:
    """Mark the documents that provably entered this agent's context.

    Traced tool results are truncated for the viewer, so this scans for
    document names rather than parsing the JSON — a best-effort confirmation
    of statically derived ``CanInject`` edges, never the only source of them.
    """
    for match in _DOCUMENT_FIELD.finditer(result or ""):
        try:
            name = json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            continue
        document_id = node_id(NodeKind.DOCUMENT, name)
        if not graph.has_node(document_id):
            continue
        graph.add_edge(
            document_id, agent_id, EdgeKind.CAN_INJECT, observed=True
        )
