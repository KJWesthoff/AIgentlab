"""Pre-built queries over the permission graph.

BloodHound's value is not the graph, it is the canned questions: "find
shortest paths to Domain Admins", "find principals with DCSync". These
are the equivalents for an agent system. Each returns findings with the
path that produced them, so a finding can always be traced back to the
configuration that caused it.

Every check here is read-only and derives from the graph alone. Nothing
in this module feeds back into orchestration — it reports, it does not
enforce. Enforcement stays in orchestration/policy.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .model import TAINT_EDGES, EdgeKind, Graph, NodeKind


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


_ORDER = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.INFO: 4,
}


@dataclass(frozen=True)
class Finding:
    check: str
    severity: Severity
    title: str
    detail: str
    remediation: str
    nodes: tuple[str, ...] = ()
    path: str = ""


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def sorted(self) -> list[Finding]:
        return sorted(
            self.findings, key=lambda f: (_ORDER[f.severity], f.check, f.title)
        )

    def count(self, severity: Severity) -> int:
        return sum(1 for f in self.findings if f.severity is severity)


def analyze(graph: Graph) -> Report:
    report = Report()
    for check in (
        _untrusted_to_write_tool,
        _confused_deputy,
        _injection_reachable_agents,
        _crosscheck_independence,
        _dangling_tool_grants,
        _capability_gaps,
        _tool_grant_without_tool_calling,
        _orphaned_tools,
        _runtime_drift,
        _approval_fatigue,
        _observed_denials,
    ):
        check(graph, report)
    return report


# --- The headline query: path to a write-capable tool -------------------


def _untrusted_to_write_tool(graph: Graph, report: Report) -> None:
    """The "shortest path to Domain Admin" of this system.

    Untrusted content reaching a tool that can change state is the
    outcome every other control exists to prevent.
    """
    write_tools = [
        node
        for node in graph.of_kind(NodeKind.TOOL)
        if node.properties.get("read_only") is False
    ]
    if not write_tools:
        return

    # One finding per tool, not per document: every document in a corpus
    # produces the same path, and repeating it buries the other findings.
    for tool in write_tools:
        for document in graph.of_kind(NodeKind.DOCUMENT):
            path = graph.shortest_path(document.id, tool.id, TAINT_EDGES)
            if path is None:
                continue

            gated = bool(graph.outgoing(tool.id, frozenset({EdgeKind.GUARDED_BY})))
            report.add(
                Finding(
                    check="untrusted-to-write-tool",
                    severity=Severity.HIGH if gated else Severity.CRITICAL,
                    title=(
                        f"Untrusted content can reach write-capable tool "
                        f"{tool.label!r}"
                    ),
                    detail=(
                        f"Content in {document.label!r} reaches {tool.label!r} "
                        f"in {len(path)} hops. "
                        + (
                            "A human approval gate stands on the path, so the "
                            "call cannot execute unattended — but the approval "
                            "request itself is attacker-influenced."
                            if gated
                            else "Nothing on the path requires human approval."
                        )
                    ),
                    remediation=(
                        "Break the chain: drop the tool from the agent's "
                        "allowlist, or stop routing untrusted artifacts into "
                        "an agent that holds a write-capable tool."
                    ),
                    nodes=(document.id, tool.id),
                    path=graph.describe_path(path),
                )
            )
            break


# --- Composed permissions: the nested-group problem ---------------------


def _confused_deputy(graph: Graph, report: Report) -> None:
    """An agent reaching a tool it was never granted, through another agent.

    Structurally identical to a user who is not a Domain Admin but is a
    member of a group nested inside one. The allowlist says no; the
    composition says yes.
    """
    coerce = frozenset({EdgeKind.CAN_COERCE})

    for agent in graph.of_kind(NodeKind.AGENT):
        granted = set(agent.properties.get("allowed_tools") or [])

        for edge in graph.outgoing(agent.id, coerce):
            deputy = graph.node(edge.target)
            if deputy is None:
                continue
            borrowed = set(deputy.properties.get("allowed_tools") or [])
            gained = borrowed - granted
            if not gained:
                continue

            report.add(
                Finding(
                    check="confused-deputy",
                    severity=Severity.MEDIUM,
                    title=(
                        f"{agent.label!r} can influence {deputy.label!r}, which "
                        f"holds {', '.join(sorted(gained))}"
                    ),
                    detail=(
                        f"{agent.label!r} is not allowed {sorted(gained)}, but "
                        f"its output enters {deputy.label!r}'s context, and "
                        f"{deputy.label!r} is. Text {agent.label!r} controls "
                        "can therefore steer a tool it does not hold."
                    ),
                    remediation=(
                        "Validate or narrow the artifact crossing the boundary, "
                        "or move the tool to an agent that consumes nothing "
                        "attacker-influenced."
                    ),
                    nodes=(agent.id, deputy.id),
                    path=edge.properties.get("via", ""),
                )
            )


def _injection_reachable_agents(graph: Graph, report: Report) -> None:
    """Agents whose context untrusted content reaches without them reading it.

    The agent that runs the search knows it handles untrusted data.
    The three downstream of it inherit the taint without ever touching a
    document, and that is the part config review misses.
    """
    direct = {
        edge.target
        for document in graph.of_kind(NodeKind.DOCUMENT)
        for edge in graph.outgoing(
            document.id, frozenset({EdgeKind.CAN_INJECT})
        )
    }

    reached: dict[str, str] = {}
    for document in graph.of_kind(NodeKind.DOCUMENT):
        for target_id, path in graph.reachable_from(
            document.id, TAINT_EDGES
        ).items():
            node = graph.node(target_id)
            if node is None or node.kind is not NodeKind.AGENT:
                continue
            reached.setdefault(target_id, graph.describe_path(path))

    for agent_id, path in sorted(reached.items()):
        if agent_id in direct:
            continue
        agent = graph.node(agent_id)
        if agent is None:
            continue

        holds_tools = bool(agent.properties.get("allowed_tools"))
        report.add(
            Finding(
                check="indirect-injection-reach",
                severity=Severity.HIGH if holds_tools else Severity.MEDIUM,
                title=(
                    f"{agent.label!r} is injection-reachable but reads no "
                    "documents"
                ),
                detail=(
                    f"{agent.label!r} never calls a corpus tool, so its "
                    "allowlist looks clean, but untrusted document content "
                    "arrives in its context through upstream artifacts."
                    + (
                        " It also holds tools."
                        if holds_tools
                        else " It holds no tools, which bounds the impact."
                    )
                ),
                remediation=(
                    "Treat inherited artifacts as untrusted the way tool "
                    "results already are — agents/runtime.py labels tool "
                    "output with UNTRUSTED_PREFIX but passes artifacts in "
                    "clean."
                ),
                nodes=(agent_id,),
                path=path,
            )
        )


def _crosscheck_independence(graph: Graph, report: Report) -> None:
    """Two agents that are supposed to check each other sharing a model.

    A reviewer that runs the same model as the writer reproduces the
    writer's blind spots, so the check reports success it did not earn.
    The AD analogue is one account administering both tiers.
    """
    pairs = (("writer", "reviewer"), ("researcher", "analyst"))
    runs_on = frozenset({EdgeKind.RUNS_ON, EdgeKind.BACKED_BY})

    for first, second in pairs:
        models: list[tuple[str, str, str]] = []
        for name in (first, second):
            agent_id = f"{NodeKind.AGENT.value}:{name}"
            if not graph.has_node(agent_id):
                break
            for target_id, _ in graph.reachable_from(agent_id, runs_on).items():
                node = graph.node(target_id)
                if node is not None and node.kind is NodeKind.MODEL:
                    models.append((name, target_id, node.label))
        if len(models) != 2 or models[0][1] != models[1][1]:
            continue

        report.add(
            Finding(
                check="crosscheck-not-independent",
                severity=Severity.MEDIUM,
                title=(
                    f"{first!r} and {second!r} both resolve to "
                    f"{models[0][2]!r}"
                ),
                detail=(
                    f"{second!r} is meant to catch {first!r}'s mistakes, but "
                    "both run the same model, so correlated failures pass "
                    "unnoticed and the check reads as stronger than it is."
                ),
                remediation=(
                    "Point the two profiles at different model families in "
                    "config/models.yaml."
                ),
                nodes=(f"{NodeKind.AGENT.value}:{first}", models[0][1]),
            )
        )


# --- Hygiene: the ACL problems BloodHound also surfaces ------------------


def _dangling_tool_grants(graph: Graph, report: Report) -> None:
    """Allowlist entries naming a tool that does not exist.

    Harmless today, dangerous the moment somebody registers a tool with
    that name — the grant is already waiting for it.
    """
    for agent in graph.of_kind(NodeKind.AGENT):
        granted = set(agent.properties.get("allowed_tools") or [])
        linked = {
            graph.label(edge.target)
            for edge in graph.outgoing(
                agent.id, frozenset({EdgeKind.ALLOWED_TO_CALL})
            )
        }
        for missing in sorted(granted - linked):
            report.add(
                Finding(
                    check="dangling-tool-grant",
                    severity=Severity.LOW,
                    title=(
                        f"{agent.label!r} is allowed {missing!r}, which is not "
                        "registered"
                    ),
                    detail=(
                        "The allowlist grants a tool no registry provides. "
                        "policy.py denies the call today, but the grant "
                        "silently activates if a tool takes that name later."
                    ),
                    remediation=(
                        f"Remove {missing!r} from the agent's allowed_tools in "
                        "config/agents.yaml, or register the tool."
                    ),
                    nodes=(agent.id,),
                )
            )


def _capability_gaps(graph: Graph, report: Report) -> None:
    """Agents requiring a capability their profile does not grant.

    ``ModelRegistry.resolve`` raises on this at run time; the graph finds
    it before a run is paid for.
    """
    for agent in graph.of_kind(NodeKind.AGENT):
        required = set(agent.properties.get("required_capabilities") or [])
        profile_id = (
            f"{NodeKind.MODEL_PROFILE.value}:"
            f"{agent.properties.get('model_profile')}"
        )
        profile = graph.node(profile_id)
        if profile is None:
            report.add(
                Finding(
                    check="unknown-model-profile",
                    severity=Severity.HIGH,
                    title=(
                        f"{agent.label!r} references undefined profile "
                        f"{agent.properties.get('model_profile')!r}"
                    ),
                    detail="Every run of this agent fails at resolution.",
                    remediation="Define the profile in config/models.yaml.",
                    nodes=(agent.id,),
                )
            )
            continue

        missing = required - set(profile.properties.get("capabilities") or [])
        if not missing:
            continue

        report.add(
            Finding(
                check="capability-gap",
                severity=Severity.HIGH,
                title=(
                    f"{agent.label!r} requires {sorted(missing)}, which "
                    f"{profile.label!r} does not grant"
                ),
                detail=(
                    "ModelRegistry.resolve raises for this combination, so "
                    "the workflow fails partway through a paid run."
                ),
                remediation=(
                    "Add the capabilities to the profile, or point the agent "
                    "at a profile that has them."
                ),
                nodes=(agent.id, profile_id),
            )
        )


def _tool_grant_without_tool_calling(graph: Graph, report: Report) -> None:
    """A tool allowlist on an agent whose model cannot call tools.

    The grant is inert but misleading: the allowlist implies a control
    that is really just an unusable capability.
    """
    for agent in graph.of_kind(NodeKind.AGENT):
        if not agent.properties.get("allowed_tools"):
            continue
        profile_id = (
            f"{NodeKind.MODEL_PROFILE.value}:"
            f"{agent.properties.get('model_profile')}"
        )
        profile = graph.node(profile_id)
        if profile is None:
            continue
        if "tool_calling" in (profile.properties.get("capabilities") or []):
            continue

        report.add(
            Finding(
                check="tool-grant-without-capability",
                severity=Severity.LOW,
                title=(
                    f"{agent.label!r} holds tools but {profile.label!r} lacks "
                    "tool_calling"
                ),
                detail=(
                    "The allowlist cannot be exercised through this profile. "
                    "Swapping the profile later would activate it silently."
                ),
                remediation=(
                    "Drop the allowlist, or declare tool_calling on the "
                    "profile so the grant is honest about what it enables."
                ),
                nodes=(agent.id, profile_id),
            )
        )


def _orphaned_tools(graph: Graph, report: Report) -> None:
    """Registered tools no agent may call — reachable surface, unused."""
    for tool in graph.of_kind(NodeKind.TOOL):
        if tool.properties.get("observed_only"):
            continue
        if graph.incoming(tool.id, frozenset({EdgeKind.ALLOWED_TO_CALL})):
            continue

        report.add(
            Finding(
                check="orphaned-tool",
                severity=Severity.INFO,
                title=f"Tool {tool.label!r} is registered but unreachable",
                detail=(
                    "No agent's allowlist includes it, so it is loaded and "
                    "callable by the runtime yet granted to nobody."
                ),
                remediation=(
                    "Remove it from the registry if it is not needed; every "
                    "registered tool is one allowlist edit from being live."
                ),
                nodes=(tool.id,),
            )
        )


# --- Runtime overlay: permitted vs. observed ----------------------------


def _runtime_drift(graph: Graph, report: Report) -> None:
    """Tool calls that succeeded on an edge configuration does not grant.

    This should be impossible — policy.py checks the allowlist before
    execution — so a finding here means the graph and the enforcement
    point disagree, and one of them is wrong.
    """
    for edge in graph.edges:
        if edge.kind is not EdgeKind.CALLED:
            continue
        permitted = any(
            e.target == edge.target
            for e in graph.outgoing(
                edge.source, frozenset({EdgeKind.ALLOWED_TO_CALL})
            )
        )
        if permitted:
            continue

        report.add(
            Finding(
                check="runtime-drift",
                severity=Severity.CRITICAL,
                title=(
                    f"{graph.label(edge.source)!r} executed "
                    f"{graph.label(edge.target)!r} without a grant"
                ),
                detail=(
                    "The trace shows a completed tool call on an edge the "
                    "configuration does not permit. Either the collected "
                    "config differs from the one that ran, or the policy "
                    "check was bypassed."
                ),
                remediation=(
                    "Confirm the trace and config come from the same run, "
                    "then audit authorize_tool_call in orchestration/policy.py."
                ),
                nodes=(edge.source, edge.target),
            )
        )


def _approval_fatigue(graph: Graph, report: Report) -> None:
    """A human approval gate that was answered once and stopped asking.

    Every usable implementation of these gates offers "don't ask again",
    because prompting on every call is unworkable. Taking that option
    converts a per-call control into a per-session one: later calls
    execute unattended, and the trace still records them as approved.

    This matters most where it is least visible. The gate is still on the
    path in the graph, so a reviewer reading the diagram sees a control
    that is no longer doing the job the diagram implies.
    """
    for edge in graph.edges:
        if edge.kind is not EdgeKind.APPROVED:
            continue
        if edge.properties.get("scope") != "session":
            continue

        report.add(
            Finding(
                check="approval-fatigue",
                severity=Severity.HIGH,
                title=(
                    f"{graph.label(edge.target)!r} was approved for the whole "
                    f"run, not per call"
                ),
                detail=(
                    f"{graph.label(edge.source)!r} received a session-wide "
                    f"grant for {graph.label(edge.target)!r}, so every later "
                    "call executed without anyone seeing it. The approval "
                    "gate is still on the path, which makes the control look "
                    "stronger in review than it was in practice."
                ),
                remediation=(
                    "Scope grants to a single call, or bound them — per "
                    "argument, per count, or per elapsed time — so a tool "
                    "reachable from untrusted content cannot inherit one "
                    "answer for a whole run."
                ),
                nodes=(edge.source, edge.target),
            )
        )


def _observed_denials(graph: Graph, report: Report) -> None:
    """Policy denials seen in the trace — attempted edges.

    The direct analogue of failed logon events: proof a principal reached
    for something, and that the control held.
    """
    for edge in graph.edges:
        if edge.kind is not EdgeKind.DENIED:
            continue
        report.add(
            Finding(
                check="observed-denial",
                severity=Severity.INFO,
                title=(
                    f"{graph.label(edge.source)!r} was denied "
                    f"{graph.label(edge.target)!r}"
                ),
                detail=(
                    f"Policy refused the call: {edge.properties.get('reason', '')}"
                    " The control worked; the attempt is still worth knowing "
                    "about."
                ),
                remediation=(
                    "None required. Investigate if the agent had no legitimate "
                    "reason to reach for that tool."
                ),
                nodes=(edge.source, edge.target),
            )
        )
