"""Saved Cypher queries for BloodHound CE.

BloodHound's Explore view opens empty — custom kinds are invisible to the
default views, so without saved queries a freshly ingested graph looks
like nothing was uploaded. These are the equivalent of BloodHound's
pre-built AD queries, and each one mirrors a check in analysis.py so the
CLI findings and the UI tell the same story.

They are ordered as a demo runs: the overview first, then the taint
story, then the composed-permission failures, then hygiene and runtime
evidence.

Note for BloodHound CE: structured-graph *findings*, remediation text
and risk metrics are Enterprise features. In CE the registerable schema
surface is node kinds with icons (see icons.py) plus these queries, so
this module is what makes an ingested graph legible.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .bloodhound import BloodHoundClient

QUERY_ENDPOINT = "/api/v2/saved-queries"

#: Prefix on every saved query, so an operator can tell at a glance which
#: queries this project installed and delete them as a set.
PREFIX = "agentlab"


@dataclass(frozen=True)
class SavedQuery:
    name: str
    description: str
    query: str

    @property
    def full_name(self) -> str:
        return f"{PREFIX}: {self.name}"


#: The subgraph worth looking at. Excludes the model/profile/provider
#: plumbing, which is real but says nothing about attack paths, and would
#: otherwise dominate the picture.
SECURITY_EDGES = "CanInject|CanCoerce|AllowedToCall|Produces|FlowsTo|GuardedBy"

QUERIES: tuple[SavedQuery, ...] = (
    SavedQuery(
        "Overview — the security-relevant graph",
        "Start here. Agents, tools, documents and artifacts with the edges "
        "that matter, leaving out model/provider plumbing.",
        f"MATCH p = (:AgentLab)-[:{SECURITY_EDGES}]->(:AgentLab) RETURN p",
    ),
    SavedQuery(
        "Everything imported from agentlab",
        "The whole import, plumbing included. Useful to confirm an ingest "
        "landed, and to delete the set.",
        "MATCH (n:AgentLab) RETURN n",
    ),
    SavedQuery(
        "Nodes implicated in a finding",
        "Every node the analyzer flagged, worst first. The demo can open "
        "here instead of hunting through the graph.",
        "MATCH (n:Tainted) RETURN n ORDER BY n.max_severity",
    ),
    # --- The taint story ---
    SavedQuery(
        "Which agents can untrusted documents reach?",
        "Every hand-off is an egress and an ingress: only one agent reads "
        "the corpus, but its artifacts land in the others looking internal "
        "when they are not (root cause 1, missing ingress boundary).",
        "MATCH p = (d:Document)-[:CanInject|Produces|FlowsTo|CanCoerce*1..]->"
        "(a:Agent) RETURN p",
    ),
    SavedQuery(
        "Injection-reachable agents that read no documents",
        "Allowlists look clean, but a peer message is a prompt — the "
        "sender built it from untrusted context, so it arrives internal "
        "but is not.",
        "MATCH (a:Agent) WHERE NOT (:Document)-[:CanInject]->(a) "
        "AND (:Document)-[:CanInject|Produces|FlowsTo|CanCoerce*1..]->(a) "
        "RETURN a",
    ),
    SavedQuery(
        "Shortest path from a document to any tool",
        "This system's 'shortest path to Domain Admin'.",
        "MATCH p = shortestPath((d:Document)-"
        "[:CanInject|Produces|FlowsTo|CanCoerce|AllowedToCall*1..]->(t:Tool)) "
        "RETURN p",
    ),
    SavedQuery(
        "Untrusted content reaching a write-capable tool",
        "The critical case. Empty is the correct result — it fires once a "
        "tool that can change state enters an allowlist.",
        "MATCH p = shortestPath((d:Document)-"
        "[:CanInject|Produces|FlowsTo|CanCoerce|AllowedToCall*1..]->(t:Tool)) "
        "WHERE t.read_only = false RETURN p",
    ),
    # --- Composed permissions ---
    SavedQuery(
        "Confused deputies: who can steer whose tools",
        "Approval is a union, not a chain — each gate approves only its "
        "own hop, so a request one gate would refuse is laundered through "
        "a peer (T14). The nested-group problem, in agent form.",
        # RETURN p, not the nodes: BloodHound only draws relationships
        # that are part of a returned path, and the connection is the
        # entire point of this one.
        "MATCH p = (a:Agent)-[:CanCoerce]->(b:Agent)-[:AllowedToCall]->"
        "(t:Tool) WHERE NOT (a)-[:AllowedToCall]->(t) RETURN p",
    ),
    SavedQuery(
        "Real permission surface — every reachable agent's tools",
        "Approval is a union, not a chain: the real permission surface is "
        "the union of every reachable agent's tools, not what any one "
        "allowlist says.",
        "MATCH (t:Tool)<-[:AllowedToCall]-(holder:Agent) "
        "OPTIONAL MATCH (other:Agent)-[:CanCoerce]->(holder) "
        "RETURN t.name AS tool, holder.name AS holder, "
        "count(DISTINCT other) AS reachable_by",
    ),
    SavedQuery(
        "Write-capable tools and what guards them",
        "The permission gate from the boundary model, on the path. Root "
        "cause 2 is what its absence looks like.",
        "MATCH p = (:Agent)-[:AllowedToCall]->(t:Tool)-[:GuardedBy]->"
        "(:ApprovalGate) WHERE t.read_only = false RETURN p",
    ),
    SavedQuery(
        "Write-capable tools with NO approval gate",
        "The dangerous shape, and the companion to the query above: a "
        "path query can only show tools that HAVE a gate, so absence "
        "there would read as safety. Empty is the correct result here.",
        "MATCH (t:Tool) WHERE t.read_only = false "
        "AND NOT (t)-[:GuardedBy]->() RETURN t",
    ),
    SavedQuery(
        "Agents sharing one model",
        "Matters where two of them are meant to check each other — a "
        "reviewer running the writer's model reproduces its blind spots, "
        "so the check reports success it did not earn.",
        # Two MATCH clauses on purpose: a single pattern cannot traverse
        # the same BackedBy relationship twice, so agents sharing one
        # *profile* — the common case — never matched.
        "MATCH p1 = (a:Agent)-[:RunsOn]->(:ModelProfile)-[:BackedBy]->"
        "(m:Model) "
        "MATCH p2 = (b:Agent)-[:RunsOn]->(:ModelProfile)-[:BackedBy]->(m) "
        "WHERE a.name < b.name RETURN p1, p2",
    ),
    # --- Hygiene ---
    SavedQuery(
        "Tools nobody may call",
        "Registered and loadable, granted to no one — one allowlist edit "
        "from being live.",
        "MATCH (t:Tool) WHERE NOT (:Agent)-[:AllowedToCall]->(t) RETURN t",
    ),
    SavedQuery(
        "Agent inventory — tools, model and budget",
        "One row per agent: what it may call, what it runs on, how many "
        "calls it gets.",
        "MATCH (a:Agent)-[:RunsOn]->(p:ModelProfile)-[:BackedBy]->(m:Model) "
        "RETURN a.name AS agent, a.allowed_tools AS tools, "
        "m.name AS model, a.max_calls AS max_calls ORDER BY agent",
    ),
    # --- Runtime evidence ---
    SavedQuery(
        "Observed at run time — calls and denials",
        "What a real run actually did, versus what config permits. "
        "Requires an export made with --trace-file.",
        "MATCH p = (:Agent)-[:Called|Denied]->(:Tool) RETURN p",
    ),
    SavedQuery(
        "Approval gates that stopped asking",
        "Root cause 8, Oversight & Alert Fatigue (T10): the permission "
        "gate runs on every iteration of the agent loop, but a "
        "session-wide grant answers it once. Requires --trace-file.",
        "MATCH p = (:Agent)-[r:Approved]->(:Tool) "
        "WHERE r.scope = 'session' RETURN p",
    ),
    SavedQuery(
        "Permitted but never used",
        "Grants a real run never exercised — the least-privilege backlog. "
        "Requires an export made with --trace-file.",
        "MATCH p = (a:Agent)-[:AllowedToCall]->(t:Tool) "
        "WHERE NOT (a)-[:Called]->(t) RETURN p",
    ),
)


#: The order the README's demo section walks through. Named here so a
#: renamed query breaks a test rather than the presentation.
DEMO_ORDER: tuple[str, ...] = (
    "Overview — the security-relevant graph",
    "Which agents can untrusted documents reach?",
    "Confused deputies: who can steer whose tools",
    "Shortest path from a document to any tool",
    "Write-capable tools and what guards them",
    "Approval gates that stopped asking",
)


@dataclass(frozen=True)
class QueryRegistration:
    created: tuple[str, ...]
    updated: tuple[str, ...]
    removed: tuple[str, ...] = ()


def write_queries(path: Path) -> Path:
    """Write a ZIP that BloodHound's saved-query import accepts.

    The import format is one JSON file per query, named for the query, so
    this mirrors exactly what ``GET /api/v2/saved-queries/export`` emits.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for saved in QUERIES:
            safe = saved.full_name.replace("/", "-").replace(":", "")
            archive.writestr(
                f"{safe}.json",
                json.dumps(
                    {
                        "name": saved.full_name,
                        "query": saved.query,
                        "description": saved.description,
                    },
                    indent=2,
                ),
            )
    return path


def register_queries(
    base_url: str,
    client: BloodHoundClient | None = None,
    prune: bool = False,
) -> QueryRegistration:
    """Create or refresh every saved query on a running BloodHound.

    Idempotent by name: queries this project owns are updated in place
    rather than duplicated, so re-running never leaves a second copy in
    the operator's sidebar.

    ``prune`` additionally deletes ``agentlab:`` queries that are no
    longer in the set — the residue a rename leaves behind, since the new
    name is created while the old one stays. Opt-in, because deleting is
    not something a registration should do behind your back, and scoped
    to the prefix so queries the operator wrote are never candidates.
    """
    client = client or BloodHoundClient.from_environment(base_url)

    listing = client.request("GET", QUERY_ENDPOINT) or {}
    existing = {
        entry.get("name"): entry.get("id")
        for entry in (listing.get("data") or [])
        if isinstance(entry, dict)
    }

    created: list[str] = []
    updated: list[str] = []

    for saved in QUERIES:
        payload = {
            "name": saved.full_name,
            "query": saved.query,
            "description": saved.description,
        }
        identifier = existing.get(saved.full_name)
        if identifier is None:
            client.request("POST", QUERY_ENDPOINT, payload)
            created.append(saved.full_name)
        else:
            client.request(
                "PUT", f"{QUERY_ENDPOINT}/{identifier}", payload
            )
            updated.append(saved.full_name)

    removed: list[str] = []
    if prune:
        wanted = {saved.full_name for saved in QUERIES}
        for name, identifier in existing.items():
            if not str(name).startswith(f"{PREFIX}:") or name in wanted:
                continue
            client.request("DELETE", f"{QUERY_ENDPOINT}/{identifier}")
            removed.append(name)

    return QueryRegistration(
        tuple(created), tuple(updated), tuple(sorted(removed))
    )
