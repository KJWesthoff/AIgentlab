"""BloodHound OpenGraph export.

OpenGraph is BloodHound CE's generic ingest format: instead of forcing
everything into ``User``/``Group``/``Computer``, it accepts custom node
and edge kinds and still gives you the real UI — pathfinding, the Cypher
console, saved queries. That matters here, because an audience reading
``Document -[CanInject]-> Agent`` learns something, while the same path
disguised as ``AdminTo`` teaches them a false analogy.

Schema (per BloodHound's OpenGraph docs):

    {"metadata": {"source_kind": "..."},
     "graph": {"nodes": [{"id", "kinds", "properties"}],
               "edges": [{"kind", "start", "end", "properties"}]}}

Two constraints the format imposes and this module respects: a node
carries at most three kinds, and the first one drives its icon in the
UI; properties must be flat primitives or arrays, never nested objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .analysis import Finding, Report, Severity
from .model import Graph

SOURCE_KIND = "AgentLab"
MAX_KINDS = 3

#: Second kind on every node, so the whole import is selectable in the UI
#: with `MATCH (n:AgentLab)` and easy to delete between demo runs.
COMMON_KIND = SOURCE_KIND

#: Third kind, applied only to nodes a finding touches, so a demo can open
#: with `MATCH (n:Tainted) RETURN n` instead of hunting through the graph.
TAINTED_KIND = "Tainted"

#: Property names BloodHound reserves on its base node schema. Setting
#: ``objectid`` on an OpenGraph node makes the whole upload fail
#: validation — confirmed by bisecting a rejected payload: identical data
#: with this key removed ingests fine. The node's ``id`` already carries
#: the identity, so nothing is lost by dropping it.
RESERVED_PROPERTIES = frozenset({"objectid"})


def to_opengraph(graph: Graph, report: Report | None = None) -> dict[str, Any]:
    """Render the graph as an OpenGraph payload.

    Findings are folded onto the nodes they implicate, so severity is
    visible in the UI's entity panel rather than living only in the CLI.
    """
    flagged = _findings_by_node(report)

    nodes = []
    for node in graph.nodes:
        kinds = [node.kind.value, COMMON_KIND]
        if node.id in flagged:
            kinds.append(TAINTED_KIND)

        properties: dict[str, Any] = {
            "name": node.label,
            "displayname": node.label,
            "agentlab_kind": node.kind.value,
        }
        properties.update(_flatten(node.properties))

        if node.id in flagged:
            findings = flagged[node.id]
            properties["finding_count"] = len(findings)
            properties["max_severity"] = _worst(findings).value
            properties["findings"] = [f.title for f in findings]
            properties["finding_checks"] = sorted({f.check for f in findings})
            # Same vocabulary as the threat-model slides, so a node's
            # entity panel names the boundary rather than only the check.
            properties["boundaries"] = sorted(
                {f.boundary.value for f in findings if f.boundary}
            )
            properties["root_causes"] = sorted(
                {f.root_cause for f in findings if f.root_cause}
            )
            properties["owasp"] = sorted(
                {entry for f in findings for entry in f.owasp}
            )

        nodes.append(
            {"id": node.id, "kinds": kinds[:MAX_KINDS], "properties": properties}
        )

    edges = [
        {
            "kind": edge.kind.value,
            "start": {"value": edge.source, "match_by": "id"},
            "end": {"value": edge.target, "match_by": "id"},
            "properties": _flatten(edge.properties),
        }
        for edge in graph.edges
    ]

    return {
        "metadata": {"source_kind": SOURCE_KIND},
        "graph": {"nodes": nodes, "edges": edges},
    }


def write_opengraph(
    graph: Graph, path: Path, report: Report | None = None
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_opengraph(graph, report), indent=2), encoding="utf-8"
    )
    return path


def _findings_by_node(report: Report | None) -> dict[str, list[Finding]]:
    flagged: dict[str, list[Finding]] = {}
    if report is None:
        return flagged
    for finding in report.findings:
        for node_id in finding.nodes:
            flagged.setdefault(node_id, []).append(finding)
    return flagged


def _worst(findings: list[Finding]) -> Severity:
    order = list(Severity)
    return min((f.severity for f in findings), key=order.index)


def _flatten(properties: dict[str, Any]) -> dict[str, Any]:
    """Coerce values into what OpenGraph accepts: primitives and arrays.

    Anything structured is stringified rather than dropped — a property
    that survives as text is still searchable in the UI.
    """
    flat: dict[str, Any] = {}
    for key, value in properties.items():
        if value is None or key in RESERVED_PROPERTIES:
            continue
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        elif isinstance(value, (list, tuple, set)):
            flat[key] = [
                v if isinstance(v, (str, int, float, bool)) else str(v)
                for v in value
            ]
        else:
            flat[key] = str(value)
    return flat
