"""``agentlab-graph`` — build, analyze and export the permission graph.

    agentlab-graph                          # findings for the current config
    agentlab-graph --trace-file run.jsonl   # overlay what a real run did
    agentlab-graph --export graph.json      # upload this to BloodHound CE
    agentlab-graph --cypher                 # queries to paste into its console

The analysis runs with no infrastructure at all, so it works in CI and in
tests; the export exists so the same graph can be explored in BloodHound's
UI when a person is in the room.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import Report, Severity, analyze
from .collect import MAX_DOCUMENTS, collect_runtime, collect_static
from .export import write_opengraph
from .model import Graph, NodeKind

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_MARKERS = {
    Severity.CRITICAL: "!!",
    Severity.HIGH: " !",
    Severity.MEDIUM: " ~",
    Severity.LOW: " -",
    Severity.INFO: " .",
}

#: Starting points for a demo. These are the OpenGraph equivalents of
#: BloodHound's pre-built queries, and each mirrors a check in analysis.py.
CYPHER_QUERIES = (
    (
        "Everything imported from agentlab",
        "MATCH (n:AgentLab) RETURN n",
    ),
    (
        "Nodes implicated in a finding",
        "MATCH (n:Tainted) RETURN n ORDER BY n.max_severity",
    ),
    (
        "Which agents can untrusted documents reach?",
        "MATCH p = (d:Document)-[:CanInject|Produces|FlowsTo|CanCoerce*1..]->"
        "(a:Agent) RETURN p",
    ),
    (
        "Shortest path from a document to any tool",
        "MATCH p = shortestPath((d:Document)-"
        "[:CanInject|Produces|FlowsTo|CanCoerce|AllowedToCall*1..]->(t:Tool)) "
        "RETURN p",
    ),
    (
        "Write-capable tools and what guards them",
        "MATCH (t:Tool) WHERE t.read_only = false "
        "OPTIONAL MATCH (t)-[:GuardedBy]->(g) RETURN t, g",
    ),
    (
        "Confused deputies: who can steer whose tools",
        "MATCH (a:Agent)-[:CanCoerce]->(b:Agent)-[:AllowedToCall]->(t:Tool) "
        "WHERE NOT (a)-[:AllowedToCall]->(t) RETURN a, b, t",
    ),
    (
        "Agents sharing one model (cross-checks that are not independent)",
        "MATCH (a:Agent)-[:RunsOn]->()-[:BackedBy]->(m:Model)<-[:BackedBy]-()"
        "<-[:RunsOn]-(b:Agent) WHERE a.name < b.name RETURN a, b, m",
    ),
    (
        "Denied at run time (requires --trace-file)",
        "MATCH p = (:Agent)-[:Denied]->(:Tool) RETURN p",
    ),
)


def build_graph(args: argparse.Namespace) -> Graph:
    graph = collect_static(
        config_dir=args.config_dir,
        corpus_dir=args.corpus_dir,
        max_documents=args.max_documents,
    )
    if args.trace_file:
        if not args.trace_file.is_file():
            raise SystemExit(f"Trace file not found: {args.trace_file}")
        collect_runtime(graph, args.trace_file)
    return graph


def print_summary(graph: Graph, report: Report, trace: Path | None) -> None:
    print("agentlab permission graph")
    print(f"  {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    counts = {
        kind.value: len(graph.of_kind(kind))
        for kind in NodeKind
        if graph.of_kind(kind)
    }
    print("  nodes: " + ", ".join(f"{k} {v}" for k, v in counts.items()))

    edge_counts: dict[str, int] = {}
    for edge in graph.edges:
        edge_counts[edge.kind.value] = edge_counts.get(edge.kind.value, 0) + 1
    print(
        "  edges: "
        + ", ".join(f"{k} {v}" for k, v in sorted(edge_counts.items()))
    )
    print(f"  runtime overlay: {trace if trace else 'none (static only)'}")
    print()

    findings = report.sorted()
    if not findings:
        print("No findings.")
        return

    tally = ", ".join(
        f"{report.count(s)} {s.value}" for s in Severity if report.count(s)
    )
    print(f"{len(findings)} findings — {tally}")
    print()

    for finding in findings:
        print(f"{_MARKERS[finding.severity]} [{finding.check}] {finding.title}")
        print(f"     {finding.detail}")
        if finding.path:
            print(f"     path: {finding.path}")
        print(f"     fix:  {finding.remediation}")
        print()


def report_to_dict(graph: Graph, report: Report) -> dict:
    return {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "findings": [
            {
                "check": f.check,
                "severity": f.severity.value,
                "title": f.title,
                "detail": f.detail,
                "remediation": f.remediation,
                "nodes": list(f.nodes),
                "path": f.path,
            }
            for f in report.sorted()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Map agentlab entities and permissions as a graph, find attack "
            "paths, and export for BloodHound CE."
        )
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=PROJECT_ROOT / "config",
        help="Directory containing models.yaml and agents.yaml.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus",
        help="Corpus the search tool may read (its documents are untrusted).",
    )
    parser.add_argument(
        "--trace-file",
        type=Path,
        default=None,
        help=(
            "Overlay a recorded run (JSON lines from --trace-file or --live) "
            "to add observed calls, denials and document reads."
        ),
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help=(
            "Write BloodHound OpenGraph JSON here, ready to upload to "
            "BloodHound CE."
        ),
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=MAX_DOCUMENTS,
        help=(
            "Cap document nodes per corpus, so a large corpus stays legible "
            "in the UI."
        ),
    )
    parser.add_argument(
        "--cypher",
        action="store_true",
        help="Print starter Cypher queries for BloodHound's console and exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit findings as JSON instead of text.",
    )
    parser.add_argument(
        "--fail-on",
        choices=[s.value for s in Severity] + ["never"],
        default="never",
        help=(
            "Exit non-zero when a finding at this severity or worse exists, "
            "for use as a CI gate."
        ),
    )
    args = parser.parse_args()

    if args.cypher:
        for title, query in CYPHER_QUERIES:
            print(f"// {title}")
            print(query)
            print()
        return

    if not args.config_dir.is_dir():
        raise SystemExit(f"Config directory not found: {args.config_dir}")

    graph = build_graph(args)
    report = analyze(graph)

    if args.json:
        print(json.dumps(report_to_dict(graph, report), indent=2))
    else:
        print_summary(graph, report, args.trace_file)

    if args.export:
        path = write_opengraph(graph, args.export, report)
        if not args.json:
            print(f"OpenGraph written to {path}")
            print(
                "Upload it in BloodHound CE under "
                "Administration → File Ingest."
            )

    if args.fail_on != "never":
        threshold = list(Severity).index(Severity(args.fail_on))
        worst = min(
            (list(Severity).index(f.severity) for f in report.findings),
            default=len(Severity),
        )
        if worst <= threshold:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
