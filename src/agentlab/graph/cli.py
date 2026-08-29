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

from dotenv import load_dotenv

from .analysis import Report, Severity, analyze, coverage
from .collect import MAX_DOCUMENTS, collect_runtime, collect_static
from .export import write_opengraph
from .bloodhound import TOKEN_ID_VARIABLE, TOKEN_KEY_VARIABLE
from .icons import ICONS, register_icons, write_icons
from .ingest import ingest_graph
from .queries import QUERIES, register_queries, write_queries
from .model import Graph, NodeKind

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_MARKERS = {
    Severity.CRITICAL: "!!",
    Severity.HIGH: " !",
    Severity.MEDIUM: " ~",
    Severity.LOW: " -",
    Severity.INFO: " .",
}

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
    for corpus in graph.of_kind(NodeKind.CORPUS):
        if corpus.properties.get("truncated"):
            print(
                f"  corpus {corpus.label!r}: showing "
                f"{len(graph.outgoing(corpus.id))} of "
                f"{corpus.properties.get('document_count')} documents "
                "(--max-documents)"
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
        if finding.boundary is not None:
            owasp = f" · {', '.join(finding.owasp)}" if finding.owasp else ""
            print(
                f"     boundary: {finding.boundary.value} — "
                f"{finding.root_cause}{owasp}"
            )
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
        "--ingest",
        metavar="URL",
        default=None,
        help=(
            "Build the graph and upload it straight to a running BloodHound "
            "CE, waiting for the ingest job to finish. Uses --export as the "
            "payload path when given. Needs the API token variables."
        ),
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "With --ingest, first delete the graph's existing AgentLab "
            "nodes so the upload replaces rather than accumulates. Scoped "
            "to this project's source kind — AD and Azure data are "
            "untouched."
        ),
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="With --ingest, return as soon as the job is queued.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=MAX_DOCUMENTS,
        help=(
            "Cap document nodes per corpus. Documents are interchangeable "
            "to the analysis, so the default keeps the graph legible; raise "
            "it to see every source in the UI."
        ),
    )
    parser.add_argument(
        "--export-icons",
        type=Path,
        default=None,
        help=(
            "Write the custom node-kind icon pack here, for POSTing to "
            "BloodHound's /api/v2/custom-nodes endpoint."
        ),
    )
    parser.add_argument(
        "--register-icons",
        metavar="URL",
        default=None,
        help=(
            "Register the icon pack directly with a running BloodHound CE "
            f"(e.g. http://127.0.0.1:8080). Needs {TOKEN_ID_VARIABLE} and "
            f"{TOKEN_KEY_VARIABLE} in the environment or .env."
        ),
    )
    parser.add_argument(
        "--export-queries",
        type=Path,
        default=None,
        help=(
            "Write the saved-query pack here as a ZIP. This is NOT graph "
            "data — it imports under Explore → Cypher, not File Ingest. "
            "Prefer --register-queries when you have API credentials."
        ),
    )
    parser.add_argument(
        "--register-queries",
        metavar="URL",
        default=None,
        help=(
            "Install the saved Cypher queries into a running BloodHound CE. "
            f"Needs {TOKEN_ID_VARIABLE} and {TOKEN_KEY_VARIABLE}."
        ),
    )
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help=(
            "With --register-queries, install only the numbered demo "
            "sequence. Combine with --prune-queries to remove the "
            "supporting queries from the sidebar."
        ),
    )
    parser.add_argument(
        "--prune-queries",
        action="store_true",
        help=(
            "With --register-queries, also delete 'agentlab:' queries no "
            "longer in the set — the residue a rename leaves behind. Only "
            "touches this project's prefix."
        ),
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help=(
            "Show which of the nine consolidated OWASP root causes the "
            "graph has checks for, and which it cannot speak to."
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

    if args.coverage:
        print("Root cause → boundary → agentlab checks\n")
        for name, boundary, checks in coverage():
            print(f"  {name}")
            print(f"    boundary: {boundary.value}")
            print(
                "    checks:   "
                + (", ".join(checks) if checks else "— none (not modeled)")
            )
            print()
        return

    if args.cypher:
        for saved in QUERIES:
            print(f"// {saved.name}")
            print(f"//   {saved.description}")
            print(saved.query)
            print()
        return

    # Icon work is independent of the graph, so it runs before collection
    # and can be the only thing a given invocation does.
    if args.export_icons:
        print(f"Icon pack written to {write_icons(args.export_icons)}")
    if args.register_icons:
        load_dotenv(PROJECT_ROOT / ".env")
        result = register_icons(args.register_icons)
        print(
            f"Icons at {args.register_icons}: "
            f"{len(result.created)} created, {len(result.updated)} updated "
            f"({len(ICONS)} node kinds). Reload the BloodHound tab — the UI "
            "caches icon definitions."
        )
    if args.export_queries:
        print(f"Query pack written to {write_queries(args.export_queries)}")
        print(
            "  Import under Explore → Cypher — this is saved queries, not "
            "graph data, and File Ingest will reject it."
        )
    if args.register_queries:
        load_dotenv(PROJECT_ROOT / ".env")
        result = register_queries(
            args.register_queries,
            prune=args.prune_queries,
            demo_only=args.demo_only,
        )
        print(
            f"Saved queries at {args.register_queries}: "
            f"{len(result.created)} created, {len(result.updated)} updated "
            f"({len(QUERIES)} available). Find them under Explore → Cypher."
        )
        for name in result.removed:
            print(f"  removed stale query: {name}")

    side_effects = (
        args.export_icons
        or args.register_icons
        or args.export_queries
        or args.register_queries
    )
    if side_effects and not args.export:
        return

    if not args.config_dir.is_dir():
        raise SystemExit(f"Config directory not found: {args.config_dir}")

    graph = build_graph(args)
    report = analyze(graph)

    if args.json:
        print(json.dumps(report_to_dict(graph, report), indent=2))
    else:
        print_summary(graph, report, args.trace_file)

    if args.export or args.ingest:
        # --ingest without --export still needs a file to send; keep it
        # beside the other generated artifacts rather than in a temp dir,
        # so a failed upload leaves something to inspect and retry.
        destination = args.export or (
            PROJECT_ROOT / "data" / "agentlab-opengraph.json"
        )
        path = write_opengraph(graph, destination, report)
        if not args.json:
            print(f"OpenGraph written to {path}")
            if not args.ingest:
                print(
                    "  Upload under Administration → File Ingest (graph data)."
                )

    if args.ingest:
        load_dotenv(PROJECT_ROOT / ".env")
        result = ingest_graph(
            args.ingest,
            path,
            wait=not args.no_wait,
            replace=args.replace,
        )
        if not args.json:
            if result.cleared:
                print("  Cleared the existing AgentLab nodes first.")
            print(
                f"Ingested into {args.ingest} as job {result.job_id} — "
                f"{result.status_name}."
            )
            if result.waited and not result.succeeded:
                print(
                    "  Job did not report success; check Administration → "
                    "File Ingest for its error detail."
                )
            elif result.succeeded:
                if result.cleared:
                    # Clearing deletes and recreates every node, so the
                    # graph ids an open Explore view is holding no longer
                    # resolve. It reads as data that went missing rather
                    # than a stale tab, and the data is fine.
                    print(
                        "  Reload the BloodHound tab — --replace gave every "
                        "node a new id, and an open Explore view still "
                        "holds the old ones."
                    )
                print(
                    "  Query it under Explore → Cypher: "
                    "MATCH (n:AgentLab) RETURN n"
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
