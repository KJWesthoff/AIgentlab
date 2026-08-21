"""Permission-graph tests.

Two halves. The first asserts the collectors describe the *real* shipped
configuration correctly, so the graph stays honest as config changes.
The second builds deliberately broken configurations in a temp directory
and asserts each analyzer check fires — the shipped config is sound, so
these dangerous shapes have to be constructed to be tested.
"""

import datetime
import json
import re
import zipfile
from pathlib import Path

import httpx
import pytest
import yaml
from pydantic import BaseModel

from agentlab.agents.definitions import load_agents
from agentlab.graph.analysis import Severity, analyze
from agentlab.graph.bloodhound import (
    TOKEN_ID_VARIABLE,
    TOKEN_KEY_VARIABLE,
    sign_request,
)
from agentlab.graph.collect import PIPELINE, collect_runtime, collect_static
from agentlab.graph.export import MAX_KINDS, RESERVED_PROPERTIES, to_opengraph
from agentlab.graph import ingest as ingest_module
from agentlab.graph.icons import ICONS, register_icons, write_icons
from agentlab.graph.ingest import ingest_graph
from agentlab.graph.model import TAINT_EDGES, EdgeKind, Graph, NodeKind
from agentlab.graph.queries import (
    PREFIX,
    QUERIES,
    register_queries,
    write_queries,
)
from agentlab.tools.definitions import Tool, ToolDefinition

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"

AGENT = NodeKind.AGENT.value
TOOL = NodeKind.TOOL.value


class NoInput(BaseModel):
    pass


def make_tool(
    name: str,
    *,
    read_only: bool = True,
    reads: list[str] | None = None,
    writes: list[str] | None = None,
) -> Tool:
    return Tool(
        ToolDefinition(
            name=name,
            description="test",
            read_only=read_only,
            reads=reads or [],
            writes=writes or [],
        ),
        NoInput,
        lambda: None,
    )


@pytest.fixture
def real_graph() -> Graph:
    return collect_static(config_dir=CONFIG_DIR, corpus_dir=CORPUS_DIR)


def write_config(
    directory: Path, agents: dict, profiles: dict | None = None
) -> Path:
    """A config directory holding only what a test needs."""
    directory.mkdir(parents=True, exist_ok=True)
    base = {
        "economical": {
            "provider": "openrouter",
            "model": "vendor/small",
            "capabilities": ["text", "structured_output"],
        },
        "researcher": {
            "provider": "openrouter",
            "model": "vendor/large",
            "capabilities": ["text", "tool_calling", "structured_output"],
        },
    }
    (directory / "models.yaml").write_text(
        yaml.safe_dump({"profiles": profiles or base})
    )
    (directory / "agents.yaml").write_text(yaml.safe_dump({"agents": agents}))
    return directory


def agent_entry(**overrides) -> dict:
    entry = {
        "description": "test",
        "model_profile": "economical",
        "system_prompt": "test",
        "allowed_tools": [],
        "required_capabilities": ["text"],
    }
    entry.update(overrides)
    return entry


# --- The shipped configuration ------------------------------------------


def test_collects_every_configured_agent(real_graph):
    names = {node.label for node in real_graph.of_kind(NodeKind.AGENT)}
    assert names == set(load_agents(CONFIG_DIR / "agents.yaml"))


def test_pipeline_declaration_matches_configured_agents():
    """Guards the one hand-maintained part of the collector.

    PIPELINE mirrors Workflow.execute, which is plain Python and cannot be
    introspected. If an agent is added to agents.yaml without a stage, the
    artifact-flow edges would silently go missing, so fail here instead.
    """
    configured = set(load_agents(CONFIG_DIR / "agents.yaml"))
    assert {stage.agent for stage in PIPELINE} == configured

    produced = {stage.produces for stage in PIPELINE}
    consumed = {a for stage in PIPELINE for a in stage.consumes}
    assert consumed <= produced


def test_agent_is_linked_to_its_profile_model_and_provider(real_graph):
    path = real_graph.shortest_path(
        f"{AGENT}:researcher",
        f"{NodeKind.PROVIDER.value}:openrouter",
        frozenset({EdgeKind.RUNS_ON, EdgeKind.BACKED_BY, EdgeKind.SERVED_BY}),
    )
    assert path is not None
    assert [edge.kind for edge in path] == [
        EdgeKind.RUNS_ON,
        EdgeKind.BACKED_BY,
        EdgeKind.SERVED_BY,
    ]


def test_only_the_researcher_may_call_the_search_tool(real_graph):
    callers = {
        real_graph.label(edge.source)
        for edge in real_graph.incoming(
            f"{TOOL}:search_documents",
            frozenset({EdgeKind.ALLOWED_TO_CALL}),
        )
    }
    assert callers == {"researcher"}


def test_untrusted_documents_reach_every_agent(real_graph):
    """The finding the whole exercise exists to surface.

    Only the researcher reads the corpus, but taint propagates through
    artifacts to all four agents — invisible in any per-agent review.
    """
    document = real_graph.of_kind(NodeKind.DOCUMENT)[0]
    reached = {
        real_graph.node(node_id).label
        for node_id in real_graph.reachable_from(document.id, TAINT_EDGES)
        if real_graph.node(node_id).kind is NodeKind.AGENT
    }
    assert reached == {"researcher", "analyst", "writer", "reviewer"}


def test_shipped_config_findings_are_the_expected_ones(real_graph):
    """The shipped config deliberately contains the demo attack path.

    Granting the writer save_report is what makes the critical check
    real, so "no high findings" is the wrong assertion — this pins the
    exact set instead, and fails if a config change adds or drops one.
    """
    checks = {
        f.check
        for f in analyze(real_graph).findings
        if f.severity in (Severity.CRITICAL, Severity.HIGH)
    }
    assert checks == {"untrusted-to-write-tool", "indirect-injection-reach"}


def test_untrusted_content_reaches_the_real_write_tool(real_graph):
    """The headline path, on the shipped config rather than a fixture."""
    findings = [
        f
        for f in analyze(real_graph).findings
        if f.check == "untrusted-to-write-tool"
    ]
    # One per write-capable tool, not one per document.
    assert len(findings) == 1
    assert findings[0].severity is Severity.HIGH  # gated on a human
    assert findings[0].path.endswith(
        "writer -[AllowedToCall]-> save_report"
    )


def test_the_write_tool_is_gated_on_a_human(real_graph):
    gate = real_graph.outgoing(
        f"{TOOL}:save_report", frozenset({EdgeKind.GUARDED_BY})
    )
    assert [real_graph.node(e.target).kind for e in gate] == [
        NodeKind.APPROVAL_GATE
    ]


def test_downstream_agents_are_flagged_as_indirectly_reachable(real_graph):
    flagged = {
        f.title.split("'")[1]
        for f in analyze(real_graph).findings
        if f.check == "indirect-injection-reach"
    }
    assert flagged == {"analyst", "writer", "reviewer"}
    assert "researcher" not in flagged


# --- Broken configurations: each check fires ----------------------------


def test_untrusted_content_reaching_an_ungated_write_tool_is_critical():
    """The "path to Domain Admin" case, with nothing guarding the sink.

    Built by hand on purpose: collect_static always attaches an approval
    gate to a write-capable tool, mirroring policy.py, so this shape
    cannot currently arise from configuration. The check still has to
    handle it, because the gate is one edit away from being optional.
    """
    graph = Graph()
    graph.add_node(f"{NodeKind.DOCUMENT.value}:evil.md", NodeKind.DOCUMENT, "evil.md")
    graph.add_node(f"{AGENT}:researcher", NodeKind.AGENT, "researcher")
    graph.add_node(f"{AGENT}:writer", NodeKind.AGENT, "writer")
    graph.add_node(f"{TOOL}:publish", NodeKind.TOOL, "publish", read_only=False)

    graph.add_edge(
        f"{NodeKind.DOCUMENT.value}:evil.md",
        f"{AGENT}:researcher",
        EdgeKind.CAN_INJECT,
    )
    graph.add_edge(f"{AGENT}:researcher", f"{AGENT}:writer", EdgeKind.CAN_COERCE)
    graph.add_edge(f"{AGENT}:writer", f"{TOOL}:publish", EdgeKind.ALLOWED_TO_CALL)

    findings = [
        f for f in analyze(graph).findings if f.check == "untrusted-to-write-tool"
    ]
    assert len(findings) == 1
    assert findings[0].severity is Severity.CRITICAL
    assert findings[0].path == (
        "evil.md -[CanInject]-> researcher -[CanCoerce]-> writer "
        "-[AllowedToCall]-> publish"
    )


def test_write_tool_behind_an_approval_gate_is_downgraded(tmp_path):
    config = write_config(
        tmp_path / "config",
        {
            "researcher": agent_entry(
                model_profile="researcher",
                allowed_tools=["search_documents", "publish"],
                required_capabilities=["text", "tool_calling"],
            )
        },
    )
    graph = collect_static(
        config_dir=config,
        corpus_dir=CORPUS_DIR,
        tools={
            "search_documents": make_tool("search_documents", reads=["corpus"]),
            "publish": make_tool("publish", read_only=False, writes=["corpus"]),
        },
    )

    findings = [
        f for f in analyze(graph).findings if f.check == "untrusted-to-write-tool"
    ]
    assert findings
    assert findings[0].severity is Severity.HIGH
    assert graph.outgoing(
        f"{TOOL}:publish", frozenset({EdgeKind.GUARDED_BY})
    )


def test_confused_deputy_when_an_upstream_agent_steers_a_held_tool(tmp_path):
    """researcher holds no tool but its artifact steers one that does."""
    config = write_config(
        tmp_path / "config",
        {
            "researcher": agent_entry(),
            "analyst": agent_entry(),
            "writer": agent_entry(allowed_tools=["publish"]),
            "reviewer": agent_entry(),
        },
    )
    graph = collect_static(
        config_dir=config,
        corpus_dir=CORPUS_DIR,
        tools={"publish": make_tool("publish")},
    )

    findings = [f for f in analyze(graph).findings if f.check == "confused-deputy"]
    steered = {(f.nodes[0], f.nodes[1]) for f in findings}
    assert (f"{AGENT}:researcher", f"{AGENT}:writer") in steered
    assert (f"{AGENT}:analyst", f"{AGENT}:writer") in steered
    # The writer holds the tool itself, so it is not its own deputy.
    assert (f"{AGENT}:writer", f"{AGENT}:writer") not in steered


def test_dangling_tool_grant_is_reported(tmp_path):
    config = write_config(
        tmp_path / "config",
        {"researcher": agent_entry(allowed_tools=["shell_execute"])},
    )
    graph = collect_static(
        config_dir=config, corpus_dir=CORPUS_DIR, tools={}
    )

    findings = [
        f for f in analyze(graph).findings if f.check == "dangling-tool-grant"
    ]
    assert len(findings) == 1
    assert "shell_execute" in findings[0].title
    # A grant to a non-existent tool must not invent the node.
    assert not graph.has_node(f"{TOOL}:shell_execute")


def test_capability_gap_is_reported(tmp_path):
    config = write_config(
        tmp_path / "config",
        {
            "researcher": agent_entry(
                required_capabilities=["text", "tool_calling"]
            )
        },
    )
    graph = collect_static(config_dir=config, corpus_dir=CORPUS_DIR, tools={})

    findings = [f for f in analyze(graph).findings if f.check == "capability-gap"]
    assert len(findings) == 1
    assert "tool_calling" in findings[0].title


def test_shared_model_breaks_crosscheck_independence(tmp_path):
    config = write_config(
        tmp_path / "config",
        {
            "researcher": agent_entry(),
            "analyst": agent_entry(),
            "writer": agent_entry(),
            "reviewer": agent_entry(),
        },
    )
    graph = collect_static(config_dir=config, corpus_dir=CORPUS_DIR, tools={})

    findings = [
        f
        for f in analyze(graph).findings
        if f.check == "crosscheck-not-independent"
    ]
    # writer and reviewer both sit on the "economical" profile here.
    assert any("writer" in f.title and "reviewer" in f.title for f in findings)


def test_orphaned_tool_is_reported(tmp_path):
    config = write_config(
        tmp_path / "config", {"researcher": agent_entry()}
    )
    graph = collect_static(
        config_dir=config,
        corpus_dir=CORPUS_DIR,
        tools={"publish": make_tool("publish")},
    )

    findings = [f for f in analyze(graph).findings if f.check == "orphaned-tool"]
    assert [f.nodes for f in findings] == [(f"{TOOL}:publish",)]


# --- Runtime overlay -----------------------------------------------------


def write_trace(path: Path, events: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


def test_runtime_overlay_records_calls_denials_and_documents(
    real_graph, tmp_path
):
    document = real_graph.of_kind(NodeKind.DOCUMENT)[0].label
    trace = write_trace(
        tmp_path / "run.jsonl",
        [
            {
                "event": "tool_result",
                "agent": "researcher",
                "tool": "search_documents",
                "result": json.dumps({"results": [{"document": document}]}),
            },
            {
                "event": "policy_decision",
                "agent": "writer",
                "tool": "shell_execute",
                "allowed": False,
                "reason": "not allowed for agent",
            },
        ],
    )
    collect_runtime(real_graph, trace)

    assert real_graph.outgoing(
        f"{AGENT}:researcher", frozenset({EdgeKind.CALLED})
    )
    denials = real_graph.outgoing(
        f"{AGENT}:writer", frozenset({EdgeKind.DENIED})
    )
    assert [real_graph.label(e.target) for e in denials] == ["shell_execute"]

    observed = [
        e
        for e in real_graph.incoming(
            f"{AGENT}:researcher", frozenset({EdgeKind.CAN_INJECT})
        )
        if e.properties.get("observed")
    ]
    assert observed

    checks = {f.check for f in analyze(real_graph).findings}
    assert "observed-denial" in checks


def test_runtime_overlay_tolerates_a_partial_final_line(real_graph, tmp_path):
    """A trace can be read while the run writing it is still going."""
    trace = tmp_path / "run.jsonl"
    trace.write_text(
        json.dumps(
            {
                "event": "tool_result",
                "agent": "researcher",
                "tool": "search_documents",
                "result": "{}",
            }
        )
        + '\n{"event": "tool_res'
    )
    collect_runtime(real_graph, trace)

    assert real_graph.outgoing(
        f"{AGENT}:researcher", frozenset({EdgeKind.CALLED})
    )


def test_a_session_wide_approval_is_reported_as_fatigue(real_graph, tmp_path):
    """The gate stayed on the path but stopped being a per-call control."""
    trace = write_trace(
        tmp_path / "run.jsonl",
        [
            {
                "event": "approval_decision",
                "agent": "writer",
                "tool": "save_report",
                "approved": True,
                "scope": "session",
            }
        ],
    )
    collect_runtime(real_graph, trace)

    findings = [
        f for f in analyze(real_graph).findings if f.check == "approval-fatigue"
    ]
    assert len(findings) == 1
    assert findings[0].severity is Severity.HIGH
    assert findings[0].nodes == (f"{AGENT}:writer", f"{TOOL}:save_report")


def test_a_per_call_approval_is_not_reported_as_fatigue(real_graph, tmp_path):
    """Answering every prompt is the gate working — not a finding."""
    trace = write_trace(
        tmp_path / "run.jsonl",
        [
            {
                "event": "approval_decision",
                "agent": "writer",
                "tool": "save_report",
                "approved": True,
                "scope": "once",
            }
        ],
    )
    collect_runtime(real_graph, trace)

    assert not [
        f for f in analyze(real_graph).findings if f.check == "approval-fatigue"
    ]


def test_a_call_without_a_grant_is_critical_drift(real_graph, tmp_path):
    """Should be impossible — policy.py checks first — so it is critical."""
    trace = write_trace(
        tmp_path / "run.jsonl",
        [
            {
                "event": "tool_result",
                "agent": "writer",
                "tool": "search_documents",
                "result": "{}",
            }
        ],
    )
    collect_runtime(real_graph, trace)

    findings = [f for f in analyze(real_graph).findings if f.check == "runtime-drift"]
    assert len(findings) == 1
    assert findings[0].severity is Severity.CRITICAL


# --- BloodHound OpenGraph export ----------------------------------------


def test_export_matches_the_opengraph_schema(real_graph):
    payload = to_opengraph(real_graph, analyze(real_graph))
    assert set(payload["graph"]) == {"nodes", "edges"}
    assert payload["metadata"]["source_kind"] == "AgentLab"

    ids = {node["id"] for node in payload["graph"]["nodes"]}
    assert len(ids) == len(payload["graph"]["nodes"])

    for node in payload["graph"]["nodes"]:
        assert node["kinds"] and len(node["kinds"]) <= MAX_KINDS
        assert node["properties"]["name"]
        for value in node["properties"].values():
            assert isinstance(value, (str, int, float, bool, list))

    for edge in payload["graph"]["edges"]:
        assert edge["kind"]
        assert edge["start"]["match_by"] == "id"
        # Every endpoint must resolve, or BloodHound drops the edge.
        assert edge["start"]["value"] in ids
        assert edge["end"]["value"] in ids


def test_export_omits_properties_bloodhound_reserves(real_graph):
    """Regression: `objectid` made BloodHound reject the whole upload.

    Bisected against a live BloodHound CE instance — the identical payload
    with this key removed ingested fine. A node's `id` already carries its
    identity, so the property was redundant as well as fatal.
    """
    payload = to_opengraph(real_graph, analyze(real_graph))
    for node in payload["graph"]["nodes"]:
        assert not RESERVED_PROPERTIES & set(node["properties"])


def test_export_drops_reserved_names_coming_from_a_collector(real_graph):
    """The guard holds even if a collector sets the property itself."""
    real_graph.add_node(
        f"{TOOL}:probe", NodeKind.TOOL, "probe", objectid="S-1-5-21"
    )
    payload = to_opengraph(real_graph)
    probe = next(
        n for n in payload["graph"]["nodes"] if n["properties"]["name"] == "probe"
    )
    assert "objectid" not in probe["properties"]


def test_export_marks_flagged_nodes_so_the_ui_can_select_them(real_graph):
    report = analyze(real_graph)
    payload = to_opengraph(real_graph, report)
    flagged = {
        node["properties"]["name"]
        for node in payload["graph"]["nodes"]
        if "Tainted" in node["kinds"]
    }
    assert "analyst" in flagged
    assert all(
        node["properties"].get("max_severity")
        for node in payload["graph"]["nodes"]
        if "Tainted" in node["kinds"]
    )


def test_every_node_kind_has_an_icon():
    """A kind with no icon renders as an anonymous default glyph."""
    assert set(ICONS) == set(NodeKind)


def test_icon_payload_matches_the_custom_nodes_api(tmp_path):
    payload = json.loads(write_icons(tmp_path / "icons.json").read_text())
    assert set(payload) == {"custom_types"}

    for kind, entry in payload["custom_types"].items():
        assert kind in {k.value for k in NodeKind}
        icon = entry["icon"]
        assert icon["type"] == "font-awesome"
        # BloodHound wants the bare Font Awesome name, no fa-/fas- prefix.
        assert not icon["name"].startswith(("fa-", "fas-"))
        assert re.fullmatch(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})", icon["color"])


def test_icons_carry_the_trust_story(real_graph):
    """Untrusted content, privilege and controls must not share a colour.

    The palette is load-bearing: a rendered path should read warm →
    gold, with green marking a control in the way.
    """
    untrusted = ICONS[NodeKind.DOCUMENT].color
    privilege = ICONS[NodeKind.TOOL].color
    control = ICONS[NodeKind.APPROVAL_GATE].color
    assert len({untrusted, privilege, control}) == 3


def test_registering_icons_without_a_token_fails_loudly(monkeypatch):
    monkeypatch.delenv(TOKEN_ID_VARIABLE, raising=False)
    monkeypatch.delenv(TOKEN_KEY_VARIABLE, raising=False)
    with pytest.raises(SystemExit, match=TOKEN_ID_VARIABLE):
        register_icons("http://127.0.0.1:8080")


def test_registering_icons_needs_both_halves_of_the_token(monkeypatch):
    """The id alone is not enough — BloodHound signs, it does not bear."""
    monkeypatch.setenv(TOKEN_ID_VARIABLE, "an-id")
    monkeypatch.delenv(TOKEN_KEY_VARIABLE, raising=False)
    with pytest.raises(SystemExit, match=TOKEN_KEY_VARIABLE):
        register_icons("http://127.0.0.1:8080")


def fake_bloodhound(monkeypatch, existing=(), failures=None, saved_queries=()):
    """Stand in for a BloodHound instance, recording what it was sent.

    Only the HTTP boundary is replaced — the payload, the signature chain
    and the create-vs-update decision are all the real code under test.
    ``existing`` names kinds the instance already knows, which is the
    state ingesting a graph leaves behind.
    """
    calls: list[dict] = []
    failures = failures or {}

    def request(method, url, *, content, headers, timeout):
        uri = url.split("8080", 1)[1]
        calls.append(
            {
                "method": method,
                "uri": uri,
                "content": content,
                "signature": headers["Signature"],
                "authorization": headers["Authorization"],
            }
        )
        if (method, uri) in failures:
            return httpx.Response(failures[(method, uri)], text="denied")
        if method == "GET" and uri == "/api/v2/saved-queries":
            return httpx.Response(200, json={"data": list(saved_queries)})
        if method == "GET":
            return httpx.Response(
                200,
                json={"data": [{"kindName": k} for k in existing]},
            )
        return httpx.Response(200, json={})

    monkeypatch.setenv(TOKEN_ID_VARIABLE, "an-id")
    monkeypatch.setenv(TOKEN_KEY_VARIABLE, "a-key")
    monkeypatch.setattr(httpx, "request", request)
    return calls


def test_registering_icons_creates_every_kind_on_a_clean_instance(monkeypatch):
    calls = fake_bloodhound(monkeypatch)
    result = register_icons("http://127.0.0.1:8080/")

    assert set(result.created) == {k.value for k in NodeKind}
    assert result.updated == ()
    # One listing, then one batch create — no per-kind traffic needed.
    assert [c["method"] for c in calls] == ["GET", "POST"]
    assert calls[0]["authorization"] == "bhesignature an-id"
    assert json.loads(calls[1]["content"])["custom_types"].keys() == {
        k.value for k in NodeKind
    }


def test_registering_icons_updates_kinds_ingest_already_created(monkeypatch):
    """The 409 case: kinds exist without icons, so each needs a PUT.

    The collection endpoint only creates, and there is no batch update,
    so refreshing an existing instance is one PUT per kind.
    """
    calls = fake_bloodhound(monkeypatch, existing=[k.value for k in NodeKind])
    result = register_icons("http://127.0.0.1:8080")

    assert result.created == ()
    assert set(result.updated) == {k.value for k in NodeKind}

    assert [c["method"] for c in calls] == ["GET"] + ["PUT"] * len(NodeKind)
    # Per-kind URI, not the collection — a batch PUT is a 405.
    assert calls[1]["uri"].startswith("/api/v2/custom-nodes/")
    assert json.loads(calls[1]["content"])["config"]["icon"]["type"] == (
        "font-awesome"
    )


def test_registering_icons_handles_a_partly_registered_instance(monkeypatch):
    calls = fake_bloodhound(monkeypatch, existing=["Agent", "Tool"])
    result = register_icons("http://127.0.0.1:8080")

    assert set(result.updated) == {"Agent", "Tool"}
    assert "Document" in result.created
    assert [c["method"] for c in calls].count("POST") == 1
    assert [c["method"] for c in calls].count("PUT") == 2


def test_each_request_is_signed_over_its_own_uri(monkeypatch):
    """URI is part of the chain, so per-kind PUTs cannot share a signature."""
    calls = fake_bloodhound(monkeypatch, existing=[k.value for k in NodeKind])
    register_icons("http://127.0.0.1:8080")

    puts = [c for c in calls if c["method"] == "PUT"]
    assert len({c["signature"] for c in puts}) == len(puts)


def test_registering_icons_surfaces_a_failure_body(monkeypatch):
    fake_bloodhound(
        monkeypatch, failures={("GET", "/api/v2/custom-nodes"): 403}
    )
    with pytest.raises(SystemExit, match="403"):
        register_icons("http://127.0.0.1:8080")


# --- Saved Cypher queries ------------------------------------------------


def test_every_query_is_named_scoped_and_documented():
    """The prefix is how an operator finds and removes this project's set."""
    for saved in QUERIES:
        assert saved.full_name.startswith(f"{PREFIX}: ")
        assert saved.description
        assert saved.query.strip().upper().startswith(("MATCH", "OPTIONAL"))

    names = [q.full_name for q in QUERIES]
    assert len(names) == len(set(names))


def test_queries_only_traverse_kinds_the_exporter_emits():
    """A query naming a kind we never emit silently returns nothing."""
    known = {k.value for k in NodeKind} | {k.value for k in EdgeKind}
    known |= {"AgentLab", "Tainted"}

    for saved in QUERIES:
        # Bare capitalised words in a query are kinds or Cypher keywords.
        for token in re.findall(r"[:|]([A-Z][A-Za-z]+)", saved.query):
            assert token in known, f"{saved.name!r} references {token!r}"


def test_query_pack_is_importable_by_bloodhound(tmp_path):
    """One JSON file per query, matching BloodHound's own export format."""
    archive = write_queries(tmp_path / "queries.zip")

    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        assert len(names) == len(QUERIES)
        for name in names:
            entry = json.loads(bundle.read(name))
            assert set(entry) == {"name", "query", "description"}
            assert entry["name"].startswith(PREFIX)


def test_registering_queries_creates_them_on_a_clean_instance(monkeypatch):
    calls = fake_bloodhound(monkeypatch)
    result = register_queries("http://127.0.0.1:8080")

    assert len(result.created) == len(QUERIES)
    assert result.updated == ()
    assert [c["method"] for c in calls] == ["GET"] + ["POST"] * len(QUERIES)


def test_registering_queries_updates_rather_than_duplicating(monkeypatch):
    """Re-running must not leave a second copy in the operator's sidebar."""
    installed = [
        {"id": index, "name": saved.full_name}
        for index, saved in enumerate(QUERIES, start=1)
    ]
    calls = fake_bloodhound(monkeypatch, saved_queries=installed)
    result = register_queries("http://127.0.0.1:8080")

    assert result.created == ()
    assert len(result.updated) == len(QUERIES)
    assert [c["method"] for c in calls] == ["GET"] + ["PUT"] * len(QUERIES)
    # Updates address the existing query by id, not by name.
    assert calls[1]["uri"] == "/api/v2/saved-queries/1"


def test_registering_queries_leaves_other_peoples_queries_alone(monkeypatch):
    calls = fake_bloodhound(
        monkeypatch, saved_queries=[{"id": 99, "name": "someone else's query"}]
    )
    result = register_queries("http://127.0.0.1:8080")

    assert len(result.created) == len(QUERIES)
    assert all("/99" not in c["uri"] for c in calls)


#: Golden value transcribed independently from SpecterOps' documented
#: apiclient.py, so this asserts the chain matches BloodHound rather than
#: merely matching itself.
SIGNING_FIXTURE = {
    "key": "secret-key",
    "method": "POST",
    "uri": "/api/v2/custom-nodes",
    "body": b'{"a": 1}',
    "when": datetime.datetime(
        2026, 8, 21, 14, 37, 5,
        tzinfo=datetime.timezone(datetime.timedelta(hours=2)),
    ),
    "date": "2026-08-21T14:37:05+02:00",
    "signature": "8N93PA/RQykcfWyZyOq1KehL890DfxgAqKFOoA8tY/8=",
}


def test_signature_matches_bloodhounds_documented_chain():
    date, signature = sign_request(
        SIGNING_FIXTURE["method"],
        SIGNING_FIXTURE["uri"],
        SIGNING_FIXTURE["body"],
        SIGNING_FIXTURE["key"],
        when=SIGNING_FIXTURE["when"],
    )
    assert date == SIGNING_FIXTURE["date"]
    assert signature == SIGNING_FIXTURE["signature"]


@pytest.mark.parametrize(
    "field, value",
    [
        ("method", "GET"),
        ("uri", "/api/v2/other"),
        ("body", b'{"a": 2}'),
        (
            "when",
            datetime.datetime(
                2026, 8, 21, 15, 37, 5,
                tzinfo=datetime.timezone(datetime.timedelta(hours=2)),
            ),
        ),
    ],
)
def test_signature_covers_method_uri_body_and_hour(field, value):
    """Chaining means tampering with any part invalidates the signature."""
    arguments = dict(SIGNING_FIXTURE)
    arguments[field] = value
    _, signature = sign_request(
        arguments["method"],
        arguments["uri"],
        arguments["body"],
        arguments["key"],
        when=arguments["when"],
    )
    assert signature != SIGNING_FIXTURE["signature"]


def test_signature_is_stable_within_the_same_hour():
    """Only the hour is signed, so minutes must not change the result."""
    later = SIGNING_FIXTURE["when"].replace(minute=59, second=59)
    _, signature = sign_request(
        SIGNING_FIXTURE["method"],
        SIGNING_FIXTURE["uri"],
        SIGNING_FIXTURE["body"],
        SIGNING_FIXTURE["key"],
        when=later,
    )
    assert signature == SIGNING_FIXTURE["signature"]


def test_export_without_a_report_omits_finding_properties(real_graph):
    payload = to_opengraph(real_graph)
    assert all(
        "Tainted" not in node["kinds"] for node in payload["graph"]["nodes"]
    )


# --- File ingest ---------------------------------------------------------


def fake_ingest(monkeypatch, statuses=(2,), start_id=7):
    """A BloodHound that accepts an upload job and reports job status."""
    calls: list[dict] = []
    remaining = list(statuses)

    def request(method, url, *, content, headers, timeout):
        uri = url.split("8080", 1)[1]
        calls.append({"method": method, "uri": uri, "content": content})
        if uri.endswith("/start"):
            return httpx.Response(201, json={"data": {"id": start_id}})
        if method == "GET" and uri == "/api/v2/file-upload":
            status = remaining.pop(0) if len(remaining) > 1 else remaining[0]
            return httpx.Response(
                200, json={"data": [{"id": start_id, "status": status}]}
            )
        return httpx.Response(200, json={})

    monkeypatch.setenv(TOKEN_ID_VARIABLE, "an-id")
    monkeypatch.setenv(TOKEN_KEY_VARIABLE, "a-key")
    monkeypatch.setattr(httpx, "request", request)
    monkeypatch.setattr(ingest_module.time, "sleep", lambda _: None)
    return calls


def test_ingest_performs_the_three_step_flow(tmp_path, monkeypatch):
    """Uploading without ending the job leaves the data unprocessed."""
    payload = tmp_path / "graph.json"
    payload.write_bytes(b'{"graph":{"nodes":[],"edges":[]}}')
    calls = fake_ingest(monkeypatch)

    result = ingest_graph("http://127.0.0.1:8080", payload)

    uris = [c["uri"] for c in calls if c["method"] == "POST"]
    assert uris == [
        "/api/v2/file-upload/start",
        "/api/v2/file-upload/7",
        "/api/v2/file-upload/7/end",
    ]
    assert result.job_id == 7
    assert result.succeeded


def test_ingest_sends_the_file_bytes_verbatim(tmp_path, monkeypatch):
    """The signature covers these bytes, so they must not be re-encoded."""
    payload = tmp_path / "graph.json"
    raw = b'{"graph": {"nodes": [], "edges": []}}'
    payload.write_bytes(raw)
    calls = fake_ingest(monkeypatch)

    ingest_graph("http://127.0.0.1:8080", payload)

    upload = next(c for c in calls if c["uri"] == "/api/v2/file-upload/7")
    assert upload["content"] == raw


def test_ingest_waits_for_the_job_to_stop_running(tmp_path, monkeypatch):
    payload = tmp_path / "graph.json"
    payload.write_bytes(b"{}")
    # ingesting → analyzing → complete
    fake_ingest(monkeypatch, statuses=(6, 7, 2))

    result = ingest_graph("http://127.0.0.1:8080", payload)
    assert result.status_name == "complete"
    assert result.waited


def test_ingest_reports_a_failed_job_rather_than_claiming_success(
    tmp_path, monkeypatch
):
    payload = tmp_path / "graph.json"
    payload.write_bytes(b"{}")
    fake_ingest(monkeypatch, statuses=(5,))

    result = ingest_graph("http://127.0.0.1:8080", payload)
    assert not result.succeeded
    assert result.status_name == "failed"


def test_ingest_can_skip_waiting(tmp_path, monkeypatch):
    payload = tmp_path / "graph.json"
    payload.write_bytes(b"{}")
    calls = fake_ingest(monkeypatch)

    result = ingest_graph("http://127.0.0.1:8080", payload, wait=False)
    assert result.waited is False
    assert not [c for c in calls if c["method"] == "GET"]
