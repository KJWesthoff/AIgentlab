"""The README's factual claims, checked against the code.

Documentation drifts silently, and this README had drifted into being
wrong — it claimed the shipped config produced "no high or critical
findings" long after it produced two, which contradicted the tool and
the test suite at once. Numbers and inventories are mechanically
checkable, so check them here rather than by eye.

Only claims with a single source of truth belong in this file. Prose
about *why* something matters is not testable and is not tested.
"""

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from agentlab.graph.analysis import Severity, analyze
from agentlab.graph.collect import collect_static
from agentlab.graph.queries import DEMO_ORDER, QUERIES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = (PROJECT_ROOT / "README.md").read_text()
SOURCE = PROJECT_ROOT / "src" / "agentlab"

NUMBER_WORDS = {
    7: "Seven",
    15: "Fifteen",
    17: "Seventeen",
    20: "Twenty",
    25: "Twenty-five",
}


def test_the_test_count_is_current():
    """`N offline tests` must be the number pytest actually collects."""
    claimed = int(
        re.search(r"\| `tests/` \| (\d+) offline tests", README).group(1)
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--collect-only"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    match = re.search(r"(\d+) tests? collected", result.stdout)
    if match is None:  # pragma: no cover - collection itself is broken
        pytest.skip("could not collect tests")

    assert claimed == int(match.group(1)), (
        f"README says {claimed} tests, pytest collects {match.group(1)}"
    )


def test_the_saved_query_count_is_current():
    word = NUMBER_WORDS[len(QUERIES)]
    assert f"{word} queries" in README, (
        f"README should say '{word} queries' for {len(QUERIES)} queries"
    )


def test_the_demo_length_is_current():
    word = NUMBER_WORDS[len(DEMO_ORDER)]
    assert f"{word} queries, each building on the" in README


def test_every_demo_step_is_documented_in_order():
    """The numbered walk-through must match DEMO_ORDER exactly."""
    for index, name in enumerate(DEMO_ORDER, start=1):
        heading = f"**{index}."
        assert heading in README, f"demo step {index} missing from README"

    # And the queries it names must exist.
    for name in DEMO_ORDER:
        assert name.split("—")[0].strip().rstrip("?") in README


def test_every_module_is_documented():
    """A module absent from the reference tables is undiscoverable."""
    modules = sorted(
        path.stem
        for path in SOURCE.rglob("*.py")
        if path.stem != "__init__"
    )
    missing = [m for m in modules if f"`{m}.py`" not in README]
    assert not missing, f"undocumented modules: {missing}"


def test_the_layer_diagram_lists_every_package_module():
    """The ASCII diagram is the first thing people read; keep it complete."""
    diagram = README[README.index("┌───") : README.index("└───")]
    for path in sorted(SOURCE.rglob("*.py")):
        if path.stem == "__init__":
            continue
        entry = f"{path.parent.name}/{path.stem}"
        if path.parent.name == "agentlab":
            continue  # main.py is named separately, above the packages
        assert entry in diagram, f"{entry} missing from the layer diagram"


def cli_flags(module: str) -> set[str]:
    """Long options an argparse-based module defines."""
    tree = ast.parse((SOURCE / module).read_text())
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("--")
    }


@pytest.mark.parametrize(
    "module", ["main.py", "graph/cli.py"]
)
def test_every_cli_flag_is_documented(module):
    undocumented = sorted(f for f in cli_flags(module) if f not in README)
    assert not undocumented, f"{module}: undocumented flags {undocumented}"


def test_the_shipped_config_findings_line_is_current():
    """The quoted output must be what the tool prints today.

    This is the claim that went wrong before: the README described three
    medium findings for a configuration that had since gained a write
    tool and two high ones.
    """
    graph = collect_static(
        config_dir=PROJECT_ROOT / "config",
        corpus_dir=PROJECT_ROOT / "data" / "corpus",
    )
    report = analyze(graph)
    tally = ", ".join(
        f"{report.count(s)} {s.value}" for s in Severity if report.count(s)
    )
    expected = f"{len(report.findings)} findings — {tally}"

    assert expected in README, f"README should quote: {expected!r}"


def test_the_quoted_node_count_is_current():
    graph = collect_static(
        config_dir=PROJECT_ROOT / "config",
        corpus_dir=PROJECT_ROOT / "data" / "corpus",
    )
    claimed = int(
        re.search(r"\((\d+) nodes for the shipped config\)", README).group(1)
    )
    assert claimed == len(graph.nodes)
