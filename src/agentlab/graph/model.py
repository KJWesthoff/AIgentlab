"""The entity/permission graph and how to walk it.

Modeled on BloodHound's central observation about Active Directory: no
single permission in a compromised domain is usually wrong — their
*composition* is. The same holds here. The researcher reading an
untrusted corpus is fine. The writer holding a tool is fine. A path from
the first to the second is not.

Edges are directional and always mean "start can influence or reach
end", so a path is a chain someone could actually walk. Two layers share
one graph:

- **Permission edges** describe what configuration permits, running from
  principal to resource the way ``AdminTo`` does: ``ALLOWED_TO_CALL``,
  ``RUNS_ON``, ``READS``.
- **Flow edges** describe where content can travel, running in the
  direction data moves the way ``HasSession`` does: ``CAN_INJECT``,
  ``PRODUCES``, ``FLOWS_TO``.

A path from a ``Document`` to a write-capable ``Tool`` crosses both
layers and is this system's "path to Domain Admin".
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeKind(str, Enum):
    AGENT = "Agent"
    MODEL_PROFILE = "ModelProfile"
    MODEL = "Model"
    PROVIDER = "Provider"
    TOOL = "Tool"
    CAPABILITY = "Capability"
    CORPUS = "Corpus"
    DOCUMENT = "Document"
    ARTIFACT = "Artifact"
    APPROVAL_GATE = "ApprovalGate"
    PRINCIPAL = "Principal"
    SCOPE = "Scope"


class EdgeKind(str, Enum):
    # --- Permission layer (principal → resource) ---
    ALLOWED_TO_CALL = "AllowedToCall"
    RUNS_ON = "RunsOn"
    BACKED_BY = "BackedBy"
    SERVED_BY = "ServedBy"
    GRANTS = "Grants"
    REQUIRES = "Requires"
    READS = "Reads"
    WRITES = "Writes"
    CONTAINS = "Contains"
    GUARDED_BY = "GuardedBy"
    ACTS_FOR = "ActsFor"
    HOLDS_SCOPE = "HoldsScope"
    REQUIRES_SCOPE = "RequiresScope"

    # --- Flow layer (influence → principal) ---
    CAN_INJECT = "CanInject"
    PRODUCES = "Produces"
    FLOWS_TO = "FlowsTo"
    CAN_COERCE = "CanCoerce"

    # --- Observed at runtime, not granted by configuration ---
    CALLED = "Called"
    DENIED = "Denied"
    APPROVED = "Approved"


#: Edges a path may traverse when asking "can untrusted content starting
#: here end up causing that?". Deliberately excludes descriptive edges
#: like GRANTS and observed-only edges, which say nothing about reach.
TAINT_EDGES = frozenset(
    {
        EdgeKind.CAN_INJECT,
        EdgeKind.PRODUCES,
        EdgeKind.FLOWS_TO,
        EdgeKind.CAN_COERCE,
        EdgeKind.ALLOWED_TO_CALL,
    }
)


@dataclass(frozen=True)
class Node:
    id: str
    kind: NodeKind
    label: str
    properties: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: EdgeKind
    properties: dict[str, Any] = field(default_factory=dict, compare=False)


class Graph:
    """An in-memory directed multigraph keyed by node id.

    Small enough (tens of nodes) that adjacency lists and breadth-first
    search are the right implementation; there is no reason to reach for
    a database until a lab grows several orders of magnitude.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._outgoing: dict[str, list[Edge]] = {}
        self._incoming: dict[str, list[Edge]] = {}

    # --- Construction ---------------------------------------------------

    def add_node(
        self,
        node_id: str,
        kind: NodeKind,
        label: str,
        **properties: Any,
    ) -> Node:
        """Add a node, or merge properties into an existing one.

        Collectors run in sequence over overlapping sources, so declaring
        the same node twice is normal and must not duplicate it.
        """
        existing = self._nodes.get(node_id)
        if existing is not None:
            existing.properties.update(properties)
            return existing

        node = Node(node_id, kind, label, dict(properties))
        self._nodes[node_id] = node
        return node

    def add_edge(
        self,
        source: str,
        target: str,
        kind: EdgeKind,
        **properties: Any,
    ) -> Edge | None:
        """Add an edge between two known nodes.

        Returns ``None`` if either endpoint is missing — a dangling grant
        is a finding for the analyzer to report, not a crash, and not an
        edge that should distort pathfinding.
        """
        if source not in self._nodes or target not in self._nodes:
            return None

        for edge in self._outgoing.get(source, []):
            if edge.target == target and edge.kind is kind:
                edge.properties.update(properties)
                return edge

        edge = Edge(source, target, kind, dict(properties))
        self._edges.append(edge)
        self._outgoing.setdefault(source, []).append(edge)
        self._incoming.setdefault(target, []).append(edge)
        return edge

    # --- Inspection -----------------------------------------------------

    @property
    def nodes(self) -> list[Node]:
        return list(self._nodes.values())

    @property
    def edges(self) -> list[Edge]:
        return list(self._edges)

    def node(self, node_id: str) -> Node | None:
        return self._nodes.get(node_id)

    def label(self, node_id: str) -> str:
        node = self._nodes.get(node_id)
        return node.label if node else node_id

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def of_kind(self, kind: NodeKind) -> list[Node]:
        return [n for n in self._nodes.values() if n.kind is kind]

    def outgoing(
        self, node_id: str, kinds: frozenset[EdgeKind] | None = None
    ) -> list[Edge]:
        edges = self._outgoing.get(node_id, [])
        if kinds is None:
            return list(edges)
        return [e for e in edges if e.kind in kinds]

    def incoming(
        self, node_id: str, kinds: frozenset[EdgeKind] | None = None
    ) -> list[Edge]:
        edges = self._incoming.get(node_id, [])
        if kinds is None:
            return list(edges)
        return [e for e in edges if e.kind in kinds]

    # --- Pathfinding ----------------------------------------------------

    def shortest_path(
        self,
        source: str,
        target: str,
        kinds: frozenset[EdgeKind] | None = None,
    ) -> list[Edge] | None:
        """Fewest-edge path from source to target, or None.

        This is the query BloodHound is built around: not "does this
        principal have the right" but "how few steps away is it".
        """
        if source == target or source not in self._nodes:
            return None

        previous: dict[str, Edge] = {}
        seen = {source}
        queue = deque([source])

        while queue:
            current = queue.popleft()
            for edge in self.outgoing(current, kinds):
                if edge.target in seen:
                    continue
                seen.add(edge.target)
                previous[edge.target] = edge
                if edge.target == target:
                    return self._rebuild(previous, source, target)
                queue.append(edge.target)

        return None

    def reachable_from(
        self,
        source: str,
        kinds: frozenset[EdgeKind] | None = None,
    ) -> dict[str, list[Edge]]:
        """Every node reachable from source, each with a shortest path.

        The bulk query behind "which agents can this document reach?".
        """
        previous: dict[str, Edge] = {}
        seen = {source}
        queue = deque([source])

        while queue:
            current = queue.popleft()
            for edge in self.outgoing(current, kinds):
                if edge.target in seen:
                    continue
                seen.add(edge.target)
                previous[edge.target] = edge
                queue.append(edge.target)

        return {
            node_id: self._rebuild(previous, source, node_id)
            for node_id in seen
            if node_id != source
        }

    def _rebuild(
        self, previous: dict[str, Edge], source: str, target: str
    ) -> list[Edge]:
        path: list[Edge] = []
        cursor = target
        while cursor != source:
            edge = previous[cursor]
            path.append(edge)
            cursor = edge.source
        path.reverse()
        return path

    def describe_path(self, path: list[Edge]) -> str:
        """Render a path the way BloodHound's UI reads: A -[Edge]-> B."""
        if not path:
            return ""
        parts = [self.label(path[0].source)]
        for edge in path:
            parts.append(f"-[{edge.kind.value}]->")
            parts.append(self.label(edge.target))
        return " ".join(parts)
