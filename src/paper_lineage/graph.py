"""Lineage DAG: lines (merging tributaries), optional spine, pruning, serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import LINEAGE_ROLES, Edge, Node

ROLE_LINE_LABEL = {"builds_on": "base", "technique_source": "technique", "component": "ingredient"}
SCHEMA_VERSION = 1


@dataclass
class Line:
    head: str                 # the target's direct parent that starts this line
    role: str                 # role of head -> target edge
    members: list[str]        # ancestor keys in this line (incl. head), oldest first
    best_path: list[str]      # max-weight path from an origin to the target (incl. target)
    weight: float
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"head": self.head, "role": self.role, "label": self.label, "members": self.members,
                "best_path": self.best_path, "weight": round(self.weight, 3)}


@dataclass
class LineageGraph:
    target: str
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)
    spine: list[str] | None = None
    spine_reason: str = ""
    log: list[dict[str, Any]] = field(default_factory=list)   # keep/drop decisions
    stats: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- building
    def add_node(self, node: Node) -> Node:
        if node.key in self.nodes:
            old = self.nodes[node.key]
            old.depth = min(old.depth, node.depth)
            return old
        self.nodes[node.key] = node
        return node

    def add_edge(self, edge: Edge) -> None:
        for i, e in enumerate(self.edges):
            if e.parent == edge.parent and e.child == edge.child:
                if edge.weight > e.weight:
                    self.edges[i] = edge
                return
        self.edges.append(edge)

    def parents(self, key: str, lineage_only: bool = True) -> list[Edge]:
        return [e for e in self.edges if e.child == key and (not lineage_only or e.role in LINEAGE_ROLES)]

    def children(self, key: str, lineage_only: bool = True) -> list[Edge]:
        return [e for e in self.edges if e.parent == key and (not lineage_only or e.role in LINEAGE_ROLES)]

    def ancestors(self, key: str) -> set[str]:
        seen: set[str] = set()
        stack = [key]
        while stack:
            k = stack.pop()
            for e in self.parents(k):
                if e.parent not in seen:
                    seen.add(e.parent)
                    stack.append(e.parent)
        return seen

    # ---------------------------------------------------------------- analysis
    def prune(self) -> None:
        """Drop lineage nodes that no longer reach the target; keep side nodes attached to kept nodes."""
        keep = self.ancestors(self.target) | {self.target}
        side_kinds = ("alternative", "motivation", "descendant", "concurrent")
        for e in self.edges:
            if e.child in keep and self.nodes.get(e.parent) and self.nodes[e.parent].kind in side_kinds:
                keep.add(e.parent)
            if e.parent in keep and self.nodes.get(e.child) and self.nodes[e.child].kind in side_kinds:
                keep.add(e.child)
        self.nodes = {k: n for k, n in self.nodes.items() if k in keep}
        self.edges = [e for e in self.edges if e.parent in keep and e.child in keep]

    # Path scoring: an edge is worth its role*directness weight, scaled by how relevant the parent is
    # to the target, minus a per-hop penalty so that long chains of weak links don't win.
    HOP_PENALTY = 1.0

    def edge_score(self, e: Edge) -> float:
        n = self.nodes.get(e.parent)
        rel = n.relevance if n and n.relevance is not None else 0.9
        factor = min(1.0, max(0.2, (rel - 0.80) / 0.12))
        return e.weight * factor - self.HOP_PENALTY

    def strongest_parent(self, k: str, exclude: set[str]) -> Edge | None:
        """The most direct lineage parent of k: best edge score, ties broken by citation pre-score.

        Chains follow each paper's most direct parent (not the longest/heaviest path, which would favour
        weak links that merely have a longer ancestry behind them). Foundations never extend a chain.
        """
        cands = [e for e in self.parents(k)
                 if e.parent not in exclude and self.nodes.get(e.parent) is not None
                 and self.nodes[e.parent].kind != "foundation" and self.edge_score(e) > 0]
        if not cands:
            return None
        return max(cands, key=lambda e: (round(self.edge_score(e), 6), e.score))

    def chain_up(self, k: str) -> tuple[list[str], float]:
        """Greedy strongest-parent chain ending at k (oldest first) and its total score."""
        chain, total = [k], 0.0
        while True:
            e = self.strongest_parent(chain[-1], set(chain))
            if e is None:
                break
            total += self.edge_score(e)
            chain.append(e.parent)
        chain.reverse()
        return chain, total

    def compute_lines(self) -> None:
        heads = self.parents(self.target)
        head_keys = {e.parent for e in heads}
        anc_of = {h: self.ancestors(h) for h in head_keys}
        # a direct parent that is an ancestor of another direct parent belongs to that line
        real_heads = [e for e in heads if not any(e.parent in anc_of[o] for o in head_keys if o != e.parent)]
        chains = {e.parent: self.chain_up(e.parent) for e in real_heads}
        lines: list[Line] = []
        assigned: set[str] = set()
        scored = sorted(real_heads, key=lambda e: (-(self.edge_score(e) + chains[e.parent][1]), -e.score))
        for e in scored:
            chain, up_score = chains[e.parent]
            members = [m for m in sorted({e.parent} | anc_of[e.parent], key=self._order) if m not in assigned]
            members = sorted(set(members) | set(chain), key=self._order)
            assigned.update(members)
            line = Line(head=e.parent, role=e.role, members=members, best_path=chain + [self.target],
                        weight=round(self.edge_score(e) + up_score + self.HOP_PENALTY, 3))
            from .render.mathtext import plain   # display label: formulas as Unicode, never raw LaTeX

            line.label = f"{ROLE_LINE_LABEL.get(e.role, e.role)}: {plain(self.nodes[e.parent].meta.short_name)}"
            lines.append(line)
        self.lines = lines

    def choose_spine(self, mode: str = "auto", margin: float = 1.3) -> None:
        self.spine, self.spine_reason = None, ""
        if mode == "none" or not self.lines:
            self.spine_reason = "spine disabled" if mode == "none" else "no lineage found"
            return
        ranked = sorted(self.lines, key=lambda line: -line.weight)
        top = ranked[0]
        if top.role not in ("builds_on", "technique_source"):
            self.spine_reason = (f"strongest line ({top.label}) enters the target as a {top.role}, "
                                 "not as its base/technique; showing lines as peers")
            return
        if len(ranked) > 1 and top.weight < margin * ranked[1].weight:
            self.spine_reason = (f"no dominant line: {top.label} ({top.weight:.1f}) vs "
                                 f"{ranked[1].label} ({ranked[1].weight:.1f}); showing lines as peers")
            return
        self.spine = top.best_path
        self.spine_reason = f"{top.label} dominates (weight {top.weight:.1f})"

    def _order(self, key: str) -> tuple[str, str]:
        """Chronological, with a stable tie-break (set iteration order varies between processes)."""
        return (self._date(key), key)

    def _date(self, key: str) -> str:
        n = self.nodes.get(key)
        if not n:
            return "9999"
        return n.meta.date or (f"{n.meta.year}-12-31" if n.meta.year else "9999")

    # ---------------------------------------------------------------- output
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LineageGraph:
        """Rebuild a graph from `to_dict()` output (e.g. a saved `-f json` run) for re-rendering."""
        from .models import PaperMeta

        meta_fields = {f for f in PaperMeta.__dataclass_fields__}
        g = cls(target=d["target"])
        for nd in d["nodes"]:
            meta = PaperMeta(**{k: v for k, v in nd.items() if k in meta_fields and k != "key"}, key=nd["key"])
            g.nodes[nd["key"]] = Node(meta=meta, kind=nd.get("kind", "ancestor"), depth=nd.get("depth", 0),
                                      relevance=nd.get("relevance"), expanded=nd.get("expanded", False),
                                      note=nd.get("note", ""))
        edge_fields = {f for f in Edge.__dataclass_fields__}
        for ed in d["edges"]:
            g.edges.append(Edge(**{k: v for k, v in ed.items() if k in edge_fields}))
        g.stats = d.get("stats", {})
        g.log = d.get("decisions", [])
        from .quality import normalize

        normalize(g)
        g.compute_lines()
        g.choose_spine("none" if d.get("spine") is None and "disabled" in d.get("spine_reason", "") else "auto")
        return g

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "target": self.target,
            "spine": self.spine,
            "spine_reason": self.spine_reason,
            "lines": [line.to_dict() for line in self.lines],
            "nodes": [n.to_dict() for n in sorted(self.nodes.values(), key=lambda n: self._order(n.key))],
            "edges": [e.to_dict() for e in self.edges],
            "stats": self.stats,
            "decisions": self.log,
        }
