"""Mermaid flowchart: oldest at top, target at bottom; one subgraph per line."""

from __future__ import annotations

import re

from ..graph import LineageGraph

ROLE_LABEL = {"builds_on": "builds on", "technique_source": "idea", "component": "uses",
              "problem_motivation": "motivates"}


def _id(key: str) -> str:
    return "n_" + re.sub(r"[^A-Za-z0-9]", "_", key)


def _esc(s: str) -> str:
    return s.replace('"', "'").replace("[", "(").replace("]", ")").replace("<", "‹").replace(">", "›")


def render_mermaid(g: LineageGraph, max_label: int = 40) -> str:
    out = ["flowchart TB"]
    spine_edges = set(zip(g.spine or [], (g.spine or [])[1:]))
    placed: set[str] = set()

    def node_line(key: str) -> str:
        n = g.nodes[key]
        name = _esc(n.meta.short_name)[:max_label]
        return f'{_id(key)}["{name}<br/><small>{n.meta.year or ""}</small>"]:::{n.kind}'

    for i, line in enumerate(g.lines, 1):
        out.append(f'  subgraph L{i}["{_esc(line.label)}"]')
        for k in line.members:
            if k not in placed:
                out.append("    " + node_line(k))
                placed.add(k)
        out.append("  end")
    for k in g.nodes:
        if k not in placed:
            out.append("  " + node_line(k))
            placed.add(k)

    link_styles = []
    idx = 0
    for e in g.edges:
        if e.parent not in g.nodes or e.child not in g.nodes:
            continue
        a, b = _id(e.parent), _id(e.child)
        if e.role == "alternative":
            out.append(f"  {a} -.->|alt| {b}")
        elif e.role == "problem_motivation":
            out.append(f"  {a} -.-> {b}")
        else:
            lab = ROLE_LABEL.get(e.role, e.role)
            arrow = "==>" if (e.parent, e.child) in spine_edges else "-->"
            out.append(f"  {a} {arrow}|{lab}| {b}")
            if (e.parent, e.child) in spine_edges:
                link_styles.append(idx)
        idx += 1
    for i in link_styles:
        out.append(f"  linkStyle {i} stroke:#c2410c,stroke-width:3px")
    out += [
        "  classDef target fill:#fde68a,stroke:#b45309,stroke-width:3px,color:#111",
        "  classDef origin fill:#bbf7d0,stroke:#15803d,color:#111",
        "  classDef ancestor fill:#e0f2fe,stroke:#0369a1,color:#111",
        "  classDef frontier fill:#fef9c3,stroke:#a16207,stroke-dasharray:4 3,color:#111",
        "  classDef foundation fill:#f1f5f9,stroke:#94a3b8,color:#475569",
        "  classDef alternative fill:#ede9fe,stroke:#7c3aed,stroke-dasharray:4 3,color:#111",
        "  classDef motivation fill:#f8fafc,stroke:#64748b,stroke-dasharray:2 2,color:#334155",
        "  classDef descendant fill:#dbeafe,stroke:#1d4ed8,color:#111",
        "  classDef concurrent fill:#f5f3ff,stroke:#8b5cf6,stroke-dasharray:2 4,color:#111",
    ]
    return "\n".join(out) + "\n"
