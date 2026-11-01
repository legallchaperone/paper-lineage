"""Terminal rendering: each line (tributary) as a chain from its origin to the target."""

from __future__ import annotations

from rich.console import Console, Group
from rich.text import Text

from ..graph import LineageGraph
from ..models import Node
from .mathtext import plain

KIND_STYLE = {
    "target": "bold magenta", "origin": "bold green", "ancestor": "cyan", "foundation": "dim",
    "frontier": "yellow", "alternative": "blue", "motivation": "dim cyan", "descendant": "bold blue",
    "concurrent": "blue",
}
ROLE_VERB = {"builds_on": "built on", "technique_source": "idea", "component": "uses"}


def _name(n: Node) -> str:
    y = n.meta.year or "?"
    nm = plain(n.meta.short_name)
    return f"{nm} ({y})"


def _badge(n: Node) -> str:
    return {"origin": "  [origin]", "frontier": "  [unexplored]", "foundation": "  [foundation]"}.get(n.kind, "")


def _quote(s: str, width: int = 160) -> str:
    """Shorten at a word boundary (never mid-word)."""
    if len(s) <= width:
        return s
    cut = s[: width - 1].rsplit(" ", 1)[0].rstrip(",;:")
    return cut + "…"


def build_renderable(g: LineageGraph, show_evidence: bool = False, full: bool = False):
    tgt = g.nodes[g.target]
    out: list = []
    head = Text()
    head.append("Lineage of ", style="bold")
    head.append(f"{plain(tgt.meta.title)} ({tgt.meta.year})", style=KIND_STYLE["target"])
    if tgt.meta.url:
        head.append(f"  {tgt.meta.url}", style="dim")
    out.append(head)
    if g.spine:
        out.append(Text(f"Main path: {g.spine_reason}", style="bold"))
    else:
        out.append(Text(f"No single main path: {g.spine_reason}", style="italic"))
    mot = [g.nodes[e.parent] for e in g.edges if e.child == g.target and e.role == "problem_motivation"]
    if mot:
        out.append(Text("Problem context: " + "; ".join(_name(n) for n in mot), style="dim cyan"))
    out.append(Text(""))

    for i, line in enumerate(g.lines, 1):
        is_spine = bool(g.spine) and line.best_path == g.spine
        title = Text(f"Line {i}: {line.label}", style="bold underline")
        if is_spine:
            title.append("   ★ main path", style="bold #c2410c")
        out.append(title)
        chain = line.best_path
        for j, key in enumerate(chain):
            n = g.nodes[key]
            t = Text("  ")
            t.append(_name(n), style=KIND_STYLE.get(n.kind, "") + (" bold" if is_spine else ""))
            t.append(_badge(n), style="dim")
            out.append(t)
            if key == g.target:
                break
            # other lineage parents feeding this node (not on the chain)
            others = [e for e in g.parents(key) if e.parent not in chain]
            if others:
                o = Text("  │   + also: ", style="dim")
                o.append(", ".join(f"{plain(g.nodes[e.parent].meta.short_name)} ({ROLE_VERB.get(e.role, e.role)})"
                                   for e in others), style="dim")
                out.append(o)
            for e in g.children(key, lineage_only=False):
                if e.role == "alternative":
                    a = Text("  │   ↳ alternative: ", style="blue")
                    a.append(_name(g.nodes[e.child]), style="blue")
                    out.append(a)
            nxt = chain[j + 1]
            e = next((x for x in g.edges if x.parent == key and x.child == nxt), None)
            arrow = Text("  │ ", style="dim")
            arrow.append(f"↓ {ROLE_VERB.get(e.role, e.role) if e else ''}", style="bold" if is_spine else "")
            if e and e.inherited:
                arrow.append(f": {plain(e.inherited)}", style="italic")
            out.append(arrow)
            if show_evidence and e and e.evidence:
                out.append(Text(f"  │     “{_quote(plain(e.evidence))}”", style="dim italic"))
        # other members of this line not on the chain
        extra = [k for k in line.members if k not in chain]
        if extra and full:
            out.append(Text("  (also in this line: " + "; ".join(_name(g.nodes[k]) + _badge(g.nodes[k])
                                                            for k in extra) + ")", style="dim"))
        elif extra:
            out.append(Text(f"  (+{len(extra)} more papers in this line: --full, or click through in -f html)",
                            style="dim"))
        out.append(Text(""))

    alts = [e for e in g.children(g.target, lineage_only=False) if e.role == "alternative"]
    if alts:
        out.append(Text("Done differently (approaches the target contrasts itself with):", style="bold"))
        for e in alts:
            a = g.nodes[e.child]
            t = Text("  ↳ ", style="blue")
            t.append(_name(a), style="blue")
            if "concurrent" in a.note:
                t.append("  [concurrent]", style="dim")
            out.append(t)
            if show_evidence and e.evidence:
                out.append(Text(f"      “{_quote(plain(e.evidence))}”", style="dim italic"))
        out.append(Text(""))
    found = [n for n in g.nodes.values() if n.kind == "foundation"]
    if found:
        out.append(Text("Foundations (generic, not traced further): " + "; ".join(_name(n) for n in found),
                        style="dim"))
    desc = [n for n in g.nodes.values() if n.kind == "descendant"]
    if desc:
        out.append(Text("Descendants (papers that build on the target):", style="bold"))
        for n in desc:
            e = next((x for x in g.edges if x.child == n.key and x.parent == g.target), None)
            t = Text("  ▸ ", style="bold blue")
            t.append(_name(n), style="bold blue")
            if e and e.inherited:
                t.append(f": {plain(e.inherited)}", style="italic")
            out.append(t)
            if show_evidence and e and e.evidence:
                out.append(Text(f"      “{_quote(plain(e.evidence))}”", style="dim italic"))
    conc = [n for n in g.nodes.values() if n.kind == "concurrent"]
    if conc:
        out.append(Text("Concurrent work (same time window, shares the target's parents):", style="bold"))
        for n in conc:
            t = Text(f"  ~ {_name(n)}", style="blue")
            t.append(f"  {n.note}", style="dim")
            out.append(t)
    s = g.stats
    out.append(Text(f"read {s.get('expanded')} papers, classified {s.get('classified_candidates')} "
                    f"citations ({s.get('classifier')})", style="dim"))
    return Group(*out)


def render_tree(g: LineageGraph, console: Console | None = None, show_evidence: bool = False,
                full: bool = False) -> str:
    console = console or Console(record=True, width=110)
    console.print(build_renderable(g, show_evidence=show_evidence, full=full))
    return console.export_text()
