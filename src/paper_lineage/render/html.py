"""Single-file HTML page in a hand-drawn (Excalidraw-like) style.

Left: a sketchy canvas where every line of work is a titled frame, papers are hand-drawn boxes and
arrows flow left-to-right into the target. Right: a serif walkthrough with one section per frame and
one step per paper; scrolling the text glides the canvas to the frame/paper it is talking about.

Everything is embedded (rough.js, MIT; the Virgil font, SIL OFL 1.1), so the page works offline.
The walkthrough text is generated from the graph only: verbs come from the citation roles, and
every claim is followed by the verbatim sentence from the citing paper.
"""

from __future__ import annotations

import base64
import html
import json
import pkgutil
from typing import Any

from ..graph import LineageGraph
from ..models import LINEAGE_ROLES
from .mathtext import plain, rich

ROLE_SHORT = {"builds_on": "built on", "technique_source": "idea from", "component": "uses",
              "alternative": "alternative", "problem_motivation": "motivates"}
VERB = {"builds_on": "is built on", "technique_source": "takes its core idea from", "component": "uses"}
DIRECT = {"builds_on": "also builds directly on", "technique_source": "also takes an idea directly from",
          "component": "also uses"}
LINE_TITLE = {"builds_on": "The base", "technique_source": "The core idea", "component": "A key ingredient"}
SECTION_NAME = {"intro": "introduction", "related": "related work", "prelim": "preliminaries", "method": "method",
                "experiments": "experiments", "appendix": "appendix", "figure": "a figure caption",
                "table": "a table", "abstract": "abstract", "discussion": "discussion", "conclusion": "conclusion"}


def _asset(name: str) -> bytes:
    data = pkgutil.get_data("paper_lineage.render", f"assets/{name}")
    if data is None:  # pragma: no cover
        raise FileNotFoundError(name)
    return data


def graph_payload(g: LineageGraph) -> dict:
    nodes = {}
    for k, n in g.nodes.items():
        m = n.meta
        nodes[k] = {"name": plain(m.short_name), "title": plain(m.title), "year": m.year, "url": m.url, "kind": n.kind,
                    "cites": m.citation_count,
                    "rel": None if n.relevance is None else round(n.relevance, 3), "note": n.note}
    edges = [{"p": e.parent, "c": e.child, "role": e.role, "d": e.directness, "inh": e.inherited,
              "ev": e.evidence, "sec": e.section_kind} for e in g.edges
             if e.parent in g.nodes and e.child in g.nodes]
    lines = [{"label": line.label, "role": line.role, "head": line.head, "path": line.best_path,
              "members": line.members} for line in g.lines]
    return {"target": g.target, "spine": g.spine, "spineReason": g.spine_reason, "lines": lines,
            "nodes": nodes, "edges": edges, "stats": g.stats, "roles": ROLE_SHORT}


# ----------------------------------------------------------------------------- walkthrough text

def _e(s: Any) -> str:
    return html.escape(str(s if s is not None else ""))


class _Story:
    def __init__(self, g: LineageGraph):
        self.g = g
        self.T = g.nodes[g.target]
        self.sections: list[tuple[str, str, str]] = []   # (id, title, html)

    def name(self, k: str) -> str:
        return self.g.nodes[k].meta.short_name

    def year(self, k: str) -> str:
        y = self.g.nodes[k].meta.year
        return str(y) if y else "?"

    def link(self, k: str, label: str | None = None) -> str:
        n = self.g.nodes[k]
        text = rich(label or n.meta.short_name)
        if n.meta.url:
            return (f'<a href="{_e(n.meta.url)}" target="_blank" rel="noopener" '
                    f'title="{_e(plain(n.meta.title))}">{text}</a>')
        return f'<span title="{_e(plain(n.meta.title))}">{text}</span>'

    def mention(self, k: str, lane: int | str = "") -> str:
        """A paper name in the text that focuses it on the canvas when clicked."""
        return (f'<button class="mention" data-node="{_e(k)}" data-lane="{lane}" '
                f'title="{_e(plain(self.g.nodes[k].meta.title))}">{rich(self.name(k))}</button>')

    def quote(self, e) -> str:
        if e is None or not e.evidence:
            return ""
        where = SECTION_NAME.get(e.section_kind, "")
        src = f"{self.name(e.child)}{', ' + where if where else ''}"
        return f'<blockquote>“{rich(e.evidence)}”<cite>— {rich(src)}</cite></blockquote>'

    def edge(self, p: str, c: str):
        for e in self.g.edges:
            if e.parent == p and e.child == c:
                return e
        return None

    @staticmethod
    def inh(e) -> str:
        return f": {rich(e.inherited.rstrip('.'))}." if e is not None and e.inherited else "."

    def line_title(self, line) -> str:
        return f"{LINE_TITLE.get(line.role, line.role)}: {self.name(line.head)}"

    # -- sections
    def intro(self) -> None:
        g, T = self.g, self.T
        lines = g.lines
        routes = "".join(
            f'<li><a href="#line-{i}" class="route"><b>{i + 1}. {rich(self.line_title(line))}</b></a><br>'
            + " → ".join(rich(self.name(k)) for k in line.best_path) + "</li>"
            for i, line in enumerate(lines))
        if g.spine:
            dom = f"One line clearly dominates ({_e(g.spine_reason)}); it is starred on the canvas."
        elif len(lines) > 1:
            dom = "No single line dominates: they are peers that merge in this paper, so read them in any order."
        else:
            dom = ""
        n = len(lines)
        body = f"""
<p class="lede">{self.link(g.target, T.meta.title)} ({self.year(g.target)})</p>
<p>This page traces where <b>{rich(T.meta.short_name)}</b> comes from. It draws on {n} line{'s' if n != 1 else ''}
of work. Each one is a frame on the canvas, oldest paper on the left, flowing into
{rich(T.meta.short_name)} on the right. {dom}</p>
<p>Every link is backed by a sentence quoted from the newer paper. Nothing here comes from outside
knowledge: if a paper did not say it, it is not drawn.</p>
<h3>Reading routes</h3>
<ul class="routes">{routes}</ul>
<h3>How to read the drawing</h3>
<ul>
<li>A <b>frame</b> is one line of work. Arrows say how the newer paper uses the older one:
<i>built on</i>, <i>idea from</i> or <i>uses</i>.</li>
<li><b>Dashed boxes</b> hanging below a paper are approaches it contrasts itself with: how it could have
been done differently.</li>
<li><b>Hatched boxes</b> are where this run stopped tracing (paper budget or depth limit).</li>
<li>The <span class="redcircle">red circle</span> marks the paper the text is talking about. Click any
paper, on the canvas or in the text, to jump to it. <b>Whole canvas</b> zooms back out.</li>
</ul>"""
        self.sections.append(("intro", f"Where {T.meta.short_name} comes from", body))

    def line_section(self, i: int, line) -> None:
        g = self.g
        chain = line.best_path
        origin = chain[0]
        start = {
            "origin": "None of its own references survived the filters as on-topic lineage, so this is where "
                      "the line starts.",
            "frontier": "The trail continues further back, but this run stopped here (paper budget or depth "
                        "limit). Re-run with a larger <code>--max-nodes</code> or <code>--max-depth</code> "
                        "to go further.",
            "foundation": "It is a widely used foundation: shown, but not traced further.",
        }.get(g.nodes[origin].kind, "")
        parts = [f'<p>This line starts at {self.mention(origin, i)} ({self.year(origin)}) and reaches '
                 f'{rich(self.T.meta.short_name)} through {self.mention(line.head, i)}.</p>']
        for j, k in enumerate(chain):
            n = g.nodes[k]
            is_t = k == g.target
            blk = [f'<h3>{"→ " if is_t else ""}{rich(n.meta.short_name)} <span class="yr">· {self.year(k)}</span></h3>']
            if not is_t:
                label = n.meta.title if n.meta.short_name != n.meta.title else (
                    "open on arXiv" if n.meta.arxiv_id else n.meta.title)
                blk.append(f'<p class="fulltitle">{self.link(k, label)}</p>')
            if j == 0:
                if start:
                    blk.append(f"<p>{start}</p>")
            else:
                prev = chain[j - 1]
                e = self.edge(prev, k)
                if e is not None:
                    blk.append(f"<p><b>{rich(n.meta.short_name)}</b> {VERB.get(e.role, e.role)} "
                               f"{self.mention(prev, i)}{self.inh(e)}</p>")
                    blk.append(self.quote(e))
            if not is_t and k != line.head:      # shortcut straight into the target
                d = self.edge(k, g.target)
                if d is not None and d.role in LINEAGE_ROLES:
                    blk.append(f'<p class="aside">{rich(self.T.meta.short_name)} {DIRECT.get(d.role, d.role)} '
                               f'{rich(n.meta.short_name)}{self.inh(d)}</p>')
                    blk.append(self.quote(d))
            if not is_t:
                others = [e for e in g.parents(k) if e.parent not in chain]
                if others:
                    items = ", ".join(f"{self.link(e.parent)} <span class='yr'>({ROLE_SHORT.get(e.role, e.role)})</span>"
                                      for e in others)
                    blk.append(f'<p class="aside">Also feeding into {rich(n.meta.short_name)}: {items}.</p>')
                alts = [e for e in g.children(k, lineage_only=False) if e.role == "alternative"][:2]
                if alts:
                    items = ", ".join(f"{self.mention(e.child, i)} ({self.year(e.child)})" for e in alts)
                    blk.append(f'<p class="aside">Done differently: {items}.</p>')
                    for e in alts:
                        blk.append(f'<div class="alt" data-focus="node:{_e(e.child)}:{i}" data-node="{_e(e.child)}" '
                                   f'data-lane="{i}">{self.quote(e)}</div>')
            focus = f"node:{k}:{'t' if is_t else i}"
            lane = "t" if is_t else i
            parts.append(f'<div class="step" data-focus="{_e(focus)}" data-node="{_e(k)}" data-lane="{lane}">'
                         + "".join(blk) + "</div>")
        self.sections.append((f"line-{i}", f"{i + 1}. {self.line_title(line)}", "".join(parts)))

    def differently(self) -> None:
        g = self.g
        alts = [e for e in g.children(g.target, lineage_only=False) if e.role == "alternative"][:3]
        if not alts:
            return
        parts = [f"<p>{rich(self.T.meta.short_name)} positions itself against these approaches to the same "
                 "problem.</p>"]
        for e in alts:
            parts.append(f'<div class="step" data-focus="node:{_e(e.child)}:t" data-node="{_e(e.child)}" '
                         f'data-lane="t"><h3>{rich(self.name(e.child))} <span class="yr">· {self.year(e.child)}</span></h3>'
                         f'<p class="fulltitle">{self.link(e.child, g.nodes[e.child].meta.title)}</p>'
                         f'{self.quote(e)}</div>')
        self.sections.append(("differently", "Done differently", "".join(parts)))

    def extras(self) -> None:
        g = self.g
        desc = [k for k, n in g.nodes.items() if n.kind == "descendant"]
        if desc:
            parts = [f"<p>Papers that, in their own words, build on {rich(self.T.meta.short_name)}.</p>"]
            for k in desc:
                e = self.edge(g.target, k)
                verb = VERB.get(e.role, "builds on") if e else "builds on"
                parts.append(f'<div class="step" data-focus="node:{_e(k)}:d" data-node="{_e(k)}" data-lane="d">'
                             f'<h3>{rich(self.name(k))} <span class="yr">· {self.year(k)}</span></h3>'
                             f'<p class="fulltitle">{self.link(k, g.nodes[k].meta.title)}</p>'
                             f'<p><b>{rich(self.name(k))}</b> {verb} {rich(self.T.meta.short_name)}{self.inh(e)}</p>'
                             f'{self.quote(e)}</div>')
            self.sections.append(("descendants", "What came after", "".join(parts)))
        conc = [k for k, n in g.nodes.items() if n.kind == "concurrent"]
        if conc:
            items = "".join(f"<li>{self.link(k, g.nodes[k].meta.title)} <span class='yr'>({self.year(k)})</span>"
                            f"<br><span class='aside'>{rich(g.nodes[k].note)}</span></li>" for k in conc)
            self.sections.append(("concurrent", "Around the same time",
                                  "<p>Not cited by the target: found because they appeared in the same window and "
                                  f"build on the same parents.</p><ul>{items}</ul>"))

    def rest(self) -> None:
        g = self.g
        drawn = {k for line in g.lines for k in line.best_path} | {g.target}

        def chips(ks):
            return " ".join(f'<span class="chip">{self.link(k)} <span class="yr">{self.year(k)}</span></span>'
                            for k in ks)

        groups = []
        mot = [e.parent for e in g.edges if e.child == g.target and e.role == "problem_motivation"]
        if mot:
            groups.append(("Problem context", "Papers the target cites as defining the problem it works on.", mot))
        found = [k for k, n in g.nodes.items() if n.kind == "foundation"]
        if found:
            groups.append(("Foundations", "Widely used, generic papers: shown once, never traced further.", found))
        side = [k for k, n in g.nodes.items() if k not in drawn and n.kind in ("ancestor", "origin")]
        if side:
            groups.append(("Side ancestors", "Kept as lineage but not on a frame's main chain. The ones that feed "
                           "a drawn paper directly are named in the steps above.", side))
        front = [k for k, n in g.nodes.items() if k not in drawn and n.kind == "frontier"]
        if front:
            groups.append(("Not traced further", "Reached but not read (paper budget or depth limit).", front))
        if not groups:
            return
        body = "".join(f"<h3>{_e(t)}</h3><p class='aside'>{_e(h)}</p><div class='chips'>{chips(ks)}</div>"
                       for t, h, ks in groups)
        self.sections.append(("rest", "Everything else that was kept", body))

    def about(self) -> None:
        s = self.g.stats or {}
        bits = [f"{s.get('expanded', '?')} papers read",
                f"{s.get('classified_candidates', '?')} citations classified ({s.get('classifier', '?')})"]
        if s.get("llm_calls"):
            bits.append(f"{s['llm_calls']} LLM calls")
        ev = s.get("evidence_validation") or {}
        if ev:
            bits.append(f"{ev.get('exact', 0) + ev.get('repaired', 0)} quotes verified against the source text")
        body = (f"<p>{_e(' · '.join(bits))}.</p>"
                "<p>Made with <b>paper-lineage</b>. Roles come from reading each citing sentence; quotes are copied "
                "verbatim from the papers and checked against their text. Drawing style after Excalidraw: sketches "
                "by rough.js (MIT), handwriting in Virgil (SIL Open Font License 1.1).</p>")
        self.sections.append(("about", "About this page", body))

    def build(self) -> list[tuple[str, str, str]]:
        self.intro()
        for i, line in enumerate(self.g.lines):
            self.line_section(i, line)
        self.differently()
        self.extras()
        self.rest()
        self.about()
        return self.sections


def _focus_for(sid: str) -> str:
    if sid.startswith("line-"):
        return "frame:" + sid.split("-", 1)[1]
    return {"differently": "frame:t", "descendants": "frame:d"}.get(sid, "all")


# ----------------------------------------------------------------------------- page

CSS = r"""
@font-face{font-family:"Virgil";src:url(data:font/woff2;base64,__VIRGIL__) format("woff2");font-display:block}
:root{--bg:#fbfbfa;--ink:#1e1e1e;--muted:#6c6c6c;--rule:#e6e4df;--navy:#1d3a8a;--red:#e03131;--left:52%;
 color-scheme:light}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--ink)}
body{display:flex;flex-direction:column;overflow:hidden;font-family:ui-serif,"Iowan Old Style","Palatino Linotype",
 Georgia,"Times New Roman",serif}
.workspace{flex:1;display:flex;min-height:0}
.visual{width:var(--left);position:relative;min-width:260px;background:var(--bg)}
#stage{position:absolute;inset:0;touch-action:none;cursor:grab;overflow:hidden}
#stage.dragging{cursor:grabbing}
#svg{width:100%;height:100%;display:block;user-select:none}
.whole{position:absolute;left:20px;top:18px;z-index:2;font:15px/1 ui-serif,Georgia,serif;background:#fff;
 border:1px solid #d9d6cf;border-radius:4px;padding:12px 16px;cursor:pointer;color:var(--ink);
 box-shadow:0 1px 2px rgba(0,0,0,.04)}.whole:hover{border-color:var(--navy)}
.zoom{position:absolute;right:16px;bottom:14px;z-index:2;display:flex;gap:6px}
.zoom button{font:17px/1 ui-serif,Georgia,serif;width:34px;height:34px;background:#fff;border:1px solid #d9d6cf;
 border-radius:4px;cursor:pointer;color:var(--ink)}
.divider{width:9px;cursor:col-resize;position:relative;flex:none;border-left:1px solid var(--rule);background:var(--bg);
 touch-action:none}
.divider::after{content:"";position:absolute;left:-1px;top:50%;width:7px;height:38px;margin-top:-19px;
 border:1px solid #cfccc5;border-radius:5px;background:#fff}
.notes{flex:1;min-width:300px;min-height:0;display:flex;flex-direction:column;background:var(--bg)}
.reading-nav{display:flex;align-items:center;gap:18px;padding:14px 56px 10px;font-size:15px}
.brand{border:1px solid #d9d6cf;border-radius:4px;padding:6px 12px;background:#fff;font-family:"Virgil",cursive;
 font-size:16px}
details.contents{position:relative}
details.contents summary{cursor:pointer;list-style:none}
details.contents summary::-webkit-details-marker{display:none}
details.contents summary::after{content:" ⌄";color:var(--muted)}
details.contents ol{position:absolute;z-index:5;top:28px;left:0;margin:0;padding:10px 14px 10px 32px;background:#fff;
 border:1px solid #d9d6cf;border-radius:4px;min-width:290px;box-shadow:0 6px 24px rgba(0,0,0,.08)}
details.contents li{margin:4px 0}details.contents a{color:var(--ink);text-decoration:none}
details.contents a:hover{color:var(--navy);text-decoration:underline}
.counter{margin-left:auto;color:var(--muted);font-size:14.5px;white-space:nowrap;overflow:hidden;
 text-overflow:ellipsis;max-width:42%}
.reader{flex:1;overflow:auto;padding:8px 56px 45vh}
.reader article{max-width:640px}
.reader section{padding-top:24px}
.reader h1{font-weight:400;color:var(--navy);font-size:40px;line-height:1.15;margin:18px 0 22px}
.reader h2{font-weight:400;font-size:34px;line-height:1.2;margin:44px 0 18px}
.reader h3{font-size:21px;margin:30px 0 6px;font-weight:600}
.reader p,.reader li{font-size:19px;line-height:1.7;margin:0 0 16px}
.reader ul{padding-left:24px}
.reader a{color:var(--navy)}
.lede{font-style:italic;color:var(--muted)}
.yr{color:var(--muted);font-weight:400}
.reader .fulltitle{font-size:16px;line-height:1.45;margin-bottom:10px;color:var(--muted)}
.fulltitle a{color:var(--muted)}
.reader .aside{font-size:16.5px;color:#444}
.step{border-left:2px solid transparent;margin-left:-18px;padding-left:16px;transition:border-color .3s}
.step.on{border-left-color:var(--red)}
blockquote{margin:6px 0 18px;padding:2px 0 2px 16px;border-left:3px solid #d9d6cf;color:#3d3d3d;font-style:italic;
 font-size:17.5px;line-height:1.6}
blockquote cite{display:block;font-style:normal;color:var(--muted);font-size:14.5px;margin-top:4px}
.mention{font:inherit;color:var(--navy);background:none;border:0;padding:0;cursor:pointer;
 border-bottom:1px dashed rgba(29,58,138,.45)}.mention:hover{border-bottom-style:solid}
.chip{display:inline-block;border:1px solid #dcd9d2;border-radius:4px;padding:2px 9px;margin:0 6px 8px 0;
 font-size:15.5px;background:#fff}
.chips{line-height:1.9}
.routes li{margin-bottom:12px}
.route{text-decoration:none}
.reader math{font-size:1.07em;font-style:normal}.tex.fallback{font-style:italic}
.redcircle{border:2px solid var(--red);border-radius:50%;padding:0 6px}
code{font-size:.85em;background:#efede8;padding:1px 5px;border-radius:3px}
.site-footer{text-align:center;font:12.5px ui-sans-serif,system-ui,sans-serif;color:var(--muted);padding:6px;
 border-top:1px solid var(--rule)}
.paper{cursor:pointer}
.paper .hover{opacity:0;transition:opacity .15s}
.paper:hover .hover{opacity:1}
.frame{transition:opacity .45s}
.frame.dim{opacity:.18}
@media (max-width:820px){
 .workspace{flex-direction:column}.visual{width:auto;height:44vh;min-height:220px;flex:none}
 .notes{min-width:0}.site-footer{display:none}.brand{display:none}
 .divider{display:none}.reading-nav{padding:10px 18px 6px}.reader{padding:4px 18px 45vh}
 .reader h1{font-size:30px}.reader h2{font-size:27px}.reader p,.reader li{font-size:17.5px}
 .step{margin-left:-10px;padding-left:8px}.counter{display:none}
 .whole{top:10px;left:10px;padding:8px 10px}}
"""

JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const N = D.nodes, E = D.edges, T = D.target;
const NS = 'http://www.w3.org/2000/svg';
const LIN = new Set(['builds_on', 'technique_source', 'component']);
const INK = '#1e1e1e', BLUE = '#1971c2', RED = '#e03131', GREY = '#868e96', VIOLET = '#5f3dc4';
// Excalidraw's palette: strong fills for papers, faint tints for frames
const FILL = ['#a5d8ff', '#b2f2bb', '#ffd8a8', '#d0bfff', '#eebefa', '#ffc9c9'];
const TINT = ['#eef6fc', '#f0f8f0', '#fdf3ea', '#f5f2fc', '#fbf1fc', '#fcf1f1'];
const BW = 236, BH = 86, GAP = 150, FP = 34, FHEAD = 74, AW = 208, AH = 58, AGAP = 14, FGAP = 64;
const svg = document.getElementById('svg'), world = document.getElementById('world');
const rc = rough.svg(svg);
const el = (tag, attrs, parent) => { const x = document.createElementNS(NS, tag);
  for (const k in attrs) x.setAttribute(k, attrs[k]); if (parent) parent.appendChild(x); return x; };
const hash = s => { let h = 2166136261; for (const c of String(s)) h = Math.imul(h ^ c.charCodeAt(0), 16777619);
  return (h >>> 0) % 2147483647 || 1; };
const altsOf = k => E.filter(e => e.p === k && e.role === 'alternative');
const edgeOf = (p, c) => E.find(e => e.p === p && e.c === c);

// ---- text in the hand-drawn font
const mctx = document.createElement('canvas').getContext('2d');
function wrap(str, size, maxW, maxLines) {
  mctx.font = `${size}px Virgil`;
  const words = String(str).split(/\s+/); const out = []; let cur = '';
  for (const w of words) { const t = cur ? cur + ' ' + w : w;
    if (mctx.measureText(t).width <= maxW || !cur) cur = t; else { out.push(cur); cur = w; } }
  if (cur) out.push(cur);
  if (out.length > maxLines) { out.length = maxLines; out[maxLines - 1] += '…'; }
  return out.map(l => { while (mctx.measureText(l).width > maxW && l.length > 2) l = l.slice(0, -2) + '…'; return l; });
}
function text(parent, x, y, str, size, color, anchor) {
  const t = el('text', {x, y, 'font-family': 'Virgil', 'font-size': size, fill: color || INK,
    'text-anchor': anchor || 'start'}, parent);
  t.textContent = str; return t;
}
function rrect(x, y, w, h, r) {
  return `M${x + r},${y} L${x + w - r},${y} Q${x + w},${y} ${x + w},${y + r} L${x + w},${y + h - r} ` +
    `Q${x + w},${y + h} ${x + w - r},${y + h} L${x + r},${y + h} Q${x},${y + h} ${x},${y + h - r} ` +
    `L${x},${y + r} Q${x},${y} ${x + r},${y} Z`;
}
function arrowHead(parent, x2, y2, ang, color, seed) {
  for (const s of [-1, 1]) { const a = ang + Math.PI + s * 0.45;
    parent.appendChild(rc.line(x2, y2, x2 + 15 * Math.cos(a), y2 + 15 * Math.sin(a),
      {stroke: color, strokeWidth: 1.8, roughness: 0.9, seed: seed + s + 7})); }
}

// ---- layout + drawing
const boxes = {};     // "key:lane" -> {x,y,w,h}
const frames = {};    // frame id -> {x,y,w,h}
let gMarks = null;
function build() {
  world.innerHTML = '';
  const gFrames = el('g', {}, world), gEdges = el('g', {}, world); gMarks = el('g', {}, world);
  const lanes = D.lines.map((l, i) => ({i, chain: l.path.filter(k => k !== T), line: l}));
  const maxN = Math.max(1, ...lanes.map(l => l.chain.length));
  const innerW = n => n * BW + (n - 1) * GAP;
  const XR = 40 + innerW(maxN) + 2 * FP;
  // big underlined handwritten title, like an Excalidraw heading
  const tg = el('g', {class: 'frame', 'data-f': 'title'}, gFrames);
  const ttl = `Where ${N[T].name} comes from`;
  text(tg, 40, 52, ttl, 46, INK);
  mctx.font = '46px Virgil'; const tw = mctx.measureText(ttl).width;
  tg.appendChild(rc.line(36, 68, 44 + tw, 65, {stroke: INK, strokeWidth: 2.4, roughness: 1.4, seed: 3}));
  let y = 120; const centers = [];
  for (const L of lanes) {
    const n = L.chain.length;
    const nAlt = Math.max(0, ...L.chain.map(k => Math.min(2, altsOf(k).length)));
    const fw = innerW(n) + 2 * FP, fh = FHEAD + BH + (nAlt ? 22 + nAlt * (AH + AGAP) : 0) + FP;
    const fx = XR - fw, fy = y;
    const g = el('g', {class: 'frame', 'data-f': String(L.i)}, gFrames);
    frames[String(L.i)] = {x: fx, y: fy, w: fw, h: fh};
    g.appendChild(rc.path(rrect(fx, fy, fw, fh, 22), {fill: TINT[L.i % 6], fillStyle: 'solid', stroke: INK,
      strokeWidth: 2.2, roughness: 0.9, seed: hash('f' + L.i)}));
    const star = D.spine && L.line.path.join() === D.spine.join() ? '  ★' : '';
    const label = `${L.i + 1}. ${L.line.label}${star}`;
    mctx.font = '24px Virgil'; const lw = Math.min(fw - 40, mctx.measureText(label).width + 30);
    g.appendChild(rc.path(rrect(fx + 18, fy + 16, lw, 42, 10), {fill: '#ffffff', fillStyle: 'solid', stroke: INK,
      strokeWidth: 1.8, roughness: 0.8, seed: hash('ft' + L.i)}));
    text(g, fx + 33, fy + 45, wrap(label, 24, lw - 26, 1)[0], 24, INK);
    L.chain.forEach((k, j) => {
      const x = fx + FP + j * (BW + GAP), by = fy + FHEAD;
      paper(g, k, L.i, x, by, BW, BH, FILL[L.i % 6]);
      if (j > 0) arrow(g, x - GAP + 6, by + BH / 2, x - 8, by + BH / 2, edgeOf(L.chain[j - 1], k), hash(k + L.i));
      altsOf(k).slice(0, 2).forEach((e, a) => {
        const ax = x + 26, ay = by + BH + 22 + a * (AH + AGAP);
        g.appendChild(rc.path(`M${x + 12},${by + BH} L${x + 12},${ay + AH / 2} L${ax - 4},${ay + AH / 2}`,
          {stroke: GREY, strokeWidth: 1.3, roughness: 0.8, strokeLineDash: [6, 5], seed: hash(e.c + 'c')}));
        paper(g, e.c, L.i, ax, ay, AW, AH, '#ffffff', 'alt');
      });
      if (k !== L.line.head && E.some(e => e.p === k && e.c === T && LIN.has(e.role)))
        text(g, x + BW - 4, by - 10, '↘ also used directly', 16, RED, 'end');
    });
    centers.push(fy + FHEAD + BH / 2);
    y += fh + FGAP;
  }
  // the target's own frame
  const tAlts = altsOf(T).slice(0, 3);
  const TW = 290, TH = 112, tfw = TW + 2 * FP + 20;
  const tfh = FHEAD + TH + (tAlts.length ? 22 + tAlts.length * (AH + AGAP) : 0) + FP;
  const cy = centers.length ? (centers[0] + centers[centers.length - 1]) / 2 : 160;
  const tfx = XR + 210, tfy = Math.max(120, cy - FHEAD - TH / 2);
  const gT = el('g', {class: 'frame', 'data-f': 't'}, gFrames);
  frames.t = {x: tfx, y: tfy, w: tfw, h: tfh};
  gT.appendChild(rc.path(rrect(tfx, tfy, tfw, tfh, 22), {fill: '#fff9db', fillStyle: 'solid', stroke: INK,
    strokeWidth: 2.4, roughness: 0.9, seed: 99}));
  gT.appendChild(rc.path(rrect(tfx + 18, tfy + 16, 150, 42, 10), {fill: '#fff', fillStyle: 'solid', stroke: INK,
    strokeWidth: 1.8, roughness: 0.8, seed: 98}));
  text(gT, tfx + 33, tfy + 45, 'This paper', 24, INK);
  const tx = tfx + FP + 10, ty = tfy + FHEAD;
  paper(gT, T, 't', tx, ty, TW, TH, '#ffd43b', 'target');
  tAlts.forEach((e, a) => {
    const ax = tx + 28, ay = ty + TH + 22 + a * (AH + AGAP);
    gT.appendChild(rc.path(`M${tx + 12},${ty + TH} L${tx + 12},${ay + AH / 2} L${ax - 4},${ay + AH / 2}`,
      {stroke: GREY, strokeWidth: 1.3, roughness: 0.8, strokeLineDash: [6, 5], seed: hash(e.c + 'tc')}));
    paper(gT, e.c, 't', ax, ay, AW + 30, AH, '#ffffff', 'alt');
  });
  // each frame's last paper flows into the target
  for (const L of lanes) {
    const last = L.chain[L.chain.length - 1]; if (!last) continue;
    const b = boxes[last + ':' + L.i];
    const x1 = b.x + b.w + 6, y1 = b.y + b.h / 2, x2 = tx - 10, y2 = ty + TH / 2 + (L.i - (lanes.length - 1) / 2) * 16;
    const pts = [[x1, y1], [x1 + 90, y1], [x2 - 110, y2], [x2, y2]];
    gEdges.appendChild(rc.curve(pts, {stroke: INK, strokeWidth: 1.9, roughness: 0.8, seed: hash(last + 'T')}));
    arrowHead(gEdges, x2, y2, Math.atan2(y2 - pts[2][1], x2 - pts[2][0]), INK, hash(last + 'Th'));
    const e = edgeOf(last, T);
    if (e) text(gEdges, x1 + 14, y1 - 10, D.roles[e.role] || e.role, 17, BLUE);
  }
  // papers that build on the target (--descendants)
  const desc = Object.keys(N).filter(k => N[k].kind === 'descendant');
  if (desc.length) {
    const dfx = tfx + tfw + 170, dfw = AW + 30 + 2 * FP, dfh = FHEAD + desc.length * (AH + AGAP) + FP - AGAP;
    const dfy = tfy + tfh / 2 - dfh / 2;
    const gD = el('g', {class: 'frame', 'data-f': 'd'}, gFrames);
    frames.d = {x: dfx, y: dfy, w: dfw, h: dfh};
    gD.appendChild(rc.path(rrect(dfx, dfy, dfw, dfh, 22), {fill: '#f3f0ff', fillStyle: 'solid', stroke: INK,
      strokeWidth: 2.2, roughness: 0.9, seed: 77}));
    text(gD, dfx + 22, dfy + 45, 'What came after', 24, INK);
    desc.forEach((k, di) => paper(gD, k, 'd', dfx + FP, dfy + FHEAD + di * (AH + AGAP), AW + 30, AH, '#e5dbff', 'alt'));
    const x1 = tx + TW + 8, y1 = ty + TH / 2, x2 = dfx - 10, y2 = dfy + dfh / 2;
    gEdges.appendChild(rc.curve([[x1, y1], [x1 + 60, y1], [x2 - 60, y2], [x2, y2]],
      {stroke: INK, strokeWidth: 1.9, roughness: 0.8, seed: 55}));
    arrowHead(gEdges, x2, y2, 0, INK, 56);
  }
}

function arrow(g, x1, y1, x2, y2, e, seed) {
  g.appendChild(rc.line(x1, y1, x2, y2, {stroke: INK, strokeWidth: 1.9, roughness: 0.8, seed}));
  arrowHead(g, x2, y2, 0, INK, seed);
  if (e) text(g, (x1 + x2) / 2, y1 - 12, D.roles[e.role] || e.role, 17, BLUE, 'middle');
}

const KTAG = {origin: 'where it starts', frontier: 'not traced further', foundation: 'foundation',
  descendant: 'builds on it', target: 'this paper'};
function paper(parent, k, lane, x, y, w, h, fill, kind) {
  const n = N[k]; if (!n) return;
  const g = el('g', {class: 'paper', 'data-k': k, 'data-lane': String(lane)}, parent);
  const opts = {fill, fillStyle: n.kind === 'frontier' ? 'hachure' : 'solid', hachureGap: 7, fillWeight: 1,
    stroke: INK, strokeWidth: kind === 'target' ? 2.6 : 1.8, roughness: 1, seed: hash(k + lane)};
  if (kind === 'alt') { opts.strokeLineDash = [8, 6]; opts.stroke = VIOLET; }
  g.appendChild(rc.path(rrect(x, y, w, h, 12), opts));
  const size = kind === 'target' ? 27 : kind === 'alt' ? 18 : 21;
  const lines = wrap(n.name, size, w - 28, kind === 'alt' ? 1 : 2);
  const top = kind === 'alt' ? y + 25 : y + (lines.length > 1 ? 32 : 40) + (kind === 'target' ? 8 : 0);
  lines.forEach((s, i) => text(g, x + 14, top + i * (size + 4), s, size, INK));
  const tag = (n.year || '?') + (KTAG[n.kind] ? ' · ' + KTAG[n.kind] : '');
  text(g, x + 14, y + h - 12, tag, kind === 'alt' ? 14 : 15, GREY);
  el('rect', {x: x - 5, y: y - 5, width: w + 10, height: h + 10, rx: 14, fill: 'none', stroke: '#1d3a8a',
    'stroke-width': 1.4, 'stroke-dasharray': '3 4', class: 'hover'}, g);
  el('title', {}, g).textContent = n.title;
  boxes[k + ':' + lane] = {x, y, w, h};
  if (!boxes[k]) boxes[k] = {x, y, w, h};
  g.addEventListener('click', ev => { if (moved) return; ev.stopPropagation(); fromCanvas(k, String(lane)); });
}

// ---- camera
let view = {x: 0, y: 0, s: 1}, anim = 0;
const apply = () => world.setAttribute('transform', `translate(${view.x},${view.y}) scale(${view.s})`);
function fitTo(b, pad, maxS) {
  const r = svg.getBoundingClientRect();
  const s = Math.min(maxS || 1.6, (r.width - 2 * pad) / b.w, (r.height - 2 * pad) / b.h);
  return {s, x: r.width / 2 - (b.x + b.w / 2) * s, y: r.height / 2 - (b.y + b.h / 2) * s};
}
function goTo(v, ms) {
  cancelAnimationFrame(anim);
  if (!ms) { view = v; apply(); return; }
  const from = {...view}, t0 = performance.now();
  const ease = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  const step = now => { const t = Math.min(1, (now - t0) / ms), k = ease(t);
    view = {x: from.x + (v.x - from.x) * k, y: from.y + (v.y - from.y) * k, s: from.s + (v.s - from.s) * k}; apply();
    if (t < 1) anim = requestAnimationFrame(step); };
  anim = requestAnimationFrame(step);
}
function bboxAll() { const b = world.getBBox(); return {x: b.x, y: b.y, w: b.width, h: b.height}; }
function union(a, b) { const x = Math.min(a.x, b.x), y = Math.min(a.y, b.y);
  return {x, y, w: Math.max(a.x + a.w, b.x + b.w) - x, h: Math.max(a.y + a.h, b.y + b.h) - y}; }

// ---- focus, driven by the text
let focusKey = null;
function setFocus(f, instant) {
  if (f === focusKey && !instant) return; focusKey = f;
  gMarks.innerHTML = '';
  const keepOnly = keep => document.querySelectorAll('.frame').forEach(g =>
    g.classList.toggle('dim', keep !== null && !keep.includes(g.dataset.f)));
  const ms = instant ? 0 : 700;
  if (!f || f === 'all') { keepOnly(null); goTo(fitTo(bboxAll(), 24, 1.1), ms); return; }
  const i = f.indexOf(':'), j = f.lastIndexOf(':');
  const kind = f.slice(0, i);
  if (kind === 'frame') {
    const id = f.slice(i + 1), fr = frames[id]; if (!fr) { setFocus('all', instant); return; }
    keepOnly([id, 'title']);
    goTo(fitTo(id === 't' || id === 'd' ? fr : union(fr, frames.t), 30, 1.25), ms); return;
  }
  const key = f.slice(i + 1, j), lane = f.slice(j + 1);
  const bx = boxes[key + ':' + lane] || boxes[key]; if (!bx) return;
  keepOnly([lane, 't', 'title']);
  gMarks.appendChild(rc.ellipse(bx.x + bx.w / 2, bx.y + bx.h / 2, bx.w + 46, bx.h + 40,
    {stroke: RED, strokeWidth: 2.2, roughness: 1.6, seed: hash(key + 'm')}));
  // show the paper with its neighbours in the frame
  const fr = frames[lane] || bx;
  const x0 = Math.max(fr.x, bx.x - BW - GAP), x1 = Math.min(fr.x + fr.w, bx.x + bx.w + BW + GAP);
  goTo(fitTo(union({x: x0, y: fr.y, w: x1 - x0, h: fr.h}, {x: bx.x - 30, y: bx.y - 30, w: bx.w + 60, h: bx.h + 60}),
    28, 1.35), ms);
}

// ---- text <-> canvas
const reader = document.getElementById('reader');
const counter = document.getElementById('counter');
const focusables = [...document.querySelectorAll('[data-focus]')];
function currentFocus() {
  const r = reader.getBoundingClientRect(), line = r.top + r.height * 0.33;
  let best = null;
  for (const x of focusables) { const b = x.getBoundingClientRect();
    if (b.top <= line && b.bottom > r.top && (!best || b.top >= best.getBoundingClientRect().top)) best = x; }
  return best || focusables[0];
}
function onScroll() {
  const x = currentFocus(); if (!x) return;
  document.querySelectorAll('.step.on').forEach(s => s.classList.remove('on'));
  const step = x.closest('.step'); if (step) step.classList.add('on');
  const sec = x.closest('section'); if (sec) counter.textContent = sec.dataset.title || '';
  setFocus(x.dataset.focus);
}
let raf = 0;
reader.addEventListener('scroll', () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(onScroll); });
function scrollToEl(x) { reader.scrollTo({top: x.offsetTop - reader.clientHeight * 0.2, behavior: 'smooth'}); }
function stepFor(k, lane) {
  const cands = [...document.querySelectorAll('.step, .alt')].filter(s => s.dataset.node === k);
  return cands.find(c => c.dataset.lane === lane) || cands[0];
}
function fromCanvas(k, lane) { const s = stepFor(k, lane); if (s) scrollToEl(s); else setFocus(`node:${k}:${lane}`); }
document.querySelectorAll('.mention').forEach(m => m.addEventListener('click', () => {
  const s = [...document.querySelectorAll('.step')].find(x => x.dataset.node === m.dataset.node && x.dataset.lane === m.dataset.lane);
  if (s) scrollToEl(s); else setFocus(`node:${m.dataset.node}:${m.dataset.lane}`);
}));
document.querySelectorAll('a.route, details.contents a').forEach(a => a.addEventListener('click', ev => {
  const t = document.querySelector(a.getAttribute('href')); if (!t) return; ev.preventDefault();
  reader.scrollTo({top: t.offsetTop - 10, behavior: 'smooth'});
  const d = a.closest('details'); if (d) d.removeAttribute('open'); }));

// ---- manual pan / zoom (Excalidraw conventions: wheel pans, ctrl/cmd+wheel or pinch zooms)
const stage = document.getElementById('stage');
function zoomAt(f, cx, cy) { cancelAnimationFrame(anim); const s = Math.min(3, Math.max(0.12, view.s * f)), k = s / view.s;
  view = {s, x: cx - (cx - view.x) * k, y: cy - (cy - view.y) * k}; apply(); }
stage.addEventListener('wheel', ev => { ev.preventDefault(); cancelAnimationFrame(anim);
  const r = svg.getBoundingClientRect();
  if (ev.ctrlKey || ev.metaKey) zoomAt(Math.exp(-ev.deltaY * 0.01), ev.clientX - r.left, ev.clientY - r.top);
  else { view.x -= ev.shiftKey ? ev.deltaY : ev.deltaX; view.y -= ev.shiftKey ? 0 : ev.deltaY; apply(); } },
  {passive: false});
const ptrs = new Map(); let last = null, pinch = null, moved = false;
stage.addEventListener('pointerdown', ev => { ptrs.set(ev.pointerId, ev); moved = false;
  if (ptrs.size === 1) last = {x: ev.clientX, y: ev.clientY};
  if (ptrs.size === 2) { const [a, b] = [...ptrs.values()]; pinch = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY); } });
stage.addEventListener('pointermove', ev => { if (!ptrs.has(ev.pointerId)) return; ptrs.set(ev.pointerId, ev);
  if (ptrs.size === 2 && pinch) { const [a, b] = [...ptrs.values()];
    const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY), r = svg.getBoundingClientRect();
    zoomAt(d / pinch, (a.clientX + b.clientX) / 2 - r.left, (a.clientY + b.clientY) / 2 - r.top); pinch = d; moved = true; return; }
  if (last) { const dx = ev.clientX - last.x, dy = ev.clientY - last.y;
    if (!moved && Math.abs(dx) + Math.abs(dy) > 3) { moved = true; cancelAnimationFrame(anim); stage.classList.add('dragging'); }
    if (moved) { view.x += dx; view.y += dy; apply(); } last = {x: ev.clientX, y: ev.clientY}; } });
const up = ev => { ptrs.delete(ev.pointerId); if (ptrs.size < 2) pinch = null;
  if (!ptrs.size) { last = null; stage.classList.remove('dragging'); setTimeout(() => { moved = false; }, 0); } };
stage.addEventListener('pointerup', up); stage.addEventListener('pointercancel', up);
document.getElementById('whole').onclick = () => { focusKey = null; setFocus('all'); };
document.getElementById('zin').onclick = () => { const r = svg.getBoundingClientRect(); zoomAt(1.25, r.width / 2, r.height / 2); };
document.getElementById('zout').onclick = () => { const r = svg.getBoundingClientRect(); zoomAt(0.8, r.width / 2, r.height / 2); };

// ---- resizable divider
const divider = document.getElementById('divider');
divider.addEventListener('pointerdown', ev => { divider.setPointerCapture(ev.pointerId);
  const move = e => document.documentElement.style.setProperty('--left',
    Math.min(75, Math.max(25, e.clientX / window.innerWidth * 100)) + '%');
  const stop = () => { divider.removeEventListener('pointermove', move); divider.removeEventListener('pointerup', stop);
    const f = focusKey; focusKey = null; setFocus(f, true); };
  divider.addEventListener('pointermove', move); divider.addEventListener('pointerup', stop); });
window.addEventListener('resize', () => { const f = focusKey; focusKey = null; setFocus(f, true); });

// typeset formulas (Temml -> native MathML); on failure the readable Unicode fallback stays
document.querySelectorAll('.tex').forEach(s => {
  try { temml.render(s.dataset.tex, s, {throwOnError: true}); s.classList.add('typeset'); }
  catch (err) { s.classList.add('fallback'); s.title = 'Formula could not be typeset'; }
});

document.fonts.load('20px Virgil').catch(() => {}).then(() => document.fonts.ready).then(() => {
  build(); setFocus('all', true); onScroll(); document.body.dataset.ready = '1';
});
"""


def render_html(g: LineageGraph) -> str:
    tgt = g.nodes[g.target]
    sections = _Story(g).build()
    payload = json.dumps(graph_payload(g), ensure_ascii=False).replace("</", "<\\/")
    font = base64.b64encode(_asset("Virgil-Regular.woff2")).decode()
    rough_js = _asset("rough.js").decode("utf-8")
    temml_js = _asset("temml.min.js").decode("utf-8")
    temml_css = _asset("Temml-Local.css").decode("utf-8").replace(
        "url('Temml.woff2')", "url(data:font/woff2;base64," + base64.b64encode(_asset("Temml.woff2")).decode() + ")")
    contents = "".join(f'<li><a href="#{sid}">{rich(title)}</a></li>' for sid, title, _ in sections)
    body = []
    for idx, (sid, title, inner) in enumerate(sections):
        tag = "h1" if idx == 0 else "h2"
        body.append(f'<section id="{sid}" data-title="{_e(plain(title))}">'
                    f'<{tag} data-focus="{_focus_for(sid)}">{rich(title)}</{tag}>{inner}</section>')
    css = CSS.replace("__VIRGIL__", font)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#fbfbfa">
<title>Lineage: {_e(plain(tgt.meta.short_name))}</title><style>{css}</style>
<style>/* Temml (c) Ron Kok, MIT License */
{temml_css}</style></head>
<body>
<main class="workspace">
  <section class="visual" aria-label="Hand-drawn lineage canvas">
    <button class="whole" id="whole" type="button">Whole canvas &nbsp;↖</button>
    <div id="stage"><svg id="svg" role="img" aria-label="Lineage of {_e(plain(tgt.meta.title))}: lines of work drawn as frames flowing into the paper"><g id="world"></g></svg></div>
    <div class="zoom"><button id="zout" type="button" aria-label="Zoom out">−</button><button id="zin" type="button" aria-label="Zoom in">+</button></div>
  </section>
  <div class="divider" id="divider" role="separator" aria-orientation="vertical" aria-label="Resize"></div>
  <aside class="notes" aria-label="Walkthrough">
    <div class="reading-nav">
      <span class="brand">paper-lineage</span>
      <details class="contents"><summary>Contents</summary><ol>{contents}</ol></details>
      <span class="counter" id="counter"></span>
    </div>
    <div class="reader" id="reader" tabindex="0"><article>{''.join(body)}</article></div>
  </aside>
</main>
<footer class="site-footer">Made with paper-lineage · every link quotes the paper that makes it</footer>
<script type="application/json" id="data">{payload}</script>
<script>/* rough.js 4.6.6, (c) 2019 Preet Shihn, MIT License */
{rough_js}</script>
<script>/* Temml 0.14.0, (c) 2020 Ron Kok, MIT License */
{temml_js}</script>
<script>{JS}</script>
</body></html>
"""
