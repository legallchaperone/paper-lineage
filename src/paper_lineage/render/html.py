"""Single-file HTML report: Mermaid diagram + clickable evidence for every edge."""

from __future__ import annotations

import html
import json

from ..graph import LineageGraph
from .mermaid import _id, render_mermaid

MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"

ROLE_TEXT = {
    "builds_on": "builds on", "technique_source": "takes its core idea from", "component": "uses",
    "alternative": "contrasts with", "problem_motivation": "is motivated by",
}

CSS = """
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--card:#ffffff;--line:#e7e5e4;--accent:#c2410c}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;
 --card:#292524;--line:#44403c;--accent:#fb923c}}
:root[data-theme=dark]{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--card:#292524;--line:#44403c;--accent:#fb923c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
 font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.4rem;margin:0 0 4px}h2{font-size:1.1rem;margin:28px 0 8px}
.muted{color:var(--muted)}a{color:var(--accent)}
.diagram{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px;overflow-x:auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:10px 0}
.card h3{margin:0 0 4px;font-size:1rem}.pill{display:inline-block;font-size:.75rem;padding:1px 8px;
 border-radius:99px;border:1px solid var(--line);color:var(--muted);margin-right:6px}
blockquote{margin:6px 0 0;padding:4px 10px;border-left:3px solid var(--accent);color:var(--muted);font-size:.9rem}
.card:target{outline:2px solid var(--accent)}
"""


def render_html(g: LineageGraph) -> str:
    tgt = g.nodes[g.target]
    mm = render_mermaid(g)
    # make nodes clickable -> jump to their card
    clicks = "\n".join(f'  click {_id(k)} href "#{_id(k)}"' for k in g.nodes)
    mm = mm + clicks + "\n"
    order = sorted(g.nodes.values(), key=lambda n: (n.meta.date or str(n.meta.year or 9999)))
    cards = []
    for n in order:
        ins = [e for e in g.edges if e.child == n.key]
        rows = []
        for e in ins:
            p = g.nodes[e.parent]
            verb = ROLE_TEXT.get(e.role, e.role)
            rows.append(
                f'<div><b>{html.escape(n.meta.short_name)}</b> {verb} '
                f'<a href="#{_id(p.key)}">{html.escape(p.meta.short_name)}</a>'
                f'{" — " + html.escape(e.inherited) if e.inherited else ""}'
                f' <span class="pill">directness {e.directness}</span>'
                f'<span class="pill">{html.escape(e.section_kind or "?")}</span>'
                + (f"<blockquote>“{html.escape(e.evidence)}”</blockquote>" if e.evidence else "")
                + "</div>"
            )
        link = f'<a href="{n.meta.url}" target="_blank" rel="noopener">{html.escape(n.meta.title)}</a>' \
            if n.meta.url else html.escape(n.meta.title)
        cc = f"{n.meta.citation_count} citations" if n.meta.citation_count is not None else ""
        rel = f"relevance {n.relevance:.2f}" if n.relevance is not None else ""
        cards.append(
            f'<div class="card" id="{_id(n.key)}"><h3>{link}</h3>'
            f'<span class="pill">{n.kind}</span><span class="muted">{n.meta.year or ""} · {cc} '
            f'{"· " + rel if rel else ""} {"· " + html.escape(n.note) if n.note else ""}</span>'
            + "".join(rows) + "</div>"
        )
    lines_html = "".join(
        f"<li><b>{html.escape(line.label)}</b>: "
        + " → ".join(html.escape(g.nodes[k].meta.short_name) for k in line.best_path)
        + "</li>" for line in g.lines
    )
    spine = (f"<p><b>Main path:</b> {html.escape(g.spine_reason)}</p>" if g.spine
             else f"<p><b>No single main path</b> — {html.escape(g.spine_reason)}</p>")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lineage: {html.escape(tgt.meta.short_name)}</title><style>{CSS}</style></head>
<body><main>
<h1>Lineage of {html.escape(tgt.meta.title)}</h1>
<div class="muted">{html.escape(tgt.meta.url or "")}</div>
{spine}
<ul>{lines_html}</ul>
<div class="diagram"><pre class="mermaid">{html.escape(mm)}</pre></div>
<p class="muted">Click a node to see why it is there. Every edge quotes the citing sentence verbatim.</p>
<h2>Papers and evidence</h2>
{''.join(cards)}
<p class="muted">{html.escape(json.dumps(g.stats))}</p>
</main>
<script src="{MERMAID_CDN}"></script>
<script>mermaid.initialize({{startOnLoad:true,securityLevel:'loose',theme:'default',flowchart:{{htmlLabels:true}}}});</script>
</body></html>
"""
