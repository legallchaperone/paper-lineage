"""Optional: descendants (papers building on the target) and concurrent work."""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import TYPE_CHECKING

from .heuristics import CUES, cosine, score_candidate
from .models import LINEAGE_ROLES, Candidate, Edge, Node, PaperMeta
from .resolve import enrich, meta_from_s2, norm_text

if TYPE_CHECKING:
    from .engine import Engine
    from .graph import LineageGraph

log = logging.getLogger(__name__)


def _find_target_bib(parsed, target: PaperMeta) -> str | None:
    nt = norm_text(target.title)
    head = norm_text(target.title.split(":")[0]) if ":" in target.title else None
    for k, b in parsed.bib.items():
        if target.arxiv_id and b.arxiv_id == target.arxiv_id:
            return k
        text = norm_text(b.text)
        if nt and nt in text:
            return k
        if head and len(head) > 6 and head in text and str(target.year or "") in b.text:
            return k
    return None


def add_descendants(engine: Engine, g: LineageGraph, target: PaperMeta) -> None:
    """Papers that cite the target AND (verified from their own text) build on it."""
    want = engine.cfg.descendants
    s2 = engine.s2
    if s2 is None or not target.arxiv_id:
        return
    engine.progress("descendants: fetching citing papers")
    try:
        cits = s2.citations(f"arXiv:{target.arxiv_id}", limit=1000)
    except Exception as e:
        log.warning("could not fetch citations: %s", e)
        return
    pre = []
    for c in cits:
        cp = c.get("citingPaper") or {}
        ext = cp.get("externalIds") or {}
        if not ext.get("ArXiv"):
            continue
        ctxs = c.get("contexts") or []
        cue = sum(1 for x in ctxs if CUES["build"].search(x) or CUES["inspire"].search(x))
        s = 2.0 * cue + 0.5 * len(ctxs) + math.log1p(cp.get("citationCount") or 0)
        pre.append((s, cp))
    pre.sort(key=lambda x: -x[0])
    kept: list[tuple[int, int, PaperMeta, object]] = []
    for _, cp in pre[: max(3 * want, 12)]:
        meta = meta_from_s2(cp)
        parsed = engine.src.parsed(meta)
        if parsed is None:
            continue
        key = _find_target_bib(parsed, target)
        if not key:
            continue
        ctx = parsed.contexts_for(key)
        if not ctx:
            continue
        cand = Candidate(bib_key=key, bib=parsed.bib[key], meta=replace(target, alias=target.alias),
                         contexts=ctx)
        score_candidate(cand, parsed.full_text, engine.hcfg)
        cls = engine.clf.classify(meta, target, [cand]).get(key)
        if cls and cls.role in LINEAGE_ROLES and cls.directness >= 2 and cls.evidence:
            kept.append((cls.directness, meta.citation_count or 0, meta, cls))
        is_desc = bool(kept) and kept[-1][2] is meta
        g.log.append({"child": meta.short_name, "paper": target.title,
                      "decision": "descendant" if is_desc else "not-descendant",
                      "role": cls.role if cls else None})
        if len(kept) >= want * 2:
            break
    kept.sort(key=lambda x: (-x[0], -x[1]))
    for d, _, meta, cls in kept[:want]:
        n = g.add_node(Node(meta=meta, kind="descendant", depth=-1))
        if n.kind != "descendant":
            continue
        g.add_edge(Edge(parent=target.key, child=meta.key, role=cls.role, directness=cls.directness,
                        inherited=cls.inherited, evidence=cls.evidence, source=cls.source))


def add_concurrent(engine: Engine, g: LineageGraph, target: PaperMeta) -> None:
    """Concurrent work = papers published within ±window of the target that cite at least two of the
    target's direct lineage parents (bibliographic coupling), plus cited alternatives in the window."""
    want = engine.cfg.concurrent
    window = engine.cfg.concurrent_days
    from .engine import _d

    td = _d(target)
    if td is None:
        return
    for n in g.nodes.values():
        nd = _d(n.meta)
        if n.kind == "alternative" and nd and abs((nd - td).days) <= window:
            n.note = (n.note + "; " if n.note else "") + "concurrent with target"
    s2 = engine.s2
    if s2 is None:
        return
    parents = [g.nodes[e.parent] for e in g.parents(g.target)]
    parents = [p for p in parents if p.meta.arxiv_id or p.meta.s2_id]
    if not parents:
        return
    engine.progress("concurrent: looking for papers that share the target's parents")
    shared: dict[str, tuple[PaperMeta, set[str]]] = {}
    for p in parents:
        pid = f"arXiv:{p.meta.arxiv_id}" if p.meta.arxiv_id else p.meta.s2_id
        try:
            cits = s2.citations(pid, limit=5000, fields="title,year,externalIds,citationCount,publicationDate")
        except Exception as e:
            log.warning("citations of %s failed: %s", pid, e)
            continue
        for c in cits:
            cp = c.get("citingPaper") or {}
            if not cp.get("title"):
                continue
            m = meta_from_s2(cp)
            md = _d(m)
            if md is None or abs((md - td).days) > window or m.key in g.nodes:
                continue
            if norm_text(m.title) == norm_text(target.title):
                continue
            entry = shared.setdefault(m.key, (m, set()))
            entry[1].add(p.meta.short_name)
    need = 2 if len(parents) >= 2 else 1
    pool = [(m, ps) for m, ps in shared.values() if len(ps) >= need]
    pool.sort(key=lambda x: (-len(x[1]), -(x[0].citation_count or 0)))
    pool = pool[: max(want * 5, 20)]
    try:
        enrich(s2, [m for m, _ in pool])
    except Exception as e:
        log.warning("enrich failed: %s", e)
    scored = []
    for m, ps in pool:
        r = cosine(m.embedding, target.embedding)
        if r is None or r < engine.cfg.min_relevance:
            continue
        scored.append((len(ps), r, m, ps))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    for _, r, m, ps in scored[:want]:
        days = (_d(m) - td).days  # type: ignore[operator]
        when = f"{abs(days)} days {'after' if days >= 0 else 'before'}"
        g.add_node(Node(meta=m, kind="concurrent", depth=-1, relevance=r,
                        note=f"also builds on {', '.join(sorted(ps))}; published {when} the target"))
