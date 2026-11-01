"""Resolve bibliography entries to Semantic Scholar / arXiv papers."""

from __future__ import annotations

import hashlib
import logging
import re

from .models import BibEntry, PaperMeta
from .s2 import S2Client

log = logging.getLogger(__name__)


def norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def jaccard(a: str, b: str) -> float:
    ta, tb = set(norm_text(a).split()), set(norm_text(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def meta_from_s2(p: dict) -> PaperMeta:
    ext = p.get("externalIds") or {}
    arx = ext.get("ArXiv")
    pid = p.get("paperId")
    key = f"arXiv:{arx}" if arx else (f"S2:{pid}" if pid else "bib:" + hashlib.md5(
        (p.get("title") or "").encode()).hexdigest()[:12])
    emb = (p.get("embedding") or {}).get("vector")
    return PaperMeta(
        key=key,
        title=p.get("title") or "",
        year=p.get("year"),
        date=p.get("publicationDate"),
        arxiv_id=arx,
        s2_id=pid,
        citation_count=p.get("citationCount"),
        abstract=p.get("abstract"),
        embedding=emb,
    )


def _usable_ref_title(t: str) -> bool:
    n = norm_text(t)
    return len(n) >= 12 and len(n.split()) >= 2


def resolve_bib(
    bib: dict[str, BibEntry],
    s2_refs: list[dict],
    citing_title: str = "",
) -> dict[str, PaperMeta]:
    """Match each bib entry to one of the citing paper's S2 references (title containment / arXiv id)."""
    refs = [r for r in s2_refs if r.get("title") and norm_text(r["title"]) != norm_text(citing_title)]
    by_arxiv = {(r.get("externalIds") or {}).get("ArXiv"): r for r in refs if (r.get("externalIds") or {}).get("ArXiv")}
    normed = [(norm_text(r["title"]), r) for r in refs if _usable_ref_title(r["title"])]
    out: dict[str, PaperMeta] = {}
    used: set[int] = set()
    for key, b in bib.items():
        hit = None
        if b.arxiv_id and b.arxiv_id in by_arxiv:
            hit = by_arxiv[b.arxiv_id]
        if hit is None:
            text = " " + norm_text(b.text) + " "
            best = None
            for nt, r in normed:
                if f" {nt} " in text and (best is None or len(nt) > len(best[0])):
                    best = (nt, r)
            if best:
                hit = best[1]
        if hit is None and b.title:
            scored = [(jaccard(b.title, r["title"]), r) for r in refs]
            scored = [x for x in scored if x[0] >= 0.75]
            if scored:
                hit = max(scored, key=lambda x: x[0])[1]
        if hit is not None:
            out[key] = meta_from_s2(hit)
            used.add(id(hit))
        elif b.arxiv_id:
            out[key] = PaperMeta(key=f"arXiv:{b.arxiv_id}", title=b.title or b.text[:150], arxiv_id=b.arxiv_id,
                                 year=_year(b.text))
    return out


def _year(text: str) -> int | None:
    ys = re.findall(r"\b(19[5-9]\d|20[0-4]\d)\b", text)
    return int(ys[-1]) if ys else None


def resolve_unmatched(
    s2: S2Client, entries: list[BibEntry], max_calls: int = 10
) -> dict[str, PaperMeta]:
    """Title search for important entries S2's reference list didn't cover."""
    out: dict[str, PaperMeta] = {}
    for b in entries[:max_calls]:
        title = b.title
        if not title or len(norm_text(title)) < 12:
            continue
        try:
            p = s2.match_title(title)
        except Exception as e:  # network trouble shouldn't kill the run
            log.debug("title match failed for %r: %s", title, e)
            continue
        if not p or jaccard(title, p.get("title") or "") < 0.75:
            continue
        y = _year(b.text)
        if y and p.get("year") and abs(int(p["year"]) - y) > 1:
            continue
        out[b.key] = meta_from_s2(p)
    return out


def enrich(s2: S2Client, metas: list[PaperMeta]) -> None:
    """Fill in embedding / abstract / date / citation count in place via the batch endpoint."""
    want = [m for m in metas if m.embedding is None or m.abstract is None]
    ids = []
    for m in want:
        if m.arxiv_id:
            ids.append(f"arXiv:{m.arxiv_id}")
        elif m.s2_id:
            ids.append(m.s2_id)
    if not ids:
        return
    got = s2.batch(ids)
    for m in want:
        p = got.get(f"arXiv:{m.arxiv_id}") if m.arxiv_id else got.get(m.s2_id or "")
        if not p:
            continue
        full = meta_from_s2(p)
        m.embedding = m.embedding or full.embedding
        m.abstract = m.abstract or full.abstract
        m.date = m.date or full.date
        m.year = m.year or full.year
        m.s2_id = m.s2_id or full.s2_id
        if full.citation_count is not None:
            m.citation_count = full.citation_count
        if not m.arxiv_id and full.arxiv_id:
            m.arxiv_id = full.arxiv_id
            m.key = f"arXiv:{full.arxiv_id}"
