"""Best-first recursive expansion of a paper's lineage."""

from __future__ import annotations

import hashlib
import heapq
import logging
from dataclasses import dataclass
from datetime import date
from typing import Callable, Protocol

from .graph import LineageGraph
from .quality import normalize
from .heuristics import HeuristicConfig, cosine, score_candidate
from .models import LINEAGE_ROLES, ROLE_WEIGHT, Candidate, Classification, Edge, Node, PaperMeta, ParsedPaper
from .parse import norm_ws
from .resolve import _year, enrich, norm_text, resolve_bib, resolve_unmatched

log = logging.getLogger(__name__)


@dataclass
class Config:
    max_depth: int = 4
    max_nodes: int = 30                # papers whose full text is read (expanded), incl. target
    parents_per_node: int = 4
    alternatives: int = 2              # per node, only for nodes with depth <= alt_depth
    alt_depth: int = 1
    min_directness: int = 2
    min_relevance: float = 0.86        # SPECTER2 cosine to the target
    hub_citations: int = 15000         # never expand beyond this many citations
    component_hub_citations: int = 3000  # components this popular are treated as generic foundations
    llm_candidates: int = 25           # top heuristic candidates sent to the classifier per paper
    concurrent_days: int = 90
    descendants: int = 0               # how many descendants to show (0 = off)
    concurrent: int = 0                # how many concurrent works to show (0 = off)
    spine: str = "auto"


class Source(Protocol):
    def meta_for_arxiv(self, arxiv_id: str) -> PaperMeta: ...
    def parsed(self, meta: PaperMeta) -> ParsedPaper | None: ...
    def references(self, meta: PaperMeta) -> list[dict]: ...


class Classifier(Protocol):
    name: str

    def classify(self, child: PaperMeta, target: PaperMeta, cands: list[Candidate]) -> dict[str, Classification]: ...


def _d(meta: PaperMeta) -> date | None:
    if meta.date:
        try:
            return date.fromisoformat(meta.date[:10])
        except ValueError:
            pass
    if meta.year:
        return date(int(meta.year), 7, 1)
    return None


def bib_only_meta(c: Candidate) -> PaperMeta:
    title = c.bib.title or c.bib.text[:150]
    h = hashlib.md5(norm_text(title).encode()).hexdigest()[:12]
    return PaperMeta(key=f"bib:{h}", title=title, year=_year(c.bib.text))


class Engine:
    def __init__(self, source: Source, classifier: Classifier, cfg: Config | None = None,
                 s2=None, progress: Callable[[str], None] | None = None):
        self.src = source
        self.clf = classifier
        self.cfg = cfg or Config()
        self.s2 = s2 if s2 is not None else getattr(source, "s2", None)
        self.progress = progress or (lambda msg: None)
        self.hcfg = HeuristicConfig(hub_citations=self.cfg.hub_citations)

    # ------------------------------------------------------------------ helpers
    def _relevance(self, meta: PaperMeta, target: PaperMeta, cls: Classification) -> float | None:
        return cosine(meta.embedding, target.embedding)

    def _relevant(self, rel: float | None, cls: Classification) -> bool:
        if rel is not None:
            return rel >= self.cfg.min_relevance
        if cls.relevance is not None:
            return cls.relevance >= 1.5
        return True  # no signal: let the role/directness decide

    def candidates(self, meta: PaperMeta, parsed: ParsedPaper) -> list[Candidate]:
        refs = self.src.references(meta)
        metas = resolve_bib(parsed.bib, refs, parsed.title)
        cands: list[Candidate] = []
        for key, b in parsed.bib.items():
            ctx = parsed.contexts_for(key)
            if not ctx:
                continue
            c = Candidate(bib_key=key, bib=b, meta=metas.get(key), contexts=ctx)
            score_candidate(c, parsed.full_text, self.hcfg)
            cands.append(c)
        cands.sort(key=lambda c: -c.score)
        # shortlist for the (expensive) classifier
        short = cands[: self.cfg.llm_candidates]
        extra = [c for c in cands[self.cfg.llm_candidates:]
                 if c.features.get("cue_inspire") or c.features.get("cue_build", 0) >= 2]
        short += extra[:10]
        # resolve important entries that S2's reference list missed
        missing = [c for c in short if c.meta is None]
        if missing and self.s2 is not None:
            found = resolve_unmatched(self.s2, [c.bib for c in missing], max_calls=6)
            for c in missing:
                c.meta = found.get(c.bib_key)
        for c in short:
            if c.meta is None:
                c.meta = bib_only_meta(c)
        if self.s2 is not None:
            try:
                enrich(self.s2, [c.meta for c in short if c.meta])
            except Exception as e:
                log.warning("enrich failed: %s", e)
        # self-citations of the child itself (same title) are dropped
        return [c for c in short if norm_text(c.title) != norm_text(meta.title)]

    # ------------------------------------------------------------------ main
    def run(self, arxiv_id: str) -> LineageGraph:
        cfg = self.cfg
        target = self.src.meta_for_arxiv(arxiv_id)
        self.progress(f"target: {target.title}")
        g = LineageGraph(target=target.key)
        g.add_node(Node(meta=target, kind="target", depth=0, relevance=1.0))
        heap: list[tuple[float, int, str]] = [(0.0, 0, target.key)]
        counter = 1
        expanded = 0
        n_class = 0
        while heap and expanded < cfg.max_nodes:
            _, _, key = heapq.heappop(heap)
            node = g.nodes[key]
            if node.expanded:
                continue
            parsed = self.src.parsed(node.meta)
            if parsed is None:
                node.note = "no full text available"
                g.log.append({"paper": node.meta.title, "decision": "not expanded", "reason": node.note})
                continue
            expanded += 1
            node.expanded = True
            self.progress(f"[{expanded}/{cfg.max_nodes}] reading {node.meta.short_name} "
                          f"({node.meta.year}) depth {node.depth}")
            if node.meta.abstract is None and parsed.abstract:
                node.meta.abstract = parsed.abstract
            cands = self.candidates(node.meta, parsed)
            classes = self.clf.classify(node.meta, target, cands)
            n_class += len(cands)
            def push(pri: float, k: str) -> None:
                nonlocal counter
                counter += 1
                heapq.heappush(heap, (pri, counter, k))

            self._select(g, node, target, cands, classes, heap_push=push)

        for _, _, k in heap:
            n = g.nodes.get(k)
            if n and not n.expanded and n.kind == "ancestor":
                n.kind = "frontier"
                n.note = n.note or "not expanded (node budget)"
        # origins: expanded ancestors without non-foundation lineage parents
        for n in g.nodes.values():
            if n.kind == "ancestor" and n.expanded:
                ps = [e for e in g.parents(n.key) if g.nodes[e.parent].kind != "foundation"]
                if not ps:
                    n.kind = "origin"
            elif n.kind == "ancestor" and not n.expanded:
                n.kind = "origin" if not n.note.startswith("depth") else "frontier"
        g.prune()
        normalize(g)
        g.compute_lines()
        g.choose_spine(cfg.spine)
        g.stats.update({"expanded": expanded, "classified_candidates": n_class,
                        "classifier": getattr(self.clf, "name", "?")})
        if hasattr(self.clf, "stats"):
            g.stats["evidence_validation"] = dict(self.clf.stats)
        if cfg.descendants:
            from .extras import add_descendants
            add_descendants(self, g, target)
        if cfg.concurrent:
            from .extras import add_concurrent
            add_concurrent(self, g, target)
        from .intros import add_intros
        self.progress("writing paper introductions")
        add_intros(g, llm=getattr(self.clf, "llm", None), s2=self.s2, arxiv=getattr(self.src, "arxiv", None))
        return g

    def _select(self, g: LineageGraph, node: Node, target: PaperMeta, cands: list[Candidate],
                classes: dict[str, Classification], heap_push) -> None:
        cfg = self.cfg
        child_date = _d(node.meta)
        lineage, alts, motiv = [], [], []
        for c in cands:
            cls = classes.get(c.bib_key)
            meta = c.meta
            if cls is None or meta is None:
                continue
            rel = self._relevance(meta, target, cls)
            rec = {"child": node.meta.short_name, "paper": meta.title, "role": cls.role,
                   "directness": cls.directness, "relevance": None if rel is None else round(rel, 3),
                   "score": c.score, "source": cls.source}
            pdate = _d(meta)
            later = bool(child_date and pdate and pdate > child_date)
            if later and cls.role in LINEAGE_ROLES:
                if (pdate - child_date).days <= cfg.concurrent_days:
                    rec.update(decision="concurrent", reason="published after the citing paper (within window)")
                    if cls.role != "alternative":
                        cls = Classification(role="alternative", directness=cls.directness, inherited=cls.inherited,
                                             evidence=cls.evidence, relevance=cls.relevance, source=cls.source)
                else:
                    rec.update(decision="drop", reason="published after the citing paper (cited in a later revision)")
                    g.log.append(rec)
                    continue
            if cls.role in LINEAGE_ROLES and cls.directness >= cfg.min_directness:
                if not self._relevant(rel, cls):
                    rec.update(decision="drop", reason=f"off-topic for target (relevance {rel})")
                elif not cls.evidence:
                    rec.update(decision="drop", reason="no verifiable evidence sentence")
                else:
                    lineage.append((c, cls, rel))
                    rec.update(decision="candidate-parent")
            elif cls.role == "alternative" and cls.directness >= 1 and node.depth <= cfg.alt_depth:
                if self._relevant(rel, cls) and cls.evidence:
                    alts.append((c, cls, rel))
                    rec.update(decision="candidate-alternative")
                else:
                    rec.update(decision="drop", reason="alternative not relevant / no evidence")
            elif cls.role == "problem_motivation" and node.depth == 0 and cls.directness >= 1:
                motiv.append((c, cls, rel))
                rec.update(decision="candidate-motivation")
            else:
                rec.update(decision="drop", reason=f"role {cls.role} / directness {cls.directness}")
            g.log.append(rec)

        lineage.sort(key=lambda x: (-(ROLE_WEIGHT[x[1].role] * x[1].directness), -x[0].score))
        for c, cls, rel in lineage[: cfg.parents_per_node]:
            self._add_parent(g, node, c, cls, rel, heap_push)
        for c, cls, rel in lineage[cfg.parents_per_node:]:
            g.log.append({"child": node.meta.short_name, "paper": c.title, "decision": "drop",
                          "reason": "over parents-per-node limit"})
        alts.sort(key=lambda x: (-x[1].directness, -x[0].score))
        for c, cls, rel in alts[: cfg.alternatives]:
            n = g.add_node(Node(meta=c.meta, kind="alternative", depth=node.depth + 1, relevance=rel))
            if n.kind not in ("alternative", "concurrent"):
                continue  # already in the graph as lineage; don't also draw as alternative
            g.add_edge(Edge(parent=node.key, child=n.key, role="alternative", directness=cls.directness,
                            inherited=cls.inherited, evidence=cls.evidence, source=cls.source,
                            section_kind=_section_of(c, cls.evidence)))
        motiv.sort(key=lambda x: (-x[1].directness, -x[0].score))
        for c, cls, rel in motiv[:2]:
            n = g.add_node(Node(meta=c.meta, kind="motivation", depth=1, relevance=rel))
            if n.kind != "motivation":
                continue
            g.add_edge(Edge(parent=n.key, child=node.key, role="problem_motivation", directness=cls.directness,
                            inherited=cls.inherited, evidence=cls.evidence, source=cls.source,
                            section_kind=_section_of(c, cls.evidence)))

    def _add_parent(self, g: LineageGraph, child: Node, c: Candidate, cls: Classification,
                    rel: float | None, heap_push) -> None:
        cfg = self.cfg
        meta = c.meta
        assert meta is not None
        if meta.key in g.nodes and g.nodes[meta.key].kind in ("alternative", "motivation", "concurrent"):
            g.nodes[meta.key].kind = "ancestor"  # promoted: real lineage beats side role
        existed = meta.key in g.nodes
        # cycle guard: child must not already be an ancestor of the parent
        if child.key in g.ancestors(meta.key):
            g.log.append({"child": child.meta.short_name, "paper": meta.title, "decision": "drop",
                          "reason": "would create a cycle"})
            return
        node = g.add_node(Node(meta=meta, kind="ancestor", depth=child.depth + 1, relevance=rel))
        g.add_edge(Edge(parent=node.key, child=child.key, role=cls.role, directness=cls.directness,
                        inherited=cls.inherited, evidence=cls.evidence, source=cls.source,
                        section_kind=_section_of(c, cls.evidence), score=c.score))
        g.log.append({"child": child.meta.short_name, "paper": meta.title, "decision": "keep",
                      "role": cls.role, "directness": cls.directness})
        if existed:
            return
        cc = meta.citation_count or 0
        if cc >= cfg.hub_citations or (cls.role == "component" and cc >= cfg.component_hub_citations):
            node.kind = "foundation"
            node.note = f"foundational/generic ({cc} citations); not expanded"
        elif not meta.arxiv_id:
            node.note = "not on arXiv; not expanded"
        elif node.depth >= cfg.max_depth:
            node.note = "depth limit"
        else:
            pri = -(ROLE_WEIGHT[cls.role] * cls.directness) * (rel if rel else 0.9) / (1 + 0.5 * node.depth)
            heap_push(pri, node.key)


def _section_of(c: Candidate, sentence: str) -> str:
    s = norm_ws(sentence)
    for x in c.contexts:
        if x.sentence == s or x.sentence == sentence:
            return x.section_kind
    return ""
