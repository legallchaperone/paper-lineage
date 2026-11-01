"""Core data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Roles a cited paper can play for the citing paper.
ROLES = (
    "builds_on",          # the child directly extends / is built on top of this paper (base model, framework)
    "technique_source",   # the child's core idea/technique is taken or adapted from this paper
    "component",          # a key ingredient used inside the method (teacher model, backbone, module)
    "alternative",        # a different approach to the same problem the child contrasts itself with
    "problem_motivation", # establishes the problem / goal the child addresses
    "baseline_only",      # only compared against in experiments
    "dataset_metric_tool",# dataset, benchmark, metric, solver, library
    "background",         # generic background / laundry-list citation
)
LINEAGE_ROLES = ("builds_on", "technique_source", "component")

ROLE_WEIGHT = {
    "builds_on": 1.0,
    "technique_source": 1.0,
    "component": 0.7,
    "alternative": 0.0,
    "problem_motivation": 0.0,
    "baseline_only": 0.0,
    "dataset_metric_tool": 0.0,
    "background": 0.0,
}

SECTION_KINDS = (
    "abstract", "intro", "related", "prelim", "method", "experiments",
    "discussion", "conclusion", "appendix", "figure", "other",
)


@dataclass
class PaperMeta:
    """Metadata for a paper, mostly from Semantic Scholar / arXiv."""

    key: str                      # canonical id used in the graph: "arXiv:2507.07982" or "S2:<paperId>"
    title: str
    year: int | None = None
    date: str | None = None       # ISO date (arXiv v1 when known)
    arxiv_id: str | None = None
    s2_id: str | None = None
    citation_count: int | None = None
    abstract: str | None = None
    embedding: list[float] | None = None
    alias: str | None = None      # how citing papers refer to it, e.g. "REPA", "DFoT"

    @property
    def url(self) -> str | None:
        if self.arxiv_id:
            return f"https://arxiv.org/abs/{self.arxiv_id}"
        if self.s2_id:
            return f"https://www.semanticscholar.org/paper/{self.s2_id}"
        return None

    @property
    def short_name(self) -> str:
        return self.alias or short_name(self.title)

    def to_dict(self, with_embedding: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not with_embedding:
            d.pop("embedding", None)
        d["url"] = self.url
        d["short_name"] = self.short_name
        return d


def short_name(title: str) -> str:
    """Compact display name (see quality.short_title: never cut mid-word)."""
    from .quality import short_title

    return short_title(title)


@dataclass
class BibEntry:
    key: str                  # e.g. "bib.bib79"
    text: str                 # full raw text of the entry
    label: str = ""           # e.g. "Wang et al. (2025b)" or "[12]"
    title: str | None = None  # structured title when available
    arxiv_id: str | None = None


@dataclass
class Citation:
    """One sentence in the citing paper that cites one or more bib entries."""

    keys: list[str]
    sentence: str
    section_kind: str
    section_title: str = ""
    # (offset of the citation group within `sentence`, keys in that group)
    spans: list[tuple[int, list[str]]] = field(default_factory=list)

    @property
    def group_size(self) -> int:
        return len(self.keys)


@dataclass
class ParsedPaper:
    arxiv_id: str
    title: str
    abstract: str
    bib: dict[str, BibEntry]
    citations: list[Citation]
    full_text: str = ""
    source: str = "html"   # "html" or "pdf"

    def contexts_for(self, bib_key: str) -> list[Citation]:
        return [c for c in self.citations if bib_key in c.keys]


@dataclass
class Candidate:
    """A cited paper being considered as a parent."""

    bib_key: str
    bib: BibEntry
    meta: PaperMeta | None
    contexts: list[Citation]
    score: float = 0.0
    features: dict[str, Any] = field(default_factory=dict)
    guess_role: str = "background"

    @property
    def title(self) -> str:
        if self.meta:
            return self.meta.title
        return self.bib.title or self.bib.text[:120]


@dataclass
class Classification:
    role: str
    directness: int            # 0-3
    inherited: str = ""        # what the child took from the parent
    evidence: str = ""         # verbatim sentence from the child paper
    relevance: float | None = None  # optional 0-3 judgment from the LLM
    source: str = "llm"        # "llm" or "heuristic"


@dataclass
class Node:
    meta: PaperMeta
    kind: str = "ancestor"     # target|ancestor|origin|foundation|frontier|alternative|motivation|descendant|concurrent
    depth: int = 0
    relevance: float | None = None
    expanded: bool = False
    note: str = ""             # why it was not expanded, etc.

    @property
    def key(self) -> str:
        return self.meta.key

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "depth": self.depth,
            "relevance": None if self.relevance is None else round(self.relevance, 4),
            "expanded": self.expanded,
            "note": self.note,
            **self.meta.to_dict(),
        }


@dataclass
class Edge:
    parent: str     # node key (older paper)
    child: str      # node key (newer paper, the one that cites)
    role: str
    directness: int
    inherited: str
    evidence: str
    section_kind: str = ""
    source: str = "llm"
    score: float = 0.0     # heuristic pre-score of the citation (tie-breaker between equally-rated parents)

    @property
    def weight(self) -> float:
        return ROLE_WEIGHT.get(self.role, 0.0) * max(self.directness, 0)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["weight"] = self.weight
        return d
