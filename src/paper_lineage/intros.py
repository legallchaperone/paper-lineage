"""A short introduction for every paper, grounded in its own abstract.

Two layers, both stored on `PaperMeta`:

* `intro_quote`: one sentence copied verbatim from the abstract, the one that says what the paper does
  ("we propose ...", "VGGT is a ..."), or else the first sentence. Deterministic, no LLM; computed for
  every paper with an abstract (also when a saved run is re-rendered).
* `intro`: an optional one-sentence plain-language summary written by the LLM from the abstract only.
  It is kept only if it stays inside the abstract's vocabulary (see `grounded`); otherwise it is dropped
  and the verbatim sentence is shown instead.

A paper without an abstract gets no introduction: nothing is guessed from the title.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import LineageGraph
    from .llm import LLM
    from .models import PaperMeta

log = logging.getLogger(__name__)

MAX_INTRO_WORDS = 40
LLM_BATCH = 15

_ABBR = re.compile(r"\b(e\.g|i\.e|et al|etc|vs|cf|Fig|Figs|Eq|Eqs|Sec|approx|resp|w\.r\.t|a\.k\.a|No)\.", re.I)
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"“$])")
_DOES = re.compile(r"(?<![\w-])(?:propos|introduc|present|develop|design|describ|formulat|build|offer|refin|extend|"
                   r"adapt|unif|combin|leverag|tackl|construct|devis|creat|releas)\w*", re.I)
_FINDS = re.compile(r"(?<![\w-])(?:show|demonstrat|prove|proving|stud|investigat|explor|establish|argu|analy[sz]|find|"
                    r"reveal|examin|revisit)\w*", re.I)
_ASIDE = re.compile(r"(?:we (?:believe|hope|expect|anticipate)|(?:our |the )?(?:code|models?|data(?:set)?s?|"
                    r"weights) (?:is|are|will be) (?:publicly |freely )?(?:available|released))", re.I)
_SELF = re.compile(r"\b(?:we|our|this (?:paper|work|article|study|note)|here)\b", re.I)


def sentences(text: str | None) -> list[str]:
    """Split an abstract into sentences (keeps 'e.g.', 'et al.', '$x.y$' intact)."""
    t = " ".join(str(text or "").split())
    if not t:
        return []
    maths: list[str] = []

    def hide(m: re.Match) -> str:
        maths.append(m.group(0))
        return f"\x00{len(maths) - 1}\x00"

    t = re.sub(r"\$[^$]*\$", hide, t)
    t = _ABBR.sub(lambda m: m.group(0)[:-1] + "\x01", t)
    out = []
    for s in _SPLIT.split(t):
        s = s.replace("\x01", ".")
        s = re.sub(r"\x00(\d+)\x00", lambda m: maths[int(m.group(1))], s)
        if s.strip():
            out.append(s.strip())
    return out


def _names(title: str, alias: str = "") -> list[str]:
    """Method names a paper may call itself: its alias and name-like title words ('NaViT', 'VGGT')."""
    from .quality import is_namelike

    out = [alias] if alias else []
    for w in re.findall(r"[\w-]+", title or ""):
        if is_namelike(w) and len(w) >= 3 and not w.isdigit() and w not in out:
            out.append(w)
    return out


def pick_sentence(abstract: str | None, title: str = "", alias: str = "") -> str:
    """The abstract sentence that says what the paper does, verbatim ('' without an abstract).

    Preference: a sentence where the paper introduces itself (we/this paper + propose/introduce/... or
    its method name, or '<Name> is ...'); then one stating a finding (we show/study/...); then the first."""
    ss = [s for s in sentences(abstract) if len(s.split()) >= 5]
    if not ss:
        return ""
    core = [s for s in ss if not _ASIDE.match(s)] or ss
    names = _names(title, alias)

    def named(s: str) -> bool:
        return any(re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", s) for n in names)

    def opens_with_name(s: str) -> bool:
        return any(s.startswith(n + " is ") or s.startswith(n + ", ") for n in names)

    for test in (lambda s: opens_with_name(s) or (_SELF.search(s) and _DOES.search(s) and named(s)),
                 lambda s: _SELF.search(s) and (_DOES.search(s) or named(s)),
                 lambda s: _SELF.search(s) and _FINDS.search(s)):
        for s in core:
            if test(s):
                return s
    return ss[0]


# ------------------------------------------------------------------------------------------- grounding
_STOP = set("""a an the and or of for to in on with by from as at is are was were be been being this that these
those it its their our we they which who whom whose than then there here into onto over under via using use
uses used based new novel approach method methods paper work model models propose proposes proposed
introduce introduces present presents framework simple show shows can also such each both more most
while where when how what not only allow allows enable enables without through between across""".split())


def _stem(w: str) -> str:
    return w[:5]


def _content(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 4 and w not in _STOP]


def grounded(intro: str, abstract: str, title: str = "") -> bool:
    """True if a generated intro stays inside the abstract: every number it gives appears there, and
    at least 80% of its content words (compared by their first five letters) do too."""
    src = f"{title} {abstract}".lower()
    src_stems = {_stem(w) for w in re.findall(r"[a-z0-9]+", src)}
    for num in re.findall(r"\d+(?:\.\d+)?", intro):
        if num not in src:
            return False
    words = _content(intro)
    if not words:
        return False
    hits = sum(_stem(w) in src_stems for w in words)
    return hits / len(words) >= 0.8


def clean_intro(text: str | None) -> str:
    t = " ".join(str(text or "").split()).strip(" \"'“”")
    if not t:
        return ""
    t = t[0].upper() + t[1:]
    if not t.endswith((".", "!", "?")):
        t += "."
    return t


# ------------------------------------------------------------------------------------------- graph-level
def ensure_quotes(g: LineageGraph) -> None:
    """Fill `intro_quote` for every paper with an abstract (idempotent, no network, no LLM)."""
    for n in g.nodes.values():
        m = n.meta
        if m.abstract and not (m.intro_quote and m.intro_quote in " ".join(m.abstract.split())):
            m.intro_quote = pick_sentence(m.abstract, m.title, m.alias or "") or None
        if not m.abstract:
            m.intro_quote = None
            m.intro = None


SYSTEM = """You write one-sentence introductions of research papers for readers skimming a map of \
how papers build on each other.

Rules:
- Use ONLY the abstract given for that paper. Do not add facts, numbers, results, or claims that are \
not in it, and do not use outside knowledge.
- One plain sentence, at most 30 words, in the present tense, saying what the paper proposes or does \
(and, if the abstract says so, what it is for). Start with the paper's method name if the abstract \
gives one, otherwise with "A method that ...", "A study of ...", etc.
- No hype words ("novel", "state-of-the-art", "groundbreaking").
- Reply with a single JSON object and nothing else: {"results": [{"id": "P1", "intro": "..."}]}"""


def _llm_intros(llm: LLM, metas: list[PaperMeta]) -> dict[str, str]:
    """{key: intro} for the papers whose generated intro passed `grounded`."""
    out: dict[str, str] = {}
    for start in range(0, len(metas), LLM_BATCH):
        chunk = metas[start:start + LLM_BATCH]
        ids = {f"P{i + 1}": m for i, m in enumerate(chunk)}
        user = "\n\n".join(f"[{pid}] Title: {m.title}\nAbstract: {' '.join((m.abstract or '').split())[:1500]}"
                           for pid, m in ids.items())
        try:
            reply = llm.complete_json(SYSTEM, user)
        except Exception as e:   # intros are a nicety: a failed call must not fail the run
            log.warning("intro LLM call failed: %s", e)
            continue
        for r in reply.get("results") or []:
            if not isinstance(r, dict):
                continue
            m = ids.get(str(r.get("id", "")).strip("[] "))
            text = clean_intro(r.get("intro"))
            if m is None or not text:
                continue
            if len(text.split()) > MAX_INTRO_WORDS or not grounded(text, m.abstract or "", m.title):
                log.info("intro for %r not grounded in its abstract; using the abstract sentence", m.title)
                continue
            out[m.key] = text
    return out


def fill_abstracts(g: LineageGraph, s2=None, arxiv=None) -> None:
    """Fetch missing abstracts (Semantic Scholar batch, then the arXiv API). Best effort."""
    missing = [n.meta for n in g.nodes.values() if not n.meta.abstract]
    if missing and s2 is not None:
        try:
            from .resolve import enrich
            enrich(s2, missing)
        except Exception as e:
            log.warning("abstract lookup (S2) failed: %s", e)
    missing = [m for m in missing if not m.abstract and m.arxiv_id]
    if missing and arxiv is not None:
        try:
            got = arxiv.metadata([m.arxiv_id for m in missing])
            for m in missing:
                m.abstract = (got.get(m.arxiv_id) or {}).get("abstract") or None
        except Exception as e:
            log.warning("abstract lookup (arXiv) failed: %s", e)


def add_intros(g: LineageGraph, llm: LLM | None = None, s2=None, arxiv=None) -> None:
    """Give every paper an introduction: the verbatim abstract sentence, plus an LLM summary if available."""
    fill_abstracts(g, s2, arxiv)
    ensure_quotes(g)
    stats = {"with_abstract": sum(1 for n in g.nodes.values() if n.meta.abstract), "llm": 0}
    if llm is not None:
        metas = sorted((n.meta for n in g.nodes.values() if n.meta.abstract and not n.meta.intro),
                       key=lambda m: g._order(m.key))
        for k, text in _llm_intros(llm, metas).items():
            g.nodes[k].meta.intro = text
        stats["llm"] = sum(1 for n in g.nodes.values() if n.meta.intro)
    g.stats["intros"] = stats
