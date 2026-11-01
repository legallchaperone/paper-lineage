"""One place that guarantees consistent output quality, whatever paper is traced.

Every graph goes through `normalize()` before anything is rendered: freshly computed runs (engine) and
saved runs loaded with `--render` alike. Renderers can then rely on a few invariants instead of
patching special cases:

* summaries ("what was inherited") are clean noun phrases that complete "X builds on Y: …"
  (no trailing period, lower-case first letter unless it is a name/acronym, bounded length);
* every paper has a short display name that is never chopped mid-word, and names are unique;
* lineage edges carry their evidence sentence.

`check()` lists violations of the invariants, so tests and `--verbose` can catch regressions on any run.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import LineageGraph

MAX_SUMMARY_WORDS = 18
MAX_NAME_CHARS = 48

# words that start a qualifier in a title: "Scaling X for Y", "X with Y", "X are Y"
_CUT_WORDS = {"for", "with", "via", "through", "using", "by", "from", "towards", "toward", "in", "on", "under",
              "beyond", "without", "is", "are", "as", "and", "at", "to", "into", "meets", "of"}
_TITLE_SEPARATORS = re.compile(r"\s*(?::|\s--?\s|\s—\s|\s–\s)\s*")
_ECHO = re.compile(r"^(?:it|they|the child(?: paper)?|this paper|the paper|we)\s+"
                   r"(?:builds?|built|takes?|took|uses?|used|adopts?|adopted|borrows?|borrowed|inherits?|inherited)"
                   r"(?:\s+(?:on|upon|from))?\s+(?:its|their|the)?\s*", re.I)


def is_namelike(word: str) -> bool:
    """'VGGT', 'DiT', 'SigLIP', 'π0', 'CLIP-based' look like names; 'Base', 'A', 'The' do not."""
    w = word.strip("\"'“”‘’()[],.;:")
    if not w:
        return False
    if any(ch.isdigit() for ch in w):
        return True
    caps = sum(ch.isupper() for ch in w)
    return caps >= 2 or (caps == 1 and not w[0].isupper())


def clean_summary(text: str | None) -> str:
    """Normalize an inherited-summary into a phrase that reads after a colon."""
    t = " ".join(str(text or "").split())
    t = t.strip(" \"'“”‘’")
    t = _ECHO.sub("", t)
    t = re.sub(r"[\s.;,:!]+$", "", t)
    if not t:
        return ""
    words = t.split(" ")
    if len(words) > MAX_SUMMARY_WORDS:
        words = words[:MAX_SUMMARY_WORDS]
        while words and words[-1].lower() in _CUT_WORDS | {"a", "an", "the"}:
            words.pop()
        words[-1] = words[-1].rstrip(",;:") + "…"
    if words and words[0][:1].isupper() and not is_namelike(words[0]):
        words[0] = words[0][0].lower() + words[0][1:]
    return " ".join(words)


def short_title(title: str) -> str:
    """Compact display name from a title, never cut mid-word.

    1. the part before a ':' / ' - ' when it is short (method names: 'VGGT: Visual Geometry …' -> 'VGGT');
    2. the whole title if it fits;
    3. the title up to the first qualifier word ('Scaling Rectified Flow Transformers for …' ->
       'Scaling Rectified Flow Transformers');
    4. otherwise whole words up to the limit, with an ellipsis.
    """
    t = " ".join((title or "").split()).rstrip(".")
    if not t:
        return "?"
    parts = _TITLE_SEPARATORS.split(t, maxsplit=1)
    if len(parts) == 2 and 0 < len(parts[0]) <= 42:
        return parts[0]
    if len(t) <= MAX_NAME_CHARS:
        return t
    words = t.split(" ")
    for i in range(3, len(words)):
        if words[i].lower() in _CUT_WORDS:
            cand = " ".join(words[:i])
            if len(cand) <= MAX_NAME_CHARS:
                return cand
            break
    out = ""
    for w in words:
        if len(out) + len(w) + 1 > MAX_NAME_CHARS - 1:
            break
        out = f"{out} {w}".strip()
    return (out or words[0]) + "…"


_ACRONYM_SKIP = _CUT_WORDS | {"a", "an", "the"}


def acronym(title: str, max_chars: int = 6) -> str:
    """Initials of a title's content words, for papers without a method name:
    'Possible Principles Underlying the Transformations of Sensory Messages' -> 'PPUTSM'.
    Name-like tokens ('GAN', '3D') are kept whole; hyphenated words give one initial per part."""
    out = ""
    for tok in re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", title or ""):
        if tok.lower() in _ACRONYM_SKIP:
            continue
        piece = "".join(p if is_namelike(p) else p[0].upper() for p in re.split(r"-", tok.replace("’", "'")) if p)
        piece = piece.replace("'", "")
        if out and len(out) + len(piece) > max_chars:
            break
        out += piece
    return out or short_title(title)


def abbreviation(meta) -> str:
    """The name the walkthrough uses once a paper has been introduced by its full title: its method name
    or alias when it has one ('VGGT', 'Barlow Twins', 'Scaling Rectified Flow Transformers'), otherwise
    an acronym, so a name is never an ellipsis-cut title."""
    nm = meta.short_name
    return acronym(meta.title) if nm.endswith("…") else nm


def normalize(g: LineageGraph) -> LineageGraph:
    """Apply the invariants in place (idempotent)."""
    for e in g.edges:
        e.inherited = clean_summary(e.inherited)
    # unique display names: a later paper with the same name gets its year appended
    seen: dict[str, str] = {}
    for k in sorted(g.nodes, key=g._order):
        m = g.nodes[k].meta
        if m.alias:
            m.alias = " ".join(m.alias.split())
            if m.alias.endswith("...") or len(m.alias) > MAX_NAME_CHARS:   # legacy hard-truncated names
                m.alias = None
        name = m.short_name.lower()
        if name in seen and seen[name] != k:
            base = m.short_name
            m.alias = f"{base} ’{str(m.year)[-2:]}" if m.year else f"{base} (2)"
        seen[m.short_name.lower()] = k
    return g


def check(g: LineageGraph) -> list[str]:
    """Invariant violations (empty list = clean)."""
    from .models import LINEAGE_ROLES

    issues: list[str] = []
    names: dict[str, str] = {}
    for k, n in g.nodes.items():
        nm = n.meta.short_name
        if len(nm) > MAX_NAME_CHARS + 1:
            issues.append(f"name too long: {nm!r}")
        if nm.endswith("...") or nm.endswith("-…"):
            issues.append(f"name cut mid-word: {nm!r}")
        if nm.lower() in names and names[nm.lower()] != k:
            issues.append(f"duplicate name: {nm!r}")
        names[nm.lower()] = k
    for e in g.edges:
        if e.inherited != clean_summary(e.inherited):
            issues.append(f"summary not normalized: {e.inherited!r}")
        if e.role in LINEAGE_ROLES and not e.evidence:
            issues.append(f"lineage edge without evidence: {e.parent} -> {e.child}")
    return issues
