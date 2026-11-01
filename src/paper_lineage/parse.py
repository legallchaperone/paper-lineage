"""Parse papers into sections, bibliography entries and citing sentences.

Primary source is arXiv's LaTeXML HTML (arxiv.org/html/<id>), whose <cite> elements link to
exact bibliography entries. A lower-confidence plain-text (PDF) parser is the fallback.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from bs4 import BeautifulSoup, NavigableString, Tag

from .arxiv import find_arxiv_id

try:  # lxml is faster; the stdlib parser gives the same structure for LaTeXML pages
    import lxml  # noqa: F401
    HTML_PARSER = "lxml"
except ImportError:  # pragma: no cover - depends on the environment
    HTML_PARSER = "html.parser"
import os as _os

HTML_PARSER = _os.environ.get("LINEAGE_HTML_PARSER", HTML_PARSER)
from .models import BibEntry, Citation, ParsedPaper

# ---------------------------------------------------------------------------
# section classification

_KIND_PATTERNS: list[tuple[str, str]] = [
    ("abstract", r"^abstract"),
    ("intro", r"introduction|^overview$"),
    ("related", r"related work|prior work|previous work|literature|related research|background and related"),
    ("prelim", r"preliminar|^background|problem (formulation|setup|statement)|notation"),
    ("experiments", r"experiment|evaluation|results|ablation|benchmark|empirical"),
    ("discussion", r"discussion|limitation|analysis|future work|broader impact"),
    ("conclusion", r"conclusion|summary|concluding"),
    ("skip", r"acknowledg|reference|bibliograph|author contribution|checklist"),
]


def classify_section_title(title: str) -> str | None:
    t = re.sub(r"^[\dA-Z]+(\.\d+)*\s+", "", title.strip()).lower()
    t = re.sub(r"^appendix\s+[a-z\d]*\s*", "", t)
    for kind, pat in _KIND_PATTERNS:
        if re.search(pat, t):
            return kind
    return None


def assign_section_kinds(titles: list[str], is_appendix: list[bool]) -> list[str]:
    """Keyword kinds, then: unknown sections between intro and experiments are 'method'."""
    kinds: list[str | None] = [classify_section_title(t) for t in titles]
    out: list[str] = []
    seen_exp = False
    for kind, app in zip(kinds, is_appendix):
        if app:
            out.append("skip" if kind == "skip" else "appendix")
            continue
        if kind == "experiments":
            seen_exp = True
        if kind is None:
            kind = "discussion" if seen_exp else "method"
        out.append(kind)
    return out


# ---------------------------------------------------------------------------
# sentence splitting

_ABBREV = {
    "e.g", "i.e", "etc", "vs", "fig", "figs", "eq", "eqs", "sec", "secs", "tab", "al", "cf",
    "resp", "approx", "no", "app", "ref", "refs", "ch", "dr", "prof", "st", "viz", "inc",
}
_SPLIT_RE = re.compile(r"[.!?][\"')\]]*\s+")


def split_sentences(text: str, protected: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Returns (start, end) spans of sentences; never splits inside protected spans (citations)."""
    spans = []
    start = 0
    for m in _SPLIT_RE.finditer(text):
        pos = m.start()
        if any(a <= pos < b for a, b in protected):
            continue
        nxt = text[m.end():m.end() + 1]
        if nxt and nxt.islower():
            continue
        word = re.search(r"([A-Za-z.]+)$", text[start:pos])
        if word and word.group(1).lower().rstrip(".") in _ABBREV:
            continue
        if word and len(word.group(1)) == 1 and word.group(1).isupper():  # initials "J. Smith"
            continue
        spans.append((start, m.end()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def norm_label(s: str) -> str:
    s = s.lower().replace("et al.", " ").replace("et al", " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# LaTeXML HTML


class _Builder:
    """Flattens an element to text while tracking where citations sit."""

    def __init__(self, label_index: dict[str, str], root: Tag | None = None):
        self.root = root
        self.parts: list[str] = []
        self.len = 0
        self.cites: list[tuple[int, int, list[str]]] = []
        self.label_index = label_index

    def add(self, s: str) -> None:
        if s:
            self.parts.append(s)
            self.len += len(s)

    def walk(self, el: Tag | NavigableString) -> None:
        if isinstance(el, NavigableString):
            self.add(str(el))
            return
        if not isinstance(el, Tag):
            return
        cls = el.get("class") or []
        if el is not self.root and (
            el.name in ("section", "figure", "figcaption", "table") and self.root is not None
            and self.root.name != "tr" or (el.name == "p" and "ltx_p" in cls)
        ):
            return  # never swallow neighbouring blocks (malformed HTML nests them)
        if el.name == "cite" or "ltx_cite" in cls:
            keys = cite_keys(el, self.label_index)
            text = norm_ws(el.get_text(" "))
            start = self.len
            self.add(text)
            if keys:
                self.cites.append((start, self.len, keys))
            return
        if el.name == "math":
            # keep formulas as delimited LaTeX so the report can typeset them (instead of raw code)
            tex = " ".join((el.get("alttext") or "").split())
            self.add(f"${tex.replace('$', '')}$" if tex else "[math]")
            return
        if "ltx_note" in cls or "ltx_tag_note" in cls or el.name in ("script", "style"):
            return  # footnotes would glue foreign text into sentences
        for ch in el.children:
            self.walk(ch)

    def text(self) -> str:
        return "".join(self.parts)


def cite_keys(cite: Tag, label_index: dict[str, str]) -> list[str]:
    keys: list[str] = []
    for ref in cite.select(".ltx_ref"):
        href = ref.get("href", "") if ref.name == "a" else ""
        if href.startswith("#bib."):
            keys.append(href[1:])
        else:
            k = label_index.get(norm_label(ref.get_text(" ")))
            if k:
                keys.append(k)
    if not keys:  # e.g. numeric "[12, 14]" without links
        for piece in re.split(r"[;,]", cite.get_text(" ")):
            k = label_index.get(norm_label(piece))
            if k:
                keys.append(k)
    return list(dict.fromkeys(keys))


def _bib_from_html(soup: BeautifulSoup) -> dict[str, BibEntry]:
    bib: dict[str, BibEntry] = {}
    for li in soup.select("li.ltx_bibitem"):
        key = li.get("id")
        if not key:
            continue
        tag = li.select_one(".ltx_tag_bibitem")
        label = norm_ws(tag.get_text(" ")) if tag else ""
        blocks = [b for b in li.select(".ltx_bibblock") if "ltx_bib_cited" not in (b.get("class") or [])]
        texts = [norm_ws(b.get_text(" ")) for b in blocks]
        text = " ".join(t for t in texts if t)
        t_el = li.select_one(".ltx_bib_title")
        if t_el:
            title = norm_ws(t_el.get_text(" "))
        elif len(texts) >= 2:
            title = texts[1]
        else:
            title = None
        if title:
            title = title.strip().rstrip(".").strip()
        hrefs = " ".join(a.get("href", "") for a in li.select("a"))
        bib[key] = BibEntry(key=key, text=text, label=label, title=title,
                            arxiv_id=find_arxiv_id(text + " " + hrefs))
    return bib


def _label_index(bib: dict[str, BibEntry]) -> dict[str, str]:
    idx: dict[str, str] = {}
    for k, b in bib.items():
        if b.label:
            idx.setdefault(norm_label(b.label), k)
    return idx


def parse_latexml(html: str, arxiv_id: str = "") -> ParsedPaper:
    soup = BeautifulSoup(html, HTML_PARSER)
    for br in soup.select("br"):
        br.replace_with(" ")
    bib = _bib_from_html(soup)
    label_index = _label_index(bib)

    title_el = soup.select_one("h1.ltx_title_document") or soup.find("title")
    title = ""
    if title_el:
        for note in title_el.select(".ltx_note"):
            note.decompose()
        title = norm_ws(title_el.get_text(" "))

    abstract_el = soup.select_one("div.ltx_abstract")
    abstract = ""
    if abstract_el:
        h = abstract_el.find(re.compile(r"^h[1-6]$"))
        if h:
            h.extract()
        abstract = norm_ws(abstract_el.get_text(" "))

    # Section structure from LaTeXML ids ("S4", "S4.SS1", "A5"), not from DOM nesting:
    # real-world arXiv HTML is sometimes malformed (e.g. unclosed <figure>), which nests
    # later sections inside earlier elements.
    top_titles: dict[str, str] = {}
    sub_titles: dict[str, str] = {}
    for h in soup.select(".ltx_title_section, .ltx_title_appendix, .ltx_title_chapter"):
        sec = h.find_parent("section")
        if sec is not None and sec.get("id"):
            top_titles.setdefault(sec["id"], norm_ws(h.get_text(" ")))
    for h in soup.select(".ltx_title_subsection, .ltx_title_subsubsection"):
        sec = h.find_parent("section")
        if sec is not None and sec.get("id"):
            sub_titles.setdefault(sec["id"], norm_ws(h.get_text(" ")))
    order = list(top_titles)
    kinds = assign_section_kinds([top_titles[i] for i in order], [i.startswith("A") for i in order])
    top_kind = dict(zip(order, kinds))

    citations: list[Citation] = []
    texts: list[str] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    def locate(el: Tag) -> tuple[str, str]:
        """(kind, section title) for an element, from its own or nearest ancestor id."""
        eid = el.get("id")
        if not eid:
            anc = el.find_parent(id=True)
            eid = anc.get("id") if anc is not None else ""
        parts = eid.split(".")
        top = parts[0]
        if top not in top_kind:
            return ("abstract" if el.find_parent(class_="ltx_abstract") else "other"), ""
        kind = top_kind[top]
        if len(parts) > 1 and kind in ("method", "other") and parts[1].startswith("SS"):
            k2 = classify_section_title(sub_titles.get(".".join(parts[:2]), ""))
            if k2 in ("prelim", "related"):
                kind = k2
        return kind, top_titles[top]

    def in_float(el: Tag) -> bool:
        eid = el.get("id") or ""
        return any(re.fullmatch(r"(F|T|tab|fig|alg|A?lg)\d+", seg) for seg in eid.split("."))

    def harvest_block(el: Tag, kind: str, sec_title: str, split: bool = True) -> None:
        b = _Builder(label_index, root=el)
        b.walk(el)
        text, cites = _collapse_ws(b.text(), b.cites)
        if not text:
            return
        texts.append(text)
        if not cites:
            return
        spans = split_sentences(text, [(a, z) for a, z, _ in cites]) if split else [(0, len(text))]
        for s0, s1 in spans:
            keys: list[str] = []
            for a, z, ks in cites:
                if a >= s0 and z <= s1 + 1:
                    keys.extend(ks)
            if not keys:
                continue
            raw_sent = text[s0:s1]
            lead = len(raw_sent) - len(raw_sent.lstrip())
            sent = raw_sent.strip()
            sig = (sent, tuple(dict.fromkeys(keys)))
            if sig in seen:
                continue
            seen.add(sig)
            spans_ = [(a - s0 - lead, ks) for a, z, ks in cites if a >= s0 and z <= s1 + 1]
            citations.append(Citation(keys=list(sig[1]), sentence=sent, section_kind=kind,
                                      section_title=sec_title, spans=spans_))

    if abstract_el is not None:
        harvest_block(abstract_el, "abstract", "Abstract")
    for p in soup.select("p.ltx_p"):
        if (p.get("id") or "").startswith("bib") or p.find_parent(class_="ltx_abstract"):
            continue
        if p.find_parent("figcaption") is not None or in_float(p):
            continue
        kind, t = locate(p)
        if kind == "skip":
            continue
        harvest_block(p, kind, t)
    for cap in soup.select("figcaption"):
        kind, t = locate(cap)
        if kind != "skip":
            harvest_block(cap, "figure", t)
    for row in soup.select("table.ltx_tabular tr"):
        if row.select_one("cite, .ltx_cite"):
            kind, t = locate(row)
            if kind != "skip":
                harvest_block(row, "table", t, split=False)

    return ParsedPaper(arxiv_id=arxiv_id, title=title, abstract=abstract, bib=bib,
                       citations=citations, full_text="\n".join(texts), source="html")


def _collapse_ws(raw: str, cites: list[tuple[int, int, list[str]]]):
    """Collapse whitespace runs to single spaces, remapping citation span offsets."""
    out: list[str] = []
    mapping = [0] * (len(raw) + 1)
    n = 0
    prev_space = True
    for i, ch in enumerate(raw):
        mapping[i] = n
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                n += 1
            prev_space = True
        else:
            out.append(ch)
            n += 1
            prev_space = False
    mapping[len(raw)] = n
    text = "".join(out)
    stripped = text.rstrip()
    new = [(mapping[a], min(mapping[z], len(stripped)), ks) for a, z, ks in cites]
    return stripped, new


# ---------------------------------------------------------------------------
# Plain-text (PDF) fallback — lower confidence.

def _despace_caps(line: str) -> str:
    """ICLR-style small caps come out of pdftotext as 'R ELATED W ORK' -> 'RELATED WORK'."""
    return re.sub(r"\b([A-Z]) (?=[A-Z]{2,}\b)", r"\1", line)


def _is_heading(line: str, prev: str, prev_nonblank: str = "") -> bool:
    t = line.strip()
    if not t or len(t) > 70 or t.endswith((".", ",", ";")) or not t[0].isalnum():
        return False
    if prev.strip() and not re.fullmatch(r"[\dA-Z](\.\d+)*\.?", prev.strip()):
        return False
    prev = prev if prev.strip() else prev_nonblank
    body = re.sub(r"^[\dA-Z]+(\.\d+)*\.?\s+", "", t)
    if len(body.split()) > 7 or not body[:1].isupper():
        return False
    if classify_section_title(t) is not None:
        return True
    numbered = bool(re.fullmatch(r"\d+", prev.strip())) or bool(re.match(r"^\d+\.?\s+[A-Z]", t))
    return numbered and len(body.split()) <= 6


def parse_plaintext(text: str, arxiv_id: str = "", title: str = "", abstract: str = "") -> ParsedPaper:
    """Heuristic parse of pdftotext output. Supports numeric and author-year citations."""
    raw = [_despace_caps(ln) for ln in text.splitlines()]
    freq: dict[str, int] = {}
    for ln in raw:
        if ln.strip():
            freq[ln.strip()] = freq.get(ln.strip(), 0) + 1
    lines = [ln for ln in raw
             if not re.fullmatch(r"\s*\d{1,3}\s*", ln)
             and not (freq.get(ln.strip(), 0) >= 3 and len(ln.strip()) < 80 and len(ln.split()) > 2)]
    # locate the last References heading
    ref_idx = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*(\d+\.?\s*)?(references|bibliography)\s*$", ln, re.I):
            ref_idx = i
    body_lines = lines[:ref_idx] if ref_idx is not None else lines
    ref_lines = lines[ref_idx + 1:] if ref_idx is not None else []
    # appendix after references: stop the bibliography at an "Appendix"/"A <Title>" heading
    for j, ln in enumerate(ref_lines):
        if j > 5 and re.match(r"^\s*(Appendix|APPENDIX|A\s+[A-Z][A-Za-z]+)", ln) and len(ln.strip()) < 70:
            ref_lines = ref_lines[:j]
            break

    bib = _plaintext_bib(ref_lines)
    numeric = any(b.label.startswith("[") for b in bib.values())

    # sections
    titles: list[str] = []
    blocks: list[list[str]] = [[]]
    prev, prev_nb = "", ""
    for ln in body_lines:
        if _is_heading(ln, prev, prev_nb):
            titles.append(ln.strip())
            blocks.append([])
        else:
            blocks[-1].append(ln)
        prev = ln
        if ln.strip():
            prev_nb = ln
    titles = ["Front"] + titles
    kinds = assign_section_kinds(titles, [False] * len(titles))
    kinds[0] = "intro"

    # author-year index: (surname, year) -> key
    groups: dict[tuple[str, str], list[str]] = {}
    for k, b in bib.items():
        ys = re.findall(r"\b((?:19|20)\d{2})[a-z]?\b", b.text)
        surname = _surname_from_bib(b.text)
        if ys and surname:
            groups.setdefault((surname.lower(), ys[-1]), []).append(k)
    ay_index: dict[tuple[str, str], str] = {}
    for (sn, y), keys in groups.items():
        ay_index[(sn, y)] = keys[0]
        for i, k in enumerate(keys[:26]):
            ay_index[(sn, y + chr(ord("a") + i))] = k

    citations: list[Citation] = []
    full: list[str] = []
    for kind, t, blk in zip(kinds, titles, blocks):
        if kind == "skip":
            continue
        para = norm_ws(" ".join(blk).replace("- ", ""))
        full.append(para)
        cites: list[tuple[int, int, list[str]]] = []
        if numeric:
            for m in re.finditer(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]", para):
                keys = []
                for part in re.split(r"\s*,\s*", m.group(1)):
                    if re.search(r"[–-]", part):
                        a, z = re.split(r"\s*[–-]\s*", part)[:2]
                        rng = range(int(a), int(z) + 1) if a.isdigit() and z.isdigit() and int(z) - int(a) < 30 else []
                        keys += [f"ref{i}" for i in rng]
                    elif part.strip().isdigit():
                        keys.append(f"ref{part.strip()}")
                keys = [k for k in keys if k in bib]
                if keys:
                    cites.append((m.start(), m.end(), keys))
        else:
            for m in re.finditer(
                r"([A-Z][A-Za-z\-'’]+)(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][A-Za-z\-'’]+)?,?\s*\(?((?:19|20)\d{2}[a-z]?)\)?",
                para,
            ):
                k = ay_index.get((m.group(1).lower(), m.group(2))) or ay_index.get(
                    (m.group(1).lower(), m.group(2)[:4]))
                if k:
                    cites.append((m.start(), m.end(), [k]))
        if not cites:
            continue
        for s0, s1 in split_sentences(para, [(a, z) for a, z, _ in cites]):
            inside = [(a, ks) for a, z, ks in cites if a >= s0 and z <= s1 + 1]
            keys = [k for _, ks in inside for k in ks]
            if keys:
                raw_sent = para[s0:s1]
                lead = len(raw_sent) - len(raw_sent.lstrip())
                citations.append(Citation(keys=list(dict.fromkeys(keys)), sentence=raw_sent.strip(),
                                          section_kind=kind, section_title=t,
                                          spans=[(a - s0 - lead, ks) for a, ks in inside]))
    return ParsedPaper(arxiv_id=arxiv_id, title=title, abstract=abstract, bib=bib, citations=citations,
                       full_text="\n".join(full), source="pdf")


def _surname_from_bib(text: str) -> str:
    first = re.split(r",| and ", text, maxsplit=1)[0].strip()
    toks = first.split()
    if not toks:
        return ""
    # "Smith, J." style -> the single token; "John Smith" / "J. Smith" style -> last token
    if len(toks) == 1:
        return toks[0].strip(".")
    return toks[-1].strip(".")


_AUTHOR_START = re.compile(
    r"^([A-Z][\w\-'’.]*(?:\s+[A-Z][\w\-'’.]*){0,3},\s|"     # "Haoyu Wu, ..."
    r"[A-Z][\w\-'’.]*(?:\s+[A-Z][\w\-'’.]*){0,3}\s+and\s+[A-Z]|"  # "David Ha and Jürgen ..."
    r"[A-Z][\w\-]+\.\s)"                                     # "OpenAI. Sora ..."
)
_NOT_START = re.compile(r"^(In|URL|arXiv|Proceedings|Advances|IEEE|International|Conference|Journal|"
                        r"Transactions|pp\.|doi|Springer|PMLR|The\s)\b")


def _plaintext_bib(lines: Iterable[str]) -> dict[str, BibEntry]:
    entries: list[tuple[str, str]] = []
    cur_label, cur = "", []
    numeric_re = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")
    lines = list(lines)
    numeric = sum(1 for ln in lines if numeric_re.match(ln)) >= 5
    for ln in lines:
        if numeric:
            m = numeric_re.match(ln)
            if m:
                if cur:
                    entries.append((cur_label, " ".join(cur)))
                cur_label, cur = f"[{m.group(1)}]", [m.group(2)]
            elif ln.strip():
                cur.append(ln.strip())
        else:  # author-year: a new entry starts with an author list after a line ending in "."
            t = ln.strip()
            if not t:
                continue
            prev_end = cur[-1].rstrip() if cur else "."
            starts = bool(_AUTHOR_START.match(t)) and not _NOT_START.match(t)
            if cur and prev_end.endswith(".") and starts:
                entries.append(("", " ".join(cur)))
                cur = []
            cur.append(t)
    if cur:
        entries.append((cur_label, " ".join(cur)))
    bib: dict[str, BibEntry] = {}
    for i, (label, text) in enumerate(entries):
        text = norm_ws(text.replace("- ", ""))
        if len(text) < 20:
            continue
        key = f"ref{label.strip('[]')}" if label else f"ref{i + 1}"
        bib[key] = BibEntry(key=key, text=text, label=label, title=None, arxiv_id=find_arxiv_id(text))
    return bib
