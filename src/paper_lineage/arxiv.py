"""arXiv: id normalization, metadata, and full-text retrieval."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from .http import Http, HttpError

log = logging.getLogger(__name__)

NEW_ID = r"\d{4}\.\d{4,5}"
OLD_ID = r"[a-z\-]+(?:\.[A-Z]{2})?/\d{7}"
ID_RE = re.compile(rf"({NEW_ID}|{OLD_ID})(v\d+)?", re.I)
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def normalize_arxiv_id(s: str) -> str:
    """Accepts urls (abs/pdf/html), 'arXiv:XXXX', ids with versions; returns the bare id."""
    s = s.strip()
    s = re.sub(r"^arxiv:\s*", "", s, flags=re.I)
    m = re.search(r"arxiv\.org/(?:abs|pdf|html)/(.+?)(?:\.pdf)?/?(?:[?#].*)?$", s, re.I)
    if m:
        s = m.group(1)
    m = ID_RE.fullmatch(s) or ID_RE.search(s)
    if not m:
        raise ValueError(f"not an arXiv id or url: {s!r}")
    return m.group(1)


def find_arxiv_id(text: str) -> str | None:
    """Find an arXiv id inside free text such as a bibliography entry."""
    m = re.search(rf"(?:arxiv[:\s/]*|arxiv\.org/(?:abs|pdf)/)({NEW_ID})", text, re.I)
    if m:
        return m.group(1)
    m = re.search(rf"(?:arxiv[:\s/]*)({OLD_ID})", text, re.I)
    return m.group(1) if m else None


class ArxivClient:
    API = "https://export.arxiv.org/api/query"

    def __init__(self, http: Http):
        self.http = http

    def metadata(self, ids: list[str]) -> dict[str, dict]:
        """Batch metadata: {id: {title, abstract, published (v1 ISO date), authors}}."""
        out: dict[str, dict] = {}
        ids = [i for i in dict.fromkeys(ids) if i]
        for start in range(0, len(ids), 50):
            chunk = ids[start:start + 50]
            try:
                xml = self.http.get(
                    self.API,
                    params={"id_list": ",".join(chunk), "max_results": len(chunk)},
                    as_json=False,
                )
            except HttpError as e:
                log.warning("arXiv metadata failed: %s", e)
                continue
            root = ET.fromstring(xml)
            for entry in root.findall("a:entry", ATOM):
                eid = entry.findtext("a:id", default="", namespaces=ATOM)
                m = ID_RE.search(eid.split("/abs/")[-1])
                if not m:
                    continue
                title = " ".join((entry.findtext("a:title", default="", namespaces=ATOM)).split())
                if not title or title.lower() == "error":
                    continue
                out[m.group(1)] = {
                    "title": title,
                    "abstract": " ".join(entry.findtext("a:summary", default="", namespaces=ATOM).split()),
                    "published": entry.findtext("a:published", default="", namespaces=ATOM)[:10],
                    "authors": [a.findtext("a:name", default="", namespaces=ATOM)
                                for a in entry.findall("a:author", ATOM)],
                }
        return out

    def html(self, arxiv_id: str) -> str | None:
        """LaTeXML HTML from arxiv.org/html, falling back to ar5iv. None if unavailable."""
        for url in (f"https://arxiv.org/html/{arxiv_id}", f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"):
            try:
                text = self.http.get(url, as_json=False)
            except HttpError as e:
                log.debug("no html at %s: %s", url, e)
                continue
            except Exception as e:  # proxy refusals, TLS errors, etc.
                log.debug("html fetch failed at %s: %s", url, e)
                continue
            if "ltx_bibitem" in text and "ltx_document" in text:
                return text
        return None

    def pdf_text(self, arxiv_id: str) -> str | None:
        """Plain text of the PDF via `pdftotext` (poppler). None if unavailable."""
        if not shutil.which("pdftotext"):
            log.warning("pdftotext not installed; cannot use PDF fallback for %s", arxiv_id)
            return None
        cached = self.http.cache.get("pdftext", arxiv_id)
        if cached is not None:
            return cached
        try:
            r = self.http.client.get(f"https://arxiv.org/pdf/{arxiv_id}")
            r.raise_for_status()
        except Exception as e:
            log.warning("PDF download failed for %s: %s", arxiv_id, e)
            return None
        with tempfile.TemporaryDirectory() as d:
            pdf = Path(d) / "p.pdf"
            pdf.write_bytes(r.content)
            res = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True)
            text = res.stdout
        if len(text) < 2000:
            return None
        self.http.cache.set("pdftext", arxiv_id, text)
        return text
