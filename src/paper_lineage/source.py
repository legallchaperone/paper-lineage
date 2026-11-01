"""Paper source: fetch + parse + S2 references, all cached."""

from __future__ import annotations

import logging

from .arxiv import ArxivClient
from .http import DiskCache, Http
from .models import PaperMeta, ParsedPaper
from .parse import parse_latexml, parse_plaintext
from .resolve import meta_from_s2
from .s2 import S2Client

log = logging.getLogger(__name__)


class PaperSource:
    def __init__(self, cache_dir: str | None):
        self.cache = DiskCache(cache_dir)
        self.http = Http(self.cache, min_interval={
            "api.semanticscholar.org": 1.05,
            "export.arxiv.org": 3.0,
            "arxiv.org": 0.5,
        })
        self.arxiv = ArxivClient(self.http)
        self.s2 = S2Client(self.http)
        self._parsed: dict[str, ParsedPaper | None] = {}

    def meta_for_arxiv(self, arxiv_id: str) -> PaperMeta:
        got = self.s2.batch([f"arXiv:{arxiv_id}"])
        p = got.get(f"arXiv:{arxiv_id}")
        ax = self.arxiv.metadata([arxiv_id]).get(arxiv_id, {})
        if p:
            m = meta_from_s2(p)
        else:
            m = PaperMeta(key=f"arXiv:{arxiv_id}", title=ax.get("title", arxiv_id), arxiv_id=arxiv_id)
        m.arxiv_id = arxiv_id
        m.key = f"arXiv:{arxiv_id}"
        if ax.get("published"):
            m.date = ax["published"]
            m.year = int(ax["published"][:4])
        m.abstract = m.abstract or ax.get("abstract")
        if ax.get("title"):
            m.title = ax["title"]
        return m

    def parsed(self, meta: PaperMeta) -> ParsedPaper | None:
        if not meta.arxiv_id:
            return None
        if meta.arxiv_id in self._parsed:
            return self._parsed[meta.arxiv_id]
        result: ParsedPaper | None = None
        try:
            html = self.arxiv.html(meta.arxiv_id)
            if html:
                result = parse_latexml(html, meta.arxiv_id)
            else:
                text = self.arxiv.pdf_text(meta.arxiv_id)
                if text:
                    result = parse_plaintext(text, meta.arxiv_id, title=meta.title, abstract=meta.abstract or "")
                    log.info("using PDF fallback for %s (lower confidence)", meta.arxiv_id)
        except Exception as e:  # one unparseable paper must not kill the whole run
            log.warning("could not parse %s: %s", meta.arxiv_id, e)
            result = None
        if result is not None and not result.title:
            result.title = meta.title
        if result is not None and len(result.citations) < 3:
            log.info("too few citations parsed for %s; treating as no full text", meta.arxiv_id)
            result = None
        self._parsed[meta.arxiv_id] = result
        return result

    def references(self, meta: PaperMeta) -> list[dict]:
        pid = f"arXiv:{meta.arxiv_id}" if meta.arxiv_id else meta.s2_id
        if not pid:
            return []
        try:
            return self.s2.references(pid)
        except Exception as e:
            log.warning("S2 references failed for %s: %s", pid, e)
            return []
