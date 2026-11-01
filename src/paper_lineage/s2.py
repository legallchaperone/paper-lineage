"""Semantic Scholar Graph API client (ID resolution, metadata, embeddings, citations).

Deliberately NOT used for deciding lineage: S2's per-reference `contexts`/`isInfluential`
mis-attribute author-year citations (e.g. in arXiv:2507.07982 the contexts of VGGT
"Wang et al., 2025b" are attached to unrelated "Wang et al." papers).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .http import Http, HttpError

log = logging.getLogger(__name__)

BASE = "https://api.semanticscholar.org/graph/v1"
META_FIELDS = "title,year,publicationDate,externalIds,citationCount,abstract"


class S2Client:
    def __init__(self, http: Http):
        self.http = http
        key = os.environ.get("S2_API_KEY")
        self.headers = {"x-api-key": key} if key else None

    def _get(self, path: str, **params: Any) -> Any:
        return self.http.get(BASE + path, params=params or None, headers=self.headers)

    def paper(self, pid: str, fields: str = META_FIELDS) -> dict | None:
        try:
            return self._get(f"/paper/{pid}", fields=fields)
        except HttpError as e:
            if e.status == 404:
                return None
            raise

    def references(self, pid: str) -> list[dict]:
        """List of cited papers (dicts with paperId,title,year,externalIds,citationCount); may include None ids."""
        out: list[dict] = []
        offset = 0
        while True:
            try:
                d = self._get(f"/paper/{pid}/references",
                              fields="title,year,externalIds,citationCount,publicationDate",
                              limit=1000, offset=offset)
            except HttpError as e:
                if e.status == 404:
                    return out
                raise
            for r in d.get("data") or []:
                cp = r.get("citedPaper") or {}
                if cp.get("title"):
                    out.append(cp)
            nxt = d.get("next")
            if not nxt or not d.get("data"):
                return out
            offset = nxt

    def citations(self, pid: str, limit: int = 500,
                  fields: str = "title,year,externalIds,citationCount,publicationDate,abstract,contexts,intents",
                  ) -> list[dict]:
        """Papers citing pid: [{citingPaper:{...}, contexts:[...], intents:[...]}]."""
        out: list[dict] = []
        offset = 0
        while len(out) < limit:
            try:
                d = self._get(f"/paper/{pid}/citations", fields=fields,
                              limit=min(1000, limit - len(out)), offset=offset)
            except HttpError as e:
                if e.status == 404:
                    return out
                raise
            out.extend(d.get("data") or [])
            nxt = d.get("next")
            if not nxt or not d.get("data"):
                break
            offset = nxt
        return out

    def batch(self, ids: list[str], fields: str = META_FIELDS + ",embedding.specter_v2") -> dict[str, dict]:
        """{requested_id: paper} for ids like 'arXiv:2410.06940' or S2 paperIds."""
        out: dict[str, dict] = {}
        ids = list(dict.fromkeys(i for i in ids if i))
        for start in range(0, len(ids), 100):
            chunk = ids[start:start + 100]
            try:
                res = self.http.post(BASE + "/paper/batch", params={"fields": fields},
                                     json_body={"ids": chunk}, headers=self.headers)
            except HttpError as e:
                log.warning("S2 batch failed: %s", e)
                continue
            for i, p in zip(chunk, res):
                if p:
                    out[i] = p
        return out

    def match_title(self, title: str) -> dict | None:
        try:
            d = self._get("/paper/search/match", query=title[:300],
                          fields="title,year,externalIds,citationCount,publicationDate")
        except HttpError as e:
            if e.status == 404:
                return None
            raise
        data = d.get("data") or []
        return data[0] if data else None
