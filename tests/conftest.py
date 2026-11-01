from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from paper_lineage.models import Candidate, PaperMeta
from paper_lineage.parse import parse_latexml
from paper_lineage.resolve import resolve_bib
from paper_lineage.heuristics import score_candidate

FIX = Path(__file__).parent / "fixtures"


def read_gz(name: str) -> str:
    return gzip.decompress((FIX / name).read_bytes()).decode("utf-8")


@pytest.fixture(scope="session")
def gf_parsed():
    return parse_latexml(read_gz("gf_2507.07982.html.gz"), "2507.07982")


@pytest.fixture(scope="session")
def df_parsed():
    return parse_latexml(read_gz("df_2407.01392.html.gz"), "2407.01392")


@pytest.fixture(scope="session")
def gf_refs():
    return json.loads((FIX / "gf_2507.07982.s2refs.json").read_text())


@pytest.fixture(scope="session")
def gf_meta():
    return PaperMeta(key="arXiv:2507.07982", arxiv_id="2507.07982", year=2025, date="2025-07-10",
                     title="Geometry Forcing: Marrying Video Diffusion and 3D Representation for "
                           "Consistent World Modeling",
                     abstract="Videos inherently represent 2D projections of a dynamic 3D world.")


@pytest.fixture()
def gf_candidates(gf_parsed, gf_refs):
    metas = resolve_bib(gf_parsed.bib, gf_refs, gf_parsed.title)
    cands = []
    for k, b in gf_parsed.bib.items():
        ctx = gf_parsed.contexts_for(k)
        if not ctx:
            continue
        c = Candidate(bib_key=k, bib=b, meta=metas.get(k), contexts=ctx)
        score_candidate(c, gf_parsed.full_text)
        cands.append(c)
    cands.sort(key=lambda c: -c.score)
    return cands


def find(cands, needle: str):
    for c in cands:
        if needle.lower() in c.title.lower():
            return c
    raise AssertionError(f"no candidate matching {needle!r}")
