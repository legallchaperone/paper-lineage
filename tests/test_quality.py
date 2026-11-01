"""Output quality as invariants: every run, whatever the paper, must satisfy these."""

from __future__ import annotations

import json
import re

import pytest

from paper_lineage.graph import LineageGraph
from paper_lineage.models import Edge, Node, PaperMeta
from paper_lineage.quality import check, clean_summary, normalize, short_title
from paper_lineage.render import render_html, render_mermaid, render_tree

from conftest import FIX

RAW_TEX = re.compile(r"\\[A-Za-z]+\s*[{_^]|\\(?:mathcal|mathbf|hat|frac|text)\b|\$[^$\s][^$]*\$")


# ---------------------------------------------------------------- unit rules

@pytest.mark.parametrize("raw,want", [
    ("Base video DiT initialization and coarse-to-fine pyramid denoising scheme.", "base video DiT initialization and coarse-to-fine pyramid denoising scheme"),
    ("A parallel camera-attention branch for camera conditioning", "a parallel camera-attention branch for camera conditioning"),
    ("VGGT backbone and pretrained weights", "VGGT backbone and pretrained weights"),
    ("It builds on their per-frame noise schedule.", "per-frame noise schedule"),
    ("  “Rectified flow formulation”  ", "rectified flow formulation"),
    ("", ""),
])
def test_clean_summary(raw, want):
    assert clean_summary(raw) == want
    assert clean_summary(clean_summary(raw)) == clean_summary(raw)       # idempotent


def test_clean_summary_bounds_length_at_word_boundary():
    s = clean_summary(" ".join(f"word{i}" for i in range(40)))
    assert len(s.split()) <= 18 and s.endswith("…") and "word17…" in s


@pytest.mark.parametrize("title,want", [
    ("VGGT: Visual Geometry Grounded Transformer", "VGGT"),
    ("Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", "Scaling Rectified Flow Transformers"),
    ("Denoising Diffusion Autoencoders are Unified Self-supervised Learners", "Denoising Diffusion Autoencoders"),
    ("Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation", "Language Model Beats Diffusion"),
    ("Diffusion Models Beat GANs on Image Synthesis", "Diffusion Models Beat GANs on Image Synthesis"),
])
def test_short_title(title, want):
    assert short_title(title) == want


def test_short_title_never_cuts_mid_word():
    for t in ["Efficient Geometry-aware Three-dimensional Generative Adversarial Networks Revisited Again",
              "Supercalifragilisticexpialidociousnessness Hyperparameterizationalities Everywhere"]:
        s = short_title(t)
        assert len(s) <= 49
        assert s.rstrip("…") in t and (s.endswith("…") is False or t[len(s) - 1] == " " or s.rstrip("…") == t[:len(s.rstrip("…"))])
        assert not s.endswith("...")


# ---------------------------------------------------------------- graphs to test on

def wc():
    return LineageGraph.from_dict(json.loads((FIX / "worldcrafter_2609.24984.graph.json").read_text()))


def gf_heuristic():
    return LineageGraph.from_dict(json.loads((FIX / "gf_2507.07982.no-llm.graph.json").read_text()))


def nasty():
    """Everything that has gone wrong (or could): LaTeX titles, very long titles, duplicate names,
    capitalised / echoing summaries with periods, LaTeX in quotes, alternatives at every level."""
    g = LineageGraph(target="T")
    def add(k, title, year, kind="ancestor", rel=0.93):
        g.add_node(Node(meta=PaperMeta(key=k, title=title, year=year, arxiv_id=f"2{year % 100:03d}.0000{len(g.nodes)}"),
                        kind=kind, relevance=rel))
    add("T", r"$\pi_0$-World: Flow Matching World Models with $\mathcal{O}(1)$ Memory", 2026, "target", 1.0)
    add("A", "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", 2024)
    add("B", "Efficient Geometry-aware Three-dimensional Generative Adversarial Networks Revisited Again", 2021, "origin")
    add("C", "Patchification: A Study", 2023)
    add("D", "Patchification: Another Study", 2025)
    add("E", r"$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control", 2024, "origin")
    add("X", "Some Alternative Approach With A Long Name That Goes On", 2025, "alternative")
    add("Y", "Another Alternative", 2026, "alternative")
    def E(p, c, role, d, inh, ev, sec="method"):
        g.add_edge(Edge(parent=p, child=c, role=role, directness=d, inherited=inh, evidence=ev, section_kind=sec))
    E("B", "A", "technique_source", 3, "It builds on their triplane representation.", r"We adopt the triplane \mathcal{T} of [ 12 ].")
    E("A", "D", "builds_on", 3, "A rectified flow objective.", r"Following [ 3 ], x_{t} = (1-t) x_{0} + t \epsilon .")
    E("C", "D", "component", 2, "Patch embedding", "We use the patch embedding of ( Doe et al., 2023 ).")
    E("D", "T", "builds_on", 3, "The base model.", "We initialise from Patchification [ 4 ].")
    E("E", "T", "technique_source", 2, "Flow-matching action expert", r"Like $\pi_0$ [ 7 ], we train a flow-matching expert.")
    E("D", "X", "alternative", 2, "", "Unlike [ 9 ], we avoid explicit 3D.")
    E("T", "Y", "alternative", 1, "", "In contrast to concurrent work [ 11 ], we keep memory implicit.")
    g.stats = {"expanded": 4, "classified_candidates": 40, "classifier": "llm"}
    normalize(g)
    g.compute_lines()
    g.choose_spine("auto")
    return g


GRAPHS = {"worldcrafter-llm": wc, "geometry-forcing-heuristic": gf_heuristic, "synthetic-nasty": nasty}


def visible(html_text: str) -> str:
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", html_text, flags=re.S)
    t = re.sub(r'<span class="tex"[^>]*>.*?</span>', "⟨math⟩", t, flags=re.S)   # checked separately
    t = re.sub(r'(data-tex|title|aria-label)="[^"]*"', "", t)
    block = r"p|div|li|ul|ol|h[1-6]|section|blockquote|cite|details|summary|br|nav|footer|header|aside|main|article|title"
    t = re.sub(rf"</?(?:{block})\b[^>]*>", " ", t)        # block tags separate text, like a browser
    t = re.sub(r"<[^>]+>", "", t)                           # inline tags do not
    import html as h
    return " ".join(h.unescape(t).split())


# ---------------------------------------------------------------- invariants on every graph

@pytest.mark.parametrize("name", list(GRAPHS))
def test_invariants(name):
    g = GRAPHS[name]()
    assert check(g) == [], check(g)

    page = render_html(g)
    text = visible(page)
    assert not RAW_TEX.search(text), RAW_TEX.search(text).group(0)          # formulas typeset, never raw
    assert "..." not in text                                                # no chopped names
    assert not re.search(r"\.\s*\.(?!\.)|:\s*\.|\s[,.;:](?=\s)|\(\s|\s\)", text), \
        re.search(r".{30}(?:\.\s*\.(?!\.)|:\s*\.|\s[,.;:](?=\s)|\(\s|\s\)).{30}", text)   # sentence hygiene
    assert re.search(r'class="recipe"', page) or not g.parents(g.target)

    # one timeline step per paper on each main chain; every step after the first states a relation + quote
    for i, line in enumerate(g.lines):
        sec = re.search(rf'<section id="line-{i}".*?</section>', page, re.S).group(0)
        steps = re.findall(r'<li class="step tl".*?</li>', sec, re.S)
        assert len(steps) == len(line.best_path) - 1
        for j, st in enumerate(steps[1:], 1):
            e = next(x for x in g.edges if x.parent == line.best_path[j - 1] and x.child == line.best_path[j])
            assert re.search(r'class="tl-body">(built on|idea from|uses) ', st), st[:300]
            if e.evidence:
                assert "<blockquote>" in st

    # every quote is attributed to the paper that wrote it (child of a lineage edge, parent of an alternative)
    for li in re.findall(r'<li class="alt"[^>]*data-node="([^"]+)"[^>]*>(.*?)</li>', page, re.S):
        alt, body = li
        for e in [x for x in g.edges if x.child == alt and x.role == "alternative" and x.evidence]:
            if "<cite>" in body:
                from paper_lineage.render.mathtext import rich
                assert "<cite>— " + rich(g.nodes[e.parent].meta.short_name) in body

    for out in (render_tree(g, show_evidence=True, full=True), render_mermaid(g, full=True)):
        assert not RAW_TEX.search(out) and "..." not in out

    assert render_html(g) == page                                           # deterministic


def test_check_catches_violations():
    g = nasty()
    g.edges[0].inherited = "Capitalised summary."
    g.edges[1].evidence = ""
    g.nodes["A"].meta.alias = g.nodes["C"].meta.short_name
    issues = check(g)
    assert any("not normalized" in i for i in issues)
    assert any("without evidence" in i for i in issues)
    assert any("duplicate name" in i for i in issues)


@pytest.mark.parametrize("name", list(GRAPHS))
def test_formula_fallbacks_are_readable(name):
    """If typesetting ever fails, the Unicode fallback shown instead must still be clean."""
    import html as h
    page = render_html(GRAPHS[name]())
    for tex, fb in re.findall(r'<span class="tex" data-tex="([^"]*)">(.*?)</span>', page, re.S):
        fb = h.unescape(fb)
        assert "\\" not in fb and "{" not in fb and "..." not in fb, (h.unescape(tex), fb)
        assert not re.search(r"(?<![A-Za-z])(?:in|mathbb|mathcal|times|cdot)[A-Z]", fb), (h.unescape(tex), fb)


@pytest.mark.parametrize("name", list(GRAPHS))
def test_canvas_names_are_clean(name):
    from paper_lineage.render.html import graph_payload
    for n in graph_payload(GRAPHS[name]())["nodes"].values():
        assert "..." not in n["name"] and "\\" not in n["name"] and "$" not in n["name"], n["name"]
