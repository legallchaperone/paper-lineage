import json

from paper_lineage.graph import LineageGraph
from paper_lineage.models import Edge, Node, PaperMeta
from paper_lineage.render import render_html, render_json, render_mermaid, render_tree


def small_graph():
    g = LineageGraph(target="arXiv:2")
    g.add_node(Node(meta=PaperMeta(key="arXiv:2", title="Geometry Forcing: X", year=2025, arxiv_id="2"),
                    kind="target"))
    g.add_node(Node(meta=PaperMeta(key="arXiv:1", title="Representation Alignment for Generation: Y", year=2024,
                                   alias="REPA", arxiv_id="1"), kind="origin", relevance=0.91))
    g.add_node(Node(meta=PaperMeta(key="arXiv:3", title='VideoREPA: "quoted" <b>', year=2025), kind="alternative"))
    g.add_edge(Edge(parent="arXiv:1", child="arXiv:2", role="technique_source", directness=3,
                    inherited="feature alignment", evidence="Inspired by REPA ( Yu et al., 2024a ), we align."))
    g.add_edge(Edge(parent="arXiv:2", child="arXiv:3", role="alternative", directness=2, inherited="",
                    evidence="Unlike VideoREPA, we use geometry."))
    g.compute_lines()
    g.choose_spine("auto")
    g.stats = {"expanded": 1, "classified_candidates": 3, "classifier": "test"}
    return g


def test_tree_mentions_chain_and_alternative():
    out = render_tree(small_graph(), show_evidence=True)
    assert "REPA (2024)" in out and "Geometry Forcing" in out
    assert "Done differently" in out and "VideoREPA" in out and "Inspired by REPA" in out


def test_mermaid_is_well_formed():
    mm = render_mermaid(small_graph())
    assert mm.startswith("flowchart LR")
    assert "==>" in mm            # spine edge
    assert "-.->|alt|" in mm
    assert '"quoted"' not in mm and "<b>" not in mm


def test_json_roundtrip():
    d = json.loads(render_json(small_graph()))
    assert d["schema_version"] == 1 and d["spine"] == ["arXiv:1", "arXiv:2"]
    assert {n["kind"] for n in d["nodes"]} == {"target", "origin", "alternative"}


def test_html_escapes_and_contains_evidence():
    h = render_html(small_graph())
    assert "Inspired by REPA" in h
    g = small_graph()
    g.nodes["arXiv:3"].meta.title = 'Evil </script><script>alert(1)</script>'
    h2 = render_html(g)
    assert "</script><script>alert(1)" not in h2           # titles cannot break out of the data block
    assert "Evil &lt;/script&gt;" in h2                      # and are HTML-escaped when shown


def test_html_is_self_contained_sketch_page():
    h = render_html(wc_graph())
    assert "font/woff2;base64," in h and "var rough=" in h      # Virgil + rough.js embedded
    assert "<script src" not in h                                # no external scripts: works offline
    assert "http://" not in h.split("<script")[0].split("<style>")[0]
    for sid in ("intro", "line-0", "line-1", "line-2", "differently", "rest", "about"):
        assert f'<section id="{sid}"' in h
    g = wc_graph()
    assert h.count('class="step tl"') == sum(len(line.best_path) - 1 for line in g.lines) == 13
    assert 'class="recipe"' in h and "is built on" in h          # the answer comes first
    assert h.count('<details class="ev">') >= 14                # quotes folded, one per claim
    assert "Design inspired by" in h and "kuant.space/pi" in h


def test_alternative_quotes_are_attributed_to_the_contrasting_paper():
    import re
    h = render_html(wc_graph())
    # Self-Forcing is an alternative that Helios contrasts itself with: the sentence is Helios's
    li = re.search(r'<li class="alt"[^>]*>(?:(?!</li>).)*Self Forcing: Bridging(?:(?!</li>).)*</li>', h,
                   re.S).group(0)
    assert "— Helios" in li and "— Self-Forcing" not in li and "— Self Forcing" not in li
    assert "building on top of the VGGT model" in h             # evidence quoted


def test_walkthrough_introduces_full_titles_then_short_names():
    import re
    g = wc_graph()
    h = render_html(g)
    article = h.split("<article>")[1].split("</article>")[0]
    text = re.sub(r"<[^>]+>", "", article)
    # first mention: full title with the short name in parentheses; afterwards the short name alone
    full = "Cameras as Relative Positional Encoding (PRoPE"
    assert text.count(full) == 1 and text.index(full) < text.index("2025 PRoPE")
    # a name that already leads its title is not repeated: 'VGGT: Visual Geometry Grounded Transformer'
    assert "VGGT: Visual Geometry Grounded Transformer" in text and "Transformer (VGGT)" not in text
    # titles without a method name are never shown ellipsis-cut in the walkthrough
    assert "Efficient Geometry-aware 3D Generative Adversarial Networks (EGA3DG)" in text
    assert "…" not in re.sub(r"<[^>]+>", "", re.sub(r"<details.*?</details>", "", article, flags=re.S))


def test_acronym_for_titles_without_a_name():
    from paper_lineage.quality import acronym
    assert acronym("Possible Principles Underlying the Transformations of Sensory Messages") == "PPUTSM"
    assert acronym("Proving and Improving the Platonic Representation Hypothesis") == "PIPRH"


def test_formulas_are_typeset_not_raw_but_evidence_stays_verbatim():
    import re
    g = wc_graph()
    h = render_html(g)
    visible = re.sub(r'<script.*?</script>|<style.*?</style>|data-tex="[^"]*"', "", h, flags=re.S)
    assert not re.search(r"\\(?:math\w*|hat|text|left|right)\{", visible)    # no raw LaTeX shown
    assert h.count('class="tex"') >= 15 and "temml" in h
    assert any("\\mathcal" in e.evidence for e in g.edges)               # stored evidence untouched


# --- real run (WorldCrafter, arXiv:2609.24984, LLM mode): 80 papers, 3 lines --------------------------

def wc_graph():
    from conftest import FIX
    return LineageGraph.from_dict(json.loads((FIX / "worldcrafter_2609.24984.graph.json").read_text()))


def test_from_dict_roundtrip():
    g = wc_graph()
    assert len(g.nodes) == 80 and len(g.lines) == 3
    g2 = LineageGraph.from_dict(json.loads(render_json(g)))
    assert [line.best_path for line in g2.lines] == [line.best_path for line in g.lines]


def test_html_payload_is_complete_and_valid_json():
    import re
    from paper_lineage.render.html import graph_payload
    g = wc_graph()
    h = render_html(g)
    data = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', h, re.S).group(1))
    assert data == json.loads(json.dumps(graph_payload(g)))
    assert len(data["nodes"]) == 80 and len(data["lines"]) == 3      # everything reachable via the panel
    assert all(any(e["p"] == a and e["c"] == b for e in data["edges"])
               for line in data["lines"] for a, b in zip(line["path"], line["path"][1:]))


def test_compact_views_draw_only_main_chains():
    from paper_lineage.render.mermaid import visible_keys
    g = wc_graph()
    keep = visible_keys(g)
    assert len(keep) <= 25 < len(g.nodes)
    assert len(visible_keys(g, full=True)) == 80
    mm = render_mermaid(g)
    assert mm.count(":::") == len(keep)
    out = render_tree(g)
    assert "more papers in this line" in out and "also in this line" not in out


def test_output_is_deterministic_across_processes(tmp_path):
    import hashlib
    import os
    import subprocess
    import sys
    from conftest import FIX
    digests = set()
    for seed in ("1", "2"):
        out = tmp_path / f"r{seed}"
        env = {**os.environ, "PYTHONHASHSEED": seed}
        subprocess.run([sys.executable, "-m", "paper_lineage.cli", "--render",
                        str(FIX / "worldcrafter_2609.24984.graph.json"), "-f", "html", "-f", "json", "-o", str(out)],
                       check=True, env=env, capture_output=True)
        digests.add(hashlib.md5((out.with_suffix(".html").read_bytes() + out.with_suffix(".json").read_bytes())).hexdigest())
    assert len(digests) == 1
