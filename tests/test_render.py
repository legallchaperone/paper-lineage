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
    assert "esc(n.title)" in h2                              # and are HTML-escaped when shown
    assert "<script src" not in h                         # self-contained, works offline


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
