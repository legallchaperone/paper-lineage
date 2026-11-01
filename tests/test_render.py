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
    assert mm.startswith("flowchart TB")
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
    assert "&lt;b&gt;" in h and '<b>"' not in h          # the paper title's markup is escaped
    assert "mermaid@10.9.1" in h
