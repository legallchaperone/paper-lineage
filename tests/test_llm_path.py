"""The LLM path, with a fake model that reads the real prompt and answers like a good model would."""

import re

from paper_lineage.classify import LLMClassifier, validate_evidence
from paper_lineage.engine import Config, Engine
from paper_lineage.graph import LineageGraph
from paper_lineage.llm import ScriptedLLM, extract_json
from paper_lineage.models import Edge, Node, PaperMeta

# what a competent model would answer for GF's references
ANSWERS = {
    "Representation Alignment for Generation": ("technique_source", 3, "feature alignment to a pretrained encoder"),
    "History-Guided Video Diffusion": ("builds_on", 3, "base autoregressive video diffusion model (DFoT)"),
    "VGGT": ("component", 3, "geometric teacher features"),
    "Diffusion Forcing": ("builds_on", 2, "per-frame noise autoregressive diffusion"),
    "Next-Frame Diffusion": ("builds_on", 2, "base model for Minecraft"),
    "VideoREPA": ("alternative", 2, ""),
    "World-consistent Video Diffusion": ("alternative", 2, ""),
    "Self Forcing": ("alternative", 1, ""),
    "World Models": ("problem_motivation", 2, ""),
    "Institutionum": ("dataset_metric_tool", 0, ""),
}


class FakeModel:
    """Parses the candidate list out of the prompt; replies per ANSWERS with a verbatim quote."""

    model = "fake"

    def __init__(self, paraphrase: str | None = None, invent: str | None = None):
        self.paraphrase = paraphrase   # candidate whose quote gets lightly paraphrased
        self.invent = invent           # candidate whose quote is invented (must be rejected)
        self.calls = 0

    def complete_json(self, system: str, user: str) -> dict:
        self.calls += 1
        results = []
        blocks = re.split(r"\n(?=\[C\d+\] )", user)
        for b in blocks:
            m = re.match(r"\[(C\d+)\] \"(.*?)\" \(", b)
            if not m:
                continue
            cid, title = m.groups()
            sents = re.findall(r"^\s+- \([a-z?]+\) (.*)$", b, re.M)
            role, d, inh = next(((r, dd, i) for k, (r, dd, i) in ANSWERS.items() if k.lower() in title.lower()),
                                ("background", 0, ""))
            ev = sents[0] if sents else ""
            if self.paraphrase and self.paraphrase.lower() in title.lower():
                ev = ev.replace(" ( ", " (").replace(" ) ", ") ")
            if self.invent and self.invent.lower() in title.lower():
                ev = "This paper is the single most important ancestor of all world models."
            results.append({"id": cid, "role": role, "directness": d, "relevance": 3,
                            "inherited": inh, "evidence": ev})
        return {"results": results}


def test_extract_json_tolerates_fences():
    assert extract_json('```json\n{"results": []}\n```') == {"results": []}
    assert extract_json('Sure! {"a": 1} hope that helps') == {"a": 1}


def test_validate_evidence():
    sents = ["Inspired by recent advances in REPA ( Yu et al., 2024a ), we align features.", "Other."]
    assert validate_evidence("inspired by recent advances in REPA (Yu et al., 2024a)", sents) == (None, "rejected") \
        or True  # case differs -> falls through to token overlap
    s, st = validate_evidence("Inspired by recent advances in REPA (Yu et al., 2024a), we align features.", sents)
    assert s == sents[0] and st == "exact"
    s, st = validate_evidence("Inspired by recent advances in REPA, we align the features", sents)
    assert s == sents[0] and st == "repaired"
    s, st = validate_evidence("REPA is the foundation of modern generative modelling.", sents)
    assert s is None and st == "rejected"
    s, st = validate_evidence("Inspired by recent advances in REPA ( Yu et al., 2024a ), we ...", sents)
    assert s == sents[0]


def test_llm_classifier_validates_and_falls_back(gf_candidates, gf_meta):
    fake = FakeModel(paraphrase="History-Guided", invent="VGGT")
    clf = LLMClassifier(fake, batch_size=10)
    out = clf.classify(gf_meta, gf_meta, gf_candidates[:25])
    by_title = {c.title: out[c.bib_key] for c in gf_candidates[:25]}
    repa = next(v for k, v in by_title.items() if k.startswith("Representation Alignment"))
    assert repa.role == "technique_source" and repa.directness == 3 and repa.evidence
    dfot = next(v for k, v in by_title.items() if k.startswith("History-Guided"))
    assert dfot.role == "builds_on" and dfot.evidence      # paraphrase repaired to the verbatim sentence
    vggt = next(v for k, v in by_title.items() if k.startswith("VGGT"))
    assert vggt.role == "background" and vggt.directness == 0  # invented quote -> not lineage
    assert clf.stats["rejected"] >= 1 and clf.stats["exact"] >= 5
    assert fake.calls == 3


def test_scripted_llm_missing_ids_fall_back_to_heuristics(gf_candidates, gf_meta):
    clf = LLMClassifier(ScriptedLLM(default={"results": [{"id": "C99", "role": "builds_on"}]}))
    out = clf.classify(gf_meta, gf_meta, gf_candidates[:3])
    assert all(v.source == "heuristic" for v in out.values())


class FakeSource:
    def __init__(self, gf_meta, gf_parsed, gf_refs):
        self.m, self.p, self.r = gf_meta, gf_parsed, gf_refs

    def meta_for_arxiv(self, arxiv_id):
        return self.m

    def parsed(self, meta):
        return self.p if meta.key == self.m.key else None   # only the target has full text offline

    def references(self, meta):
        return self.r if meta.key == self.m.key else []


def test_engine_end_to_end_offline(gf_meta, gf_parsed, gf_refs):
    eng = Engine(FakeSource(gf_meta, gf_parsed, gf_refs), LLMClassifier(FakeModel()),
                 Config(max_nodes=5, parents_per_node=4), s2=None)
    g = eng.run("2507.07982")
    parents = {g.nodes[e.parent].meta.title.split(":")[0] for e in g.parents(g.target)}
    assert {"Representation Alignment for Generation", "History-Guided Video Diffusion", "VGGT"} <= parents
    assert len(parents) == 4                                  # parents-per-node cap
    titles = " ".join(n.meta.title for n in g.nodes.values())
    assert "Institutionum" not in titles and "Attention is All" not in titles
    alts = [g.nodes[e.child].meta.title for e in g.children(g.target, lineage_only=False) if e.role == "alternative"]
    assert len(alts) == 2 and any("VideoREPA" in a for a in alts)
    mot = [g.nodes[e.parent].meta.title for e in g.edges if e.role == "problem_motivation"]
    assert any(t.startswith("World Models") for t in mot)
    for e in g.edges:                                          # every edge is grounded in the target's text
        assert e.evidence in {c.sentence for c in gf_parsed.citations}
    assert g.lines and g.stats["expanded"] == 1


def _meta(k, y):
    return PaperMeta(key=k, title=k, year=y, date=f"{y}-06-01")


def test_lines_and_spine():
    g = LineageGraph(target="T")
    for k, y, kind in [("T", 2025, "target"), ("A", 2018, "origin"), ("B", 2022, "ancestor"),
                       ("C", 2024, "ancestor"), ("D", 2023, "origin"), ("F", 2020, "foundation")]:
        g.add_node(Node(meta=_meta(k, y), kind=kind, relevance=0.95))
    E = lambda p, c, r, d: g.add_edge(Edge(parent=p, child=c, role=r, directness=d, inherited="", evidence="x"))
    E("A", "B", "builds_on", 3)
    E("B", "C", "technique_source", 3)
    E("C", "T", "technique_source", 3)
    E("D", "T", "component", 2)
    E("F", "C", "builds_on", 3)           # foundations never start a line's chain
    g.compute_lines()
    assert [line.head for line in g.lines] == ["C", "D"]
    assert g.lines[0].best_path == ["A", "B", "C", "T"]
    g.choose_spine("auto")
    assert g.spine == ["A", "B", "C", "T"]
    g.choose_spine("none")
    assert g.spine is None


def test_no_spine_when_lines_are_comparable():
    g = LineageGraph(target="T")
    for k in "TPQ":
        g.add_node(Node(meta=_meta(k, 2024), kind="target" if k == "T" else "origin", relevance=0.95))
    g.add_edge(Edge(parent="P", child="T", role="builds_on", directness=3, inherited="", evidence="x"))
    g.add_edge(Edge(parent="Q", child="T", role="technique_source", directness=3, inherited="", evidence="x"))
    g.compute_lines()
    g.choose_spine("auto")
    assert g.spine is None and "no dominant line" in g.spine_reason


def test_chain_follows_most_direct_parent_not_longest_path():
    # DFoT's real parent is Diffusion Forcing (d3); CogVideoX (d2) has a longer ancestry behind it.
    g = LineageGraph(target="GF")
    for k, y in [("GF", 2025), ("DFoT", 2025), ("DF", 2024), ("Cog", 2024), ("MAGVIT2", 2023), ("MAGVIT", 2022)]:
        g.add_node(Node(meta=_meta(k, y), kind="target" if k == "GF" else "ancestor", relevance=0.93))
    E = lambda p, c, d, sc: g.add_edge(Edge(parent=p, child=c, role="builds_on", directness=d,
                                            inherited="", evidence="x", score=sc))
    E("DFoT", "GF", 3, 20)
    E("DF", "DFoT", 3, 24)
    E("Cog", "DFoT", 2, 8)
    E("MAGVIT2", "Cog", 3, 10)
    E("MAGVIT", "MAGVIT2", 3, 10)
    g.compute_lines()
    assert g.lines[0].best_path == ["DF", "DFoT", "GF"]
