from paper_lineage.graph import LineageGraph
from paper_lineage.intros import add_intros, grounded, pick_sentence, sentences
from paper_lineage.llm import ScriptedLLM
from paper_lineage.models import Node, PaperMeta
from paper_lineage.quality import check, normalize

REPA_ABS = ("Recent studies have shown that the denoising process in diffusion models can induce meaningful "
            "representations, e.g. in DiTs (Peebles et al., 2023). We argue that one main bottleneck lies in "
            "learning these representations. We study this by introducing a straightforward regularization "
            "called REPresentation Alignment (REPA), which aligns hidden states with $x_{t}.y$ clean image "
            "representations. Our method speeds up SiT training by over 17.5 times. Code is available at "
            "github.com/x/y.")


def test_sentences_keep_abbreviations_and_formulas():
    ss = sentences(REPA_ABS)
    assert ss[0].endswith("(Peebles et al., 2023).") and "e.g. in DiTs" in ss[0]
    assert any("$x_{t}.y$" in s for s in ss)
    assert len(ss) == 5


def test_pick_prefers_the_sentence_where_the_paper_introduces_itself():
    s = pick_sentence(REPA_ABS, "Representation Alignment for Generation", alias="REPA")
    assert s.startswith("We study this by introducing")
    assert s in REPA_ABS                                                   # verbatim
    navit = ("Fixed resolutions are suboptimal. We take advantage of this with NaViT which packs sequences. "
             "We believe that NaViT marks a departure from the standard, CNN-designed pipeline.")
    assert pick_sentence(navit, "Patch n' Pack: NaViT, a Vision Transformer").startswith("We take advantage")
    assert pick_sentence("Too short.") == "" and pick_sentence(None) == ""
    assert pick_sentence("Images are everywhere in modern life today. Many models exist for them now.") \
        == "Images are everywhere in modern life today."                   # no self-description: first


def test_grounded_rejects_facts_not_in_the_abstract():
    assert grounded("REPA aligns diffusion hidden states with clean image representations from pretrained "
                    "encoders to speed up training.", REPA_ABS + " pretrained encoders")
    assert not grounded("REPA speeds up SiT training by 40 times.", REPA_ABS)          # invented number
    assert not grounded("REPA wins the robotics manipulation benchmark championship.", REPA_ABS)


def _graph():
    g = LineageGraph(target="arXiv:2")
    g.add_node(Node(meta=PaperMeta(key="arXiv:2", title="Target", year=2025, abstract="We propose a target "
                                   "method for video generation with geometry."), kind="target"))
    g.add_node(Node(meta=PaperMeta(key="arXiv:1", title="Representation Alignment for Generation", year=2024,
                                   alias="REPA", abstract=REPA_ABS), kind="origin"))
    g.add_node(Node(meta=PaperMeta(key="arXiv:0", title="No abstract here", year=2020), kind="origin"))
    return g


def test_add_intros_keeps_only_grounded_llm_summaries():
    g = _graph()
    llm = ScriptedLLM(default={"results": [
        {"id": "P1", "intro": "REPA aligns hidden states in diffusion training with clean image representations"},
        {"id": "P2", "intro": "A target method that wins 12 awards for robotics."},       # not grounded
    ]})
    add_intros(g, llm=llm)
    by = {k: n.meta for k, n in g.nodes.items()}
    assert by["arXiv:1"].intro == "REPA aligns hidden states in diffusion training with clean image representations."
    assert by["arXiv:2"].intro is None and by["arXiv:2"].intro_quote.startswith("We propose a target")
    assert by["arXiv:0"].intro is None and by["arXiv:0"].intro_quote is None           # nothing guessed
    assert "Abstract: " + " ".join(REPA_ABS.split())[:50] in llm.prompts[0]
    assert g.stats["intros"] == {"with_abstract": 2, "llm": 1}
    assert check(g) == []


def test_saved_runs_get_quotes_and_check_catches_non_verbatim_ones():
    g = normalize(_graph())                         # no LLM, no network: the quote is filled on load
    assert g.nodes["arXiv:1"].meta.intro_quote.startswith("We study this by introducing")
    g.nodes["arXiv:1"].meta.intro_quote = "REPA is the best method ever."
    g.nodes["arXiv:1"].meta.abstract = "Something else entirely."
    assert any("intro not verbatim" in i for i in check(g))
