from paper_lineage.heuristics import alias_from_contexts, directness_from_score, heuristic_classification
from paper_lineage.models import LINEAGE_ROLES
from paper_lineage.resolve import resolve_bib

from conftest import find

TRUE_PARENTS = [
    "Representation Alignment for Generation",      # REPA: technique source
    "History-Guided Video Diffusion",                # DFoT: base model
    "VGGT",                                          # teacher model
    "Diffusion Forcing",                             # AR video diffusion paradigm
    "Next-Frame Diffusion",                          # base model (Minecraft)
]
MUST_DROP = [
    "Institutionum calculi integralis",              # Euler, 1845 (ODE solver)
    "Towards Accurate Generative Models of Video",   # FVD
    "Unreasonable Effectiveness of Deep Features",   # LPIPS
    "Image quality assessment",                      # SSIM
    "Two Time-Scale Update Rule",                    # FID
    "Stereo magnification",                          # RealEstate10K
    "Video PreTraining",                             # VPT / Minecraft data
    "Language Models are Few-Shot Learners",
    "Language Models are Unsupervised Multitask Learners",
    "Attention is All you Need",
]


def test_resolution_uses_title_not_author_year(gf_parsed, gf_refs):
    metas = resolve_bib(gf_parsed.bib, gf_refs, gf_parsed.title)
    assert len(metas) >= 90
    # S2's own contexts attach "Wang et al., 2025b" to the wrong papers; our resolution must not
    assert metas["bib.bib79"].title.startswith("VGGT")
    assert metas["bib.bib79"].arxiv_id == "2503.11651"


def test_true_parents_rank_on_top(gf_candidates):
    top = [c.title for c in gf_candidates[:6]]
    for t in TRUE_PARENTS:
        assert any(t.lower() in x.lower() for x in top), (t, top)


def test_noise_is_not_lineage(gf_candidates):
    for t in MUST_DROP:
        cls = heuristic_classification(find(gf_candidates, t))
        assert not (cls.role in LINEAGE_ROLES and cls.directness >= 2), (t, cls)


def test_true_parents_are_lineage_in_heuristic_mode(gf_candidates):
    for t in TRUE_PARENTS:
        cls = heuristic_classification(find(gf_candidates, t))
        assert cls.role in LINEAGE_ROLES and cls.directness >= 2, (t, cls)
    repa = heuristic_classification(find(gf_candidates, "Representation Alignment for Generation"))
    assert repa.role == "technique_source"
    assert "inspired by" in repa.evidence


def test_alias_extraction(gf_candidates):
    assert find(gf_candidates, "Representation Alignment for Generation").meta.alias == "REPA"
    assert find(gf_candidates, "History-Guided Video Diffusion").meta.alias == "DFoT"
    c = find(gf_candidates, "VGGT")
    assert alias_from_contexts(c.contexts, c.bib_key) == "VGGT"


def test_directness_thresholds():
    assert directness_from_score(20) == 3
    assert directness_from_score(0.5) == 0
