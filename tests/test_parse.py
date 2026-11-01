from paper_lineage.arxiv import find_arxiv_id, normalize_arxiv_id
from paper_lineage.parse import parse_plaintext, split_sentences

from conftest import read_gz


def test_normalize_arxiv_id():
    assert normalize_arxiv_id("http://arxiv.org/abs/2507.07982") == "2507.07982"
    assert normalize_arxiv_id("https://arxiv.org/pdf/2507.07982v2.pdf") == "2507.07982"
    assert normalize_arxiv_id("https://arxiv.org/html/2507.07982v1#S2") == "2507.07982"
    assert normalize_arxiv_id("arXiv:2410.06940") == "2410.06940"
    assert normalize_arxiv_id("2410.06940v3") == "2410.06940"
    assert normalize_arxiv_id("hep-th/9901001") == "hep-th/9901001"


def test_find_arxiv_id_in_bib_text():
    assert find_arxiv_id("Representation alignment ... arXiv preprint arXiv:2410.06940, 2024.") == "2410.06940"
    assert find_arxiv_id("see https://arxiv.org/abs/2006.11239") == "2006.11239"
    assert find_arxiv_id("NeurIPS 2020") is None


def test_split_sentences_respects_citations_and_abbreviations():
    text = "Zhang et al. (2025a) proposes point maps. We follow e.g. prior work. Next one."
    cite = (0, len("Zhang et al. (2025a)"))
    spans = split_sentences(text, [cite])
    sents = [text[a:b].strip() for a, b in spans]
    assert sents == ["Zhang et al. (2025a) proposes point maps.", "We follow e.g. prior work.", "Next one."]


def test_gf_html_every_bib_entry_is_linked(gf_parsed):
    assert gf_parsed.title.startswith("Geometry Forcing")
    assert len(gf_parsed.bib) == 97
    cited = {k for c in gf_parsed.citations for k in c.keys}
    assert cited == set(gf_parsed.bib)


def test_gf_vggt_contexts_have_sections(gf_parsed):
    vggt = gf_parsed.bib["bib.bib79"]
    assert vggt.title.lower().startswith("vggt")
    kinds = {c.section_kind for c in gf_parsed.contexts_for("bib.bib79")}
    assert {"intro", "prelim", "method"} <= kinds


def test_gf_unlinked_author_year_citation_is_recovered(gf_parsed):
    # "(Zhou et al., 2018)" has no href in the HTML; must be mapped through the bib label
    key = next(k for k, b in gf_parsed.bib.items() if b.label.startswith("Zhou"))
    assert "stereo magnification" in gf_parsed.bib[key].title.lower()
    assert gf_parsed.contexts_for(key)


def test_gf_citation_spans_point_at_the_right_group(gf_parsed):
    c = next(c for c in gf_parsed.citations if c.sentence.startswith("Our approach builds upon"))
    groups = [ks for _, ks in sorted(c.spans)]
    assert len(groups) == 2
    assert "bib.bib79" in groups[1]          # "... incorporates a 3D foundation model (Wang et al., 2025b)"
    assert "bib.bib79" not in groups[0]
    off = sorted(c.spans)[0][0]
    assert c.sentence[off:].startswith("( Chen")


def test_df_numeric_and_malformed_html(df_parsed):
    # DF's HTML has an unclosed <figure> that nests the appendix inside a table; every entry must still be found
    assert len(df_parsed.bib) == 72
    cited = {k for c in df_parsed.citations for k in c.keys}
    assert cited == set(df_parsed.bib)
    tf = [c for c in df_parsed.contexts_for("bib.bib65") if c.section_kind == "intro"]
    assert tf and "teacher forcing" in tf[0].sentence


def test_plaintext_fallback_numeric():
    p = parse_plaintext(read_gz("df_2407.01392.pdftotext.txt.gz"), "2407.01392")
    assert len(p.bib) == 72
    tf = next(k for k, b in p.bib.items() if "continually running" in b.text.lower())
    assert any("teacher forcing" in c.sentence for c in p.contexts_for(tf))
    assert {"intro", "related"} <= {c.section_kind for c in p.citations}


def test_plaintext_fallback_author_year():
    p = parse_plaintext(read_gz("gf_2507.07982.pdftotext.txt.gz"), "2507.07982")
    assert len(p.bib) >= 85
    repa = next(k for k, b in p.bib.items() if "Representation alignment for generation" in b.text)
    assert any("inspired by" in c.sentence for c in p.contexts_for(repa))


def test_plaintext_many_same_author_year_entries():
    refs = "\n".join(f"[{i}] J. Smith. Paper number {i} about things. arXiv, 2020." for i in range(1, 31))
    text = "1\nIntroduction\n\nWe build on prior work [3] and [27].\n\nReferences\n" + refs
    p = parse_plaintext(text, "x")
    assert len(p.bib) == 30 and p.contexts_for("ref27")
