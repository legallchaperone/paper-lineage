import re

from paper_lineage.render.mathtext import plain, rich, split_math

from conftest import read_gz


def maths(text):
    return [c for m, c in split_math(text) if m]


def test_delimited_formulas():
    assert maths(r"we propose $\pi_0$, a model") == [r"\pi_0"]
    assert plain(r"$\pi_0$: A Vision-Language-Action Flow Model") == "π₀: A Vision-Language-Action Flow Model"


def test_bare_latex_from_older_runs():
    q = (r"The depth loss \mathcal{L}_{\text{depth}} follows DUSt3R [ 129 ] weighing \hat{D}_{i} and D_{i} "
         r"with \hat{\Sigma}_{i}^{D}.")
    assert maths(q) == [r"\mathcal{L}_{\text{depth}}", r"\hat{D}_{i}", "D_{i}", r"\hat{\Sigma}_{i}^{D}"]
    assert maths(r"on 256\times256 videos with 2 \times 2 kernel") == [r"256\times256", r"2 \times 2"]


def test_ordinary_text_is_not_math():
    for s in ["snake_case_names and file_name.py", "the 5^th time", "cost (e.g. 3-5x) [ 12 ]"]:
        assert maths(s) == [], s


def test_rich_never_shows_raw_latex():
    h = rich(r"loss \mathcal{L}_{\text{depth}} and $\hat{x}_{t}$ [ 12 ]")
    visible = re.sub(r'data-tex="[^"]*"', "", h)
    assert "\\mathcal" not in visible and "\\hat" not in visible and "$" not in visible
    assert h.count('class="tex"') == 2 and "[12]" in h


def test_parser_wraps_formulas_in_delimiters():
    from paper_lineage.parse import parse_latexml
    p = parse_latexml(read_gz("df_2407.01392.html.gz"), "2407.01392")
    sents = [c.sentence for c in p.citations if "$" in c.sentence]
    assert sents, "expected some citing sentences with formulas"
    for s in sents:
        assert s.count("$") % 2 == 0, s
        for tex in maths(s):
            assert "$" not in tex
