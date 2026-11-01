# AGENTS.md

## Using paper-lineage (the tool)

If the user wants the origins / lineage / prior-work map of an arXiv paper, follow
[`skills/paper-lineage/SKILL.md`](skills/paper-lineage/SKILL.md). Short version:

```bash
pipx install paper-lineage          # once
lineage --check-llm                 # needs LINEAGE_LLM_BASE_URL / _MODEL / _API_KEY
lineage <arxiv url or id> -f html -f json
```

## Working on this repo

```bash
pip install -e '.[dev]' && pytest -q     # tests run offline against saved fixtures
scripts/build_pyz.sh                     # rebuild the single-file dist/lineage.pyz
```

- Source is in `src/paper_lineage/`; see `SPEC.md` for the design and README "How it works".
- Every graph goes through `quality.py` before rendering; keep `tests/test_quality.py` passing.
- Output must stay deterministic and citation-grounded: never add an edge without a verbatim
  evidence sentence found in the source text.
