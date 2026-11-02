---
name: paper-lineage
description: Trace where an arXiv paper comes from. Use when the user gives an arXiv link or id and asks about a paper's origins, ancestors, what it builds on, its research lineage, or wants a family-tree / mind-map of related prior work. Produces a citation-grounded lineage with a verbatim evidence sentence for every link, as a terminal tree or an interactive HTML page.
---

# paper-lineage

`lineage` reads an arXiv paper, decides which cited works it actually descends from, reads those
recursively, and outputs the lineage: several *lines* of work (technique, base model, ingredient)
flowing into the paper. Every link carries a sentence quoted from the newer paper.

## 1. Make sure it is installed

```bash
lineage --help || pipx install paper-lineage || pipx install git+https://github.com/legallchaperone/paper-lineage
```

No pipx? Use `uv tool install paper-lineage`, or download the single-file `lineage.pyz` and run
`python3 lineage.pyz ...` (needs Python 3.9+). Do not `pip install` into the system Python.

## 2. Check the LLM is configured

The tool needs any OpenAI-compatible chat endpoint:

```bash
export LINEAGE_LLM_BASE_URL=...   # "/chat/completions" is appended
export LINEAGE_LLM_MODEL=...
export LINEAGE_LLM_API_KEY=...    # if the endpoint needs one
lineage --check-llm               # one tiny request; run this first
```

If the user has a ChatGPT subscription instead of an API endpoint, use the Codex CLI
(`npm i -g @openai/codex`, then the user runs `codex login` and picks "Sign in with ChatGPT"), then
`lineage --llm codex --check-llm` and add `--llm codex` to every run (or `export LINEAGE_LLM_PROVIDER=codex`).

Never invent or guess these values; ask the user if they are not set. If the user has no LLM, use
`--no-llm` (heuristics only: fine for direct parents, weaker for deeper ancestors; tell the user).
`S2_API_KEY` (Semantic Scholar) is optional and makes runs much faster.

## 3. Run it

Accepts an arXiv URL or bare id (`2507.07982`).

```bash
# Interactive page for the user (writes lineage-<id>.html and lineage-<id>.json)
lineage 2507.07982 -f html -f json

# Quick answer you can read yourself, with evidence sentences
lineage 2507.07982 --evidence
```

Useful options: `--max-depth N` (default 4), `--max-nodes N` (default 30, papers read),
`--descendants N` (papers that build on the target), `--full` (show every kept paper),
`-o STEM` (output path), `-v` (log every keep/drop decision).

A cold default run takes roughly 5 to 15 minutes (rate limits) and ~250k LLM tokens. Replies are
cached in `~/.cache/paper-lineage`, so reruns are fast. Run it in the background and tell the user
it will take a while. For a quick look, use `--max-depth 2 --max-nodes 12`.

Re-render a saved run without network or LLM: `lineage --render lineage-<id>.json -f html`.

## 4. Report the result

- Give the user the HTML path (they open it in a browser; it is one self-contained file).
- Summarize in plain words: the lines (e.g. "base: Helios", "ingredient: VGGT"), what each older
  paper contributed, and any "done differently" alternatives.
- Quote the evidence sentence when you claim "X builds on Y". Do not add ancestors that are not in
  the output; the tool is deliberately citation-grounded and never infers uncited ones.
- If there is "no single main path", say the lines are peers rather than picking one.

## Limits to mention when relevant

Papers without arXiv full text are leaves; the PDF fallback needs `pdftotext` (poppler) and is
lower confidence; generic foundations (DDPM, Transformer...) are shown but not expanded. If a
result looks wrong, run with `-f json` and read the `decisions` log.
