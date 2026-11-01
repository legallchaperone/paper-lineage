# paper-lineage

Give it an arXiv paper; it reads the paper, figures out which cited works it *actually descends from*,
reads those recursively, and prints the lineage as a mind-map — from the origins of each line of work
down to your paper.

```
$ lineage https://arxiv.org/abs/2507.07982 --evidence

Lineage of Geometry Forcing: Marrying Video Diffusion and 3D Representation ... (2025)
No single main path: no dominant line: technique: REPA (…) vs base: DFoT (…); showing lines as peers
Problem context: World Models (2018)

Line 1: technique: REPA
  Denoising Diffusion Autoencoders ... (2023)  [origin]
  │ ↓ idea: diffusion features are discriminative representations
  REPA (2024)
  │ ↓ idea: align diffusion features with a pretrained encoder
  │     “In this work, inspired by recent advances in REPA ( Yu et al., 2024a ), we propose ...”
  Geometry Forcing (2025)

Line 2: base: DFoT
  Diffusion Forcing (2024)
  │ ↓ built on ...
  DFoT (2025)
  ...
Line 3: ingredient: VGGT
  DUSt3R (2023) → VGGT (2025) → Geometry Forcing
Done differently: VideoREPA (2025), World-consistent Video Diffusion with Explicit 3D Modeling (2024)
```
(illustrative; see `docs/examples/` for real output)

## Design in one paragraph

A paper's lineage is usually **several lines merging** (a technique line, a base-model line, an
ingredient line), not one chain — so the result is a DAG drawn as *lines* (tributaries) that flow into
the target, with up to two "done differently" **alternatives** per node. A **main path** is highlighted
only when one line clearly dominates (`--spine auto`); otherwise lines are shown as peers. Everything is
**strictly citation-grounded**: every edge carries the verbatim sentence from the citing paper that
justifies it, the LLM only classifies citations it is shown (it never introduces papers), and an
LLM-returned quote that cannot be found in the source text gets the edge dropped.

## Install

**No pip needed.** Download `lineage.pyz` (a single file with everything bundled) and run it with the
`python3` you already have (3.9 or newer — check with `python3 --version`):

```bash
python3 lineage.pyz --help
```

Optionally make it a command: `chmod +x lineage.pyz && mv lineage.pyz ~/bin/lineage` (any directory on
your PATH), then just `lineage ...`.

No `python3` at all? On macOS run `xcode-select --install` (gives python3); on Windows install Python
from python.org or the Microsoft Store. Optional: `pdftotext` (poppler) for the PDF fallback on papers
without arXiv HTML — macOS `brew install poppler`, Debian/Ubuntu `apt install poppler-utils`.

From source (developers): `pip install -e '.[dev]'`; rebuild the single file with
`scripts/build_pyz.sh`.

## Configure the LLM (any OpenAI-compatible endpoint, e.g. a local gateway)

```bash
export LINEAGE_LLM_BASE_URL=http://localhost:3000/v1   # your gateway; /chat/completions is appended
export LINEAGE_LLM_MODEL=<model name your gateway serves>
export LINEAGE_LLM_API_KEY=<key, if the gateway needs one>
python3 lineage.pyz --check-llm                         # one tiny request to verify the setup
```

Optional: `S2_API_KEY` (Semantic Scholar key — raises rate limits, makes runs much faster).
Without an LLM configured the tool falls back to `--no-llm` heuristics (decent for the first
generation, noticeably weaker deeper in the tree — see "Quality" below).

## Usage

```bash
lineage <arxiv url or id> [options]

  --max-depth N           hard limit on generations traced back (default 4)
  --max-nodes N           max papers read, incl. the target (default 30)
  --parents-per-node K    lineage parents kept per paper (default 4)
  --alternatives K        "done differently" branches per node (default 2; target + 1st generation)
  --min-relevance X       min SPECTER2 similarity to the target (default 0.86) — stops topic drift
  --spine auto|none       highlight a main path only if one line dominates (default auto)
  --descendants N         also show N papers that build on the target (verified from their text)
  --concurrent N          also show N concurrent papers (same time window, cite the same parents)
  -f, --format FMT        tree | mermaid | json | html   (repeatable; default tree)
  -o, --output STEM       output path stem for file formats (default lineage-<id>)
  --evidence              print the evidence sentence under each edge in the tree view
  --no-llm                heuristics only
  --cache-dir DIR         default ~/.cache/paper-lineage (HTTP + LLM replies are cached)
  --full                  tree/mermaid: draw every kept paper, not only the main chains
  --render RUN.json       re-render a saved -f json run (no network, no LLM)
  -v, --verbose           log every keep/drop decision with its reason
```

`-f html` writes a self-contained page in a hand-drawn, Excalidraw-like style (no internet needed to
view it). On the left, a sketchy canvas: each line of work is a titled frame, papers are hand-drawn boxes
and labelled arrows flow left-to-right into your paper, with "done differently" alternatives as dashed
boxes. On the right, a walkthrough with one section per frame and one step per paper: *X is built on Y:
what it inherited*, followed by the sentence quoted from X. Scrolling the text glides the canvas to the
frame and circles the paper being discussed; click a paper on either side to jump to it, drag/pinch to
pan and zoom, **Whole canvas** to zoom out. Formulas in quotes and titles are typeset (MathML), never
shown as raw LaTeX.

The terminal (`tree`) and `mermaid` views show only the main chains by default; add `--full` for
everything. `-f json` keeps the full graph plus the decision log (`decisions`), which is the first
place to look when a result surprises you. Re-render a saved run without fetching anything:

```bash
python3 lineage.pyz --render lineage-2609.24984.json -f html
```

## How it works

1. **Fetch** — arXiv LaTeXML HTML (`arxiv.org/html/<id>`, then ar5iv), falling back to the PDF via
   `pdftotext`. The HTML is preferred because each `<cite>` links to its exact bibliography entry.
   Section structure comes from LaTeXML element ids, not DOM nesting (real arXiv HTML is sometimes
   malformed — e.g. an unclosed `<figure>` that swallows the appendix).
2. **Parse** — every sentence that cites something, with its section kind (intro / related / prelim /
   method / experiments / appendix / figure / table) and the offset of each citation group in it.
   Unlinked author-year citations are recovered through the bibliography labels.
3. **Resolve** — bib entries → Semantic Scholar papers by title containment / arXiv id against the
   paper's S2 reference list. S2's own per-reference `contexts`/`isInfluential` are **not used**: they
   mis-attribute author-year citations (in Geometry Forcing, VGGT's mentions are credited to SLAM3R).
4. **Pre-score (cheap)** — section weights, cue phrases attributed to the *specific* citation group
   they precede ("builds upon (A; B) and incorporates (C)"), name mentions ("REPA", "VGGT"),
   laundry-list penalty, implementation-detail and metric/dataset detection, hub penalty.
5. **Classify (LLM)** — the top ~25 candidates per paper, in batches of 12, into
   `builds_on | technique_source | component | alternative | problem_motivation | baseline_only |
   dataset_metric_tool | background`, with directness 0–3, a ≤15-word "what was inherited", and a
   verbatim evidence quote that is validated against the source.
6. **Expand** — best-first, only through lineage parents (directness ≥ 2), with these stops:
   depth limit, node budget, **relevance to the target** (SPECTER2 cosine; measured against the
   *target*, not the immediate child, which is what prevents drifting into e.g. time-series
   forecasting from Diffusion Forcing's references), **generic foundations** (≥15k citations, or
   components ≥3k — DDPM, DiT, Transformer… are shown once but never expanded), **time order**
   (a "parent" published after the child is concurrent work, not an ancestor), and **origin
   detection** (a paper whose own parents are all off-topic/foundations starts its line).
7. **Assemble** — lines are grouped by the target's direct parents; each line's chain is the best path
   through its head, scored by role × directness × relevance minus a per-hop penalty (so long chains of
   weak links don't win).
8. **Extras** — `--descendants`: citing papers whose own text shows they build on the target
   (fetched, parsed and classified the same way). `--concurrent`: bibliographic coupling — papers
   within ±90 days that cite ≥2 of the target's direct parents.

## Quality and cost

- First generation is the most reliable (the target's text is rich and specific); deeper levels rely
  more on the classifier. Heuristic mode (`--no-llm`) gets the first generation of Geometry Forcing
  right (REPA, DFoT, VGGT, Diffusion Forcing, Next-Frame Diffusion) but mistakes implementation-detail
  citations for lineage deeper down. Use an LLM.
- LLM cost: roughly 8k prompt tokens per paper read → ~250k tokens for a default 30-paper run.
  Everything is cached, so re-runs and shared ancestors are free.
- Wall time is dominated by rate limits (Semantic Scholar ≈1 req/s without a key, arXiv): a 30-paper
  run takes ~5–15 minutes cold, seconds when cached.
- Known limits: papers without arXiv full text are leaves; the PDF fallback is lower-confidence;
  uncited conceptual ancestors (e.g. FitNets for REPA) are deliberately never added.

## Development

```bash
pip install -e '.[dev]' && pytest -q
```
Tests run offline against saved fixtures (Geometry Forcing's HTML + S2 references, Diffusion Forcing's
numeric-citation HTML, pdftotext output for both). `tests/test_llm_path.py` drives the full LLM path
with a fake model that reads the real prompts, including a paraphrased quote (repaired) and an
invented one (rejected). See `SPEC.md` for the original design spec.
