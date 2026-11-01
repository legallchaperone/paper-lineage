# paper-lineage — spec

CLI tool: given an arXiv paper, build a **citation-grounded lineage graph** (ancestors →
target) by reading the paper and recursively reading its most direct parents, then render it.

## Design principles (non-negotiable)

1. **Strictly citation-grounded.** Every node is a paper reachable through actual reference
   lists. Every edge carries a **verbatim evidence quote** (the citing sentence) taken from
   the child paper's text. The LLM only *classifies* citations it is shown; it never
   introduces papers. Validate: an LLM-returned quote must be a substring (after whitespace
   normalization) of the contexts we gave it, otherwise drop the edge.
2. **The graph is the truth; the "spine" is only emphasis.** Internally a DAG. Lineage is
   usually several lines merging (technique + base model + ingredient), not a single chain.
3. **Time flows forward.** An ancestor's arXiv v1 date (or publication date) must be ≤ the
   child's v1 date. Within ±3 months → treat as *concurrent*, not ancestor.
4. **Cheap first, LLM second.** Heuristics prune obvious noise before any LLM call.
5. **Everything cached** on disk (HTTP responses, parsed papers, LLM classifications) so
   re-runs and shared ancestors are free.

## CLI

```
lineage <arxiv url | arxiv id>
  --max-depth N          hard depth limit (default 4); auto-stop rules also apply
  --max-nodes N          total budget of expanded papers (default 30)
  --parents-per-node K   max lineage parents kept per paper (default 4)
  --alternatives K       max "done differently" branches per node (default 2)
  --spine auto|none      auto = highlight a dominant line only if one exists (default auto)
  --descendants          also show selected papers that build on the target
  --concurrent           also show concurrent / sibling work around the target
  --format tree|mermaid|json|html   (default tree; may be given multiple times)
  -o, --output PATH
  --cache-dir PATH       (default ~/.cache/paper-lineage)
  --verbose              log every keep/drop decision with reason
```

LLM config via env, **OpenAI-compatible chat-completions API** (user runs a local gateway):
`LINEAGE_LLM_BASE_URL`, `LINEAGE_LLM_API_KEY`, `LINEAGE_LLM_MODEL`
(optionally `LINEAGE_LLM_MODEL_CHEAP` for classification). Use JSON-mode / structured output
when available, with a robust JSON-extraction fallback. Optional `S2_API_KEY`.

Python ≥3.10, packaged with `pyproject.toml` (entry point `lineage`). Suggested deps:
`httpx`, `selectolax` or `beautifulsoup4`, `pydantic`, `typer`, `rich`, `networkx`.
No heavyweight ML deps.

## Pipeline

### 1. Fetch (`fetch.py`)
- Normalize input to arXiv id (strip version, handle `abs/`, `pdf/`, `html/` urls).
- Metadata from arXiv API (`export.arxiv.org/api/query?id_list=`): title, abstract, authors,
  **v1 date**, categories.
- Full text, in order of preference:
  1. `https://arxiv.org/html/<id>` (LaTeXML) — best: sections + `<cite>` links to `#bib.bibN`.
  2. `https://ar5iv.labs.arxiv.org/html/<id>` (same format, covers older papers).
  3. PDF + `pdftotext` fallback with regex citation parsing for both `[12]` / `[3, 7-9]` and
     `(Author et al., 2024a; ...)` styles. Mark resulting contexts `confidence=low`.
- Non-arXiv ancestors (e.g. a 1989 journal paper): no full text; they can be nodes but are
  not expanded (they become leaves / origins).

### 2. Parse (`parse.py`)
Output a `ParsedPaper`:
- `sections`: list of {id, title, kind} with `kind ∈ {abstract, intro, related, prelim,
  method, experiments, discussion, conclusion, appendix, other}` (classify by title keywords;
  method = the section(s) between related/prelim and experiments).
- `bib`: {bib_key → raw entry text, extracted arXiv id / DOI / title / year if present}.
- `citations`: list of {bib_keys[], sentence, section_kind, paragraph_index}. Split into
  sentences; one record per citing sentence (a sentence citing 6 papers yields one record
  with 6 keys — the group size is itself a signal: big groups = laundry list).

### 3. Resolve (`resolve.py`, `s2.py`)
- Bib entry → Semantic Scholar paper id: use arXiv id/DOI if in entry; else S2
  `/graph/v1/paper/search/match?query=<title>`; accept only if normalized-title similarity is
  high and year matches ±1. Batch metadata via `POST /graph/v1/paper/batch`
  (fields: title, year, externalIds, citationCount, publicationDate, embedding.specter_v2).
- **Do not use S2 `contexts`/`isInfluential` for decisions** — verified to mis-attribute
  author-year citations (in arXiv:2507.07982, VGGT's contexts were assigned to SLAM3R).
  S2 is for ID resolution, metadata, embeddings, and the citations (descendants) endpoint.
- Drop junk entries (no title, title == the citing paper, acknowledgement text, etc.).
- S2 client: exponential backoff on 429/5xx, rate limiting (~1 req/s without key), cache.

### 4. Classify (`classify.py`)
**Heuristic pre-score** per cited paper (aggregated over its citing sentences):
- section weight: method/prelim 3, intro 2, discussion 2, related 1, experiments 0.5,
  appendix 0.3
- cue phrases (+): "inspired by", "builds (up)on", "based on", "following", "we extend",
  "we adopt", "we apply ... to/on", "baseline", "teacher", "in contrast", "unlike"
- group-size penalty (sentence cites ≥4 papers)
- generic-hub penalty (very high citationCount *and* only cited in background/experiments)
Keep the top ~25 candidates (+ any with a strong cue phrase) for the LLM.

**LLM classification**: one call per paper, input = target-paper summary (title + abstract),
the child paper's title+abstract, and for each candidate: title, year, and its citing
sentences with section labels. Output per candidate:
- `role ∈ {builds_on, technique_source, component, alternative, problem_motivation,
  baseline_only, dataset_metric_tool, background}`
- `directness` 0–3
- `inherited`: ≤15 words — what the child took from it
- `evidence`: verbatim quote (validated as described above)

Edge selection per paper:
- lineage parents = roles {builds_on, technique_source, component} with directness ≥2,
  top `--parents-per-node`
- alternatives = role `alternative` (contrast citations), top `--alternatives`; they are
  shown as branches, **never expanded**
- `problem_motivation` with directness ≥2 may be kept as a non-expanded context node
- everything else dropped (logged with reason in --verbose)

### 5. Expand (`expand.py`)
Best-first expansion from the target, priority = edge score × relevance to **target** (not to
the immediate child). Relevance = cosine(SPECTER2 embedding of candidate, target); if no
embedding, ask the LLM for 0–3 relevance in the classification call.
Stop expanding a node when any holds:
- depth ≥ `--max-depth`, or node budget exhausted
- **hub**: foundational generic paper (citationCount above a threshold, e.g. >20k, and
  relevance to target low) → keep as a "foundation" node, don't expand
- **topic drift**: relevance to target below threshold → drop
- no full text available
**Origin detection (auto stop)**: a node whose kept parents are all empty/hubs/drifted is an
*origin* of its line; mark it `is_origin=True`.
Deduplicate by S2 id; when a paper is reached from multiple children, merge (a shared
ancestor is fine, that's the DAG).

### 6. Assemble (`graph.py`)
- Node: id, title, year, v1 date, url, kind ∈ {target, ancestor, origin, foundation,
  alternative, motivation, descendant, concurrent}, depth.
- Edge: parent → child, role, directness, inherited, evidence, section_kind.
- Enforce time order (rule 3); a violating edge is re-labelled concurrent or dropped.
- **Lines**: group ancestors into lines (connected chains of the same role family; e.g.
  technique line, base-model line, ingredient line), each with a short LLM label generated
  only from node titles + `inherited` fields.
- **Spine (`--spine auto`)**: compute the max-weight root→target path; highlight it only if
  its weight beats the next line by a margin AND its first edge into the target is
  builds_on/technique_source. Otherwise no spine; lines are shown as peers.
- Prune: remove nodes not on any path to the target.

### 7. Optional extensions
- `--descendants`: S2 `/paper/{id}/citations` → candidate citing papers; keep those whose
  citing contexts (fetch their arXiv HTML when available, else S2 contexts as low-confidence)
  classify as builds_on/technique_source. Cap at 5, sorted by relevance × citations.
- `--concurrent`: papers within ±3 months of the target that are (a) cited by it as
  alternative/concurrent, or (b) cited alongside it by its descendants. Cap at 5.
Both are rendered visually distinct (e.g. dashed) and never expanded.

### 8. Render (`render/`)
- `tree`: terminal rendering with `rich`: lines as sections, spine bold, alternatives as
  `── alt:` branches, foundations listed once at the bottom.
- `mermaid`: `flowchart TB`, subgraph per line, edge labels = `inherited`.
- `json`: full graph incl. evidence (stable schema, versioned).
- `html`: single self-contained file (mermaid from a pinned CDN version or inline SVG);
  clicking a node/edge shows the evidence quote and links to arXiv.

## Testing

- Unit tests with **saved fixtures** (no network in tests): arXiv HTML of 2507.07982
  (author-year style) and one numeric-citation paper (e.g. 2407.01392); a PDF-fallback case.
- LLM calls behind an interface; tests use a fake/recorded client.
- Evidence-quote validator tests (rejects paraphrased quotes).
- **Golden test — Geometry Forcing (arXiv:2507.07982), level-1 classification**:
  - must keep as lineage parents: REPA (2410.06940), History-Guided Video Diffusion / DFoT
    (2502.06764), VGGT (2503.11651), Diffusion Forcing (2407.01392),
    Next-Frame Diffusion (2506.01380)
  - should appear as alternative/concurrent: VideoREPA, Self Forcing, and at least one of
    {World-consistent Video Diffusion with Explicit 3D Modeling, AETHER, WORLDMEM}
  - must drop: Euler 1845, FVD, LPIPS, SSIM, FID/TTUR, RealEstate10K (Stereo magnification),
    VPT, GPT-2/3, Attention Is All You Need
- Integration test (marked, network + LLM, skipped by default): full run on 2507.07982
  with defaults finishes < 10 min, ≤ 30 expanded nodes, and contains these lines:
  REPA ← DDAE-style "diffusion as representation learner"; Diffusion Forcing → DFoT;
  DUSt3R → VGGT.

## Milestones (each ends with passing tests + a short CHANGELOG entry)

1. Skeleton, CLI, cache, arXiv fetch + HTML parse + resolve (fixtures).
2. Heuristic pre-score + LLM classification + quote validation + golden test.
3. Recursive expansion, stop rules, DAG assembly, spine, time-order checks.
4. Renderers (tree, mermaid, json, html).
5. `--descendants`, `--concurrent`, PDF fallback.
6. Run on 3 real papers, write `docs/examples/` with outputs and a short error analysis.
