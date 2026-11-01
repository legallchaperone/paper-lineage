"""Cheap, explainable pre-scoring of cited papers (and the --no-llm classifier)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .models import Candidate, Citation, Classification

SECTION_WEIGHT = {
    "abstract": 3.0, "method": 3.0, "prelim": 3.0, "intro": 2.0, "discussion": 1.5,
    "figure": 1.5, "related": 1.0, "conclusion": 1.0, "experiments": 0.5, "table": 0.3,
    "appendix": 0.3, "other": 0.5,
}

CUES = {
    "build": re.compile(
        r"\b(build(s|ing)?\s+(up)?on|built\s+(up)?on|based\s+on|on\s+top\s+of|we\s+extend|extends?|extending|"
        r"we\s+adopt|adopts?|adopting|we\s+apply|apply(ing)?\s+\w+(\s+\w+)?\s+(on|to)|we\s+follow|following|"
        r"incorporat(e|es|ing)|we\s+leverage|leverag(e|es|ing)|initiali[sz]ed\s+from|fine-?tun\w*|"
        r"generali[sz](e|es|ing)|we\s+use|we\s+employ|builds?\s+upon|backbone\s+of|baseline\s+(model|method))\b", re.I),
    "inspire": re.compile(
        r"\b(inspired\s+by|motivated\s+by|in\s+the\s+spirit\s+of|draw(s|ing)?\s+(inspiration|on)|"
        r"borrow\w*|taking\s+inspiration)\b", re.I),
    "contrast": re.compile(
        r"\b(in\s+contrast|unlike|differ(s|ent)?\s+from|whereas|alternatively|orthogonal\s+to|instead\s+of|"
        r"rather\s+than|concurrent(ly)?|in\s+parallel|while\s+(they|these|such)|"
        r"(most\s+)?similar\s+to|akin\s+to|analogous\s+to|closely\s+related)\b", re.I),
    "eval": re.compile(
        r"\b(metrics?|fvd|fid|lpips|ssim|psnr|is\s+score|benchmarks?|datasets?|evaluat\w+|test\s+set|"
        r"solver|optimi[sz]er|toolkit|library|implementation\s+of)\b", re.I),
}

# words that mark an implementation-detail citation ("we use the DDIM sampler", "following X we set lr")
DETAIL = re.compile(
    r"\b(sampler|sampling\s+steps|optimi[sz]er|learning\s+rate|batch\s+size|hyper-?parameters?|"
    r"initiali[sz]\w*|embedding|tokeni[sz]\w*|vae|autoencoder|encoder\s+and\s+decoder|augmentation|"
    r"schedule|warm-?up|ema|normali[sz]ation|positional|coordinate\s+frame|activation|dropout|"
    r"implementation|codebase|checkpoints?|pre-?processing|metric|layer\s+norm|adaln|block)\b", re.I)

CUE_SECTION_FACTOR = {"abstract": 1.5, "intro": 1.5, "prelim": 1.0, "method": 1.0, "figure": 1.0,
                      "discussion": 1.0, "conclusion": 1.0, "related": 0.7, "experiments": 0.4,
                      "table": 0.0, "appendix": 0.3, "other": 0.5}

_ALIAS_STOP = {"we", "our", "in", "the", "fig", "figure", "table", "sec", "section", "eq", "appendix", "et",
               "al", "and", "or", "of", "see", "e.g", "i.e", "as", "by", "from", "with", "to", "rgb", "fps",
               "fid", "fvd", "lpips", "ssim", "psnr", "gpu", "gpus", "ar", "3d", "2d", "4d", "ii", "iii"}


def alias_from_contexts(contexts: list[Citation], key: str) -> str | None:
    """The name a citing paper uses right before the citation, e.g. 'REPA (Yu et al., 2024a)'."""
    counts: dict[str, int] = {}
    for c in contexts:
        for off, ks in c.spans:
            if key not in ks or len(ks) > 2:
                continue
            before = c.sentence[:off].rstrip(" ([")
            m = re.search(r"([A-Za-z][A-Za-z0-9\-+]*)\s*$", before)
            if not m:
                continue
            w = m.group(1)
            if w.lower() in _ALIAS_STOP or len(w) < 2 or len(w) > 16:
                continue
            if not (sum(ch.isupper() for ch in w) >= 2 or (any(ch.isdigit() for ch in w) and w[0].isupper())):
                continue
            counts[w] = counts.get(w, 0) + 1
    if not counts:
        return None
    w, n = max(counts.items(), key=lambda x: x[1])
    return w if n >= 2 or len(contexts) == 1 else None


EVAL_TITLE = re.compile(
    r"\b(dataset|benchmark|metric|image quality assessment|a large[- ]scale|database|"
    r"evaluation|toolkit|library|stereo magnification|perceptual metric|adam|optimizer|solver)\b", re.I)

# how far before a citation group a cue may sit to be attributed to it
CUE_WINDOW = 140


@dataclass
class HeuristicConfig:
    hub_citations: int = 15000


def _cue_hits(c: Citation, key: str, cue: str) -> int:
    """Count cue matches that are attributed to `key`'s citation group in this sentence.

    A cue is attributed to the first citation group that starts after it (within CUE_WINDOW chars);
    'contrast' cues may also follow the citation ("X (cite) ... In contrast, we").
    """
    rx = CUES[cue]
    spans = sorted(c.spans) or [(len(c.sentence), c.keys)]
    hits = 0
    for m in rx.finditer(c.sentence):
        p = m.start()
        nxt = next(((a, ks) for a, ks in spans if a >= m.end()), None)
        if nxt and nxt[0] - m.end() <= CUE_WINDOW and key in nxt[1]:
            hits += 1
        elif cue in ("contrast", "eval") and key in c.keys and nxt is None:
            hits += 1
        elif cue == "eval" and key in c.keys:
            # "the standard Euler solver (Euler, 1845)" style: cue right before the cite
            near = [a for a, ks in spans if key in ks and 0 <= a - p <= 60]
            hits += 1 if near else 0
    return hits


def _group_factor(c: Citation, key: str) -> float:
    for _, ks in c.spans:
        if key in ks:
            n = len(ks)
            break
    else:
        n = c.group_size
    return 1.0 if n <= 2 else 0.7 if n == 3 else 0.35


def name_mentions(title: str, full_text: str, alias: str | None = None) -> int:
    """How often the paper's method name (e.g. 'VGGT', 'REPA', 'DUSt3R') is used in the citing text."""
    if alias:
        return len(re.findall(rf"(?<![\w-]){re.escape(alias)}(?![\w-])", full_text))
    if ":" not in title:
        return 0
    head = title.split(":", 1)[0].strip()
    if not head or len(head) > 24 or len(head.split()) > 3:
        return 0
    # require something name-like: has uppercase beyond first char or digits
    if not (re.search(r"[A-Z].*[A-Z]", head) or re.search(r"\d", head) or len(head.split()) > 1):
        return 0
    return len(re.findall(rf"(?<![\w-]){re.escape(head)}(?![\w-])", full_text))


def score_candidate(cand: Candidate, full_text: str, cfg: HeuristicConfig | None = None) -> None:
    """Fills cand.score, cand.features, cand.guess_role."""
    cfg = cfg or HeuristicConfig()
    key = cand.bib_key
    f: dict = {"sections": {}, "cue_build": 0, "cue_inspire": 0, "cue_contrast": 0, "cue_eval": 0}
    ctx_score = 0.0
    f["cue_detail"] = 0
    f["intro_mentions"] = 0
    for c in cand.contexts:
        f["sections"][c.section_kind] = f["sections"].get(c.section_kind, 0) + 1
        ctx_score += SECTION_WEIGHT.get(c.section_kind, 0.5) * _group_factor(c, key)
        sf = CUE_SECTION_FACTOR.get(c.section_kind, 0.5)
        detail = bool(DETAIL.search(c.sentence))
        for cue in ("build", "inspire", "contrast", "eval"):
            hits = _cue_hits(c, key, cue)
            if not hits:
                continue
            if cue in ("build", "inspire") and detail:
                f["cue_detail"] += hits
                hits *= 0.25
            f[f"cue_{cue}"] += hits * (sf if cue != "eval" else 1.0)
        if c.section_kind in ("abstract", "intro"):
            f["intro_mentions"] += 1
    for k in ("cue_build", "cue_inspire", "cue_contrast"):
        f[k] = round(f[k], 2)
    alias = cand.meta.alias if cand.meta and cand.meta.alias else alias_from_contexts(cand.contexts, key)
    if alias and cand.meta and not cand.meta.alias:
        cand.meta.alias = alias
    f["mentions"] = name_mentions(cand.title, full_text, alias)
    f["n_contexts"] = len(cand.contexts)
    secs = f["sections"]
    core = sum(v for k, v in secs.items() if k in ("abstract", "intro", "prelim", "method", "figure", "discussion"))
    f["only_experiments"] = core == 0 and any(k in secs for k in ("experiments", "table"))
    cc = cand.meta.citation_count if cand.meta else None
    f["hub"] = bool(cc and cc >= cfg.hub_citations)
    f["eval_title"] = bool(EVAL_TITLE.search(cand.title or ""))

    score = (ctx_score
             + 2.0 * min(f["cue_build"], 3)
             + 2.5 * min(f["cue_inspire"], 2)
             + 0.8 * min(f["cue_contrast"], 2)
             + 0.3 * min(f["mentions"], 15)
             + 1.0 * min(f["intro_mentions"], 3))
    if f["only_experiments"]:
        score *= 0.5
    if f["eval_title"] or (f["cue_eval"] and f["cue_eval"] >= max(1, len(cand.contexts) // 2)):
        score *= 0.4
    if f["hub"]:
        score *= 0.6
    cand.score = round(score, 3)
    cand.features = f
    cand.guess_role = _guess_role(f, score)


def _guess_role(f: dict, score: float) -> str:
    secs = f["sections"]
    core_build = f["cue_build"] >= 0.9
    if f["eval_title"] or (f["cue_eval"] and not core_build and not f["cue_inspire"] and f["mentions"] < 3):
        return "dataset_metric_tool"
    if f["cue_inspire"] >= 0.9 and f["cue_inspire"] >= f["cue_build"]:
        return "technique_source"
    if core_build and not f["only_experiments"]:
        return "builds_on"
    if f["cue_contrast"] >= 0.5:
        return "alternative"
    if f["mentions"] >= 3 and any(k in secs for k in ("method", "prelim")):
        return "component"
    if f["cue_detail"] and not core_build:
        return "dataset_metric_tool" if f["mentions"] < 3 else "component"
    if f["only_experiments"]:
        return "baseline_only"
    if set(secs) <= {"intro", "abstract"} and f["n_contexts"] <= 2:
        return "problem_motivation"
    return "background"


def directness_from_score(score: float) -> int:
    if score >= 9:
        return 3
    if score >= 5:
        return 2
    if score >= 2.5:
        return 1
    return 0


def best_evidence(cand: Candidate, prefer: str | None = None) -> str:
    """Pick the most informative citing sentence (cue-bearing, core section, small group)."""
    def rank(c: Citation) -> float:
        r = SECTION_WEIGHT.get(c.section_kind, 0.5) * _group_factor(c, cand.bib_key)
        for cue in ("inspire", "build", "contrast"):
            if _cue_hits(c, cand.bib_key, cue):
                r += 3 if cue != "contrast" or prefer == "alternative" else 1
        return r - 0.002 * len(c.sentence)
    if not cand.contexts:
        return ""
    return max(cand.contexts, key=rank).sentence


def heuristic_classification(cand: Candidate) -> Classification:
    return Classification(
        role=cand.guess_role,
        directness=directness_from_score(cand.score),
        inherited="",
        evidence=best_evidence(cand, prefer=cand.guess_role),
        source="heuristic",
    )


def cosine(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else None
