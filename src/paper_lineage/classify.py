"""Classify each cited paper's role for the citing paper (LLM, with heuristic fallback)."""

from __future__ import annotations

import logging
import re

from .heuristics import SECTION_WEIGHT, heuristic_classification
from .llm import LLM
from .quality import clean_summary
from .models import LINEAGE_ROLES, ROLES, Candidate, Classification, PaperMeta

log = logging.getLogger(__name__)

SYSTEM = """You are a meticulous historian of machine-learning research. You analyse how a CHILD \
paper uses the works it cites, in order to trace the intellectual lineage of a TARGET paper.

Rules:
- Judge ONLY from the citing sentences provided. Do not use outside knowledge about what a cited \
paper contains or how famous it is.
- Never invent papers, never rename them.
- "evidence" must be copied VERBATIM (exact characters) from one of the sentences listed under that \
candidate. If none supports a lineage role, use a non-lineage role.
- Reply with a single JSON object and nothing else."""

ROLE_HELP = """Roles (choose exactly one per candidate):
- builds_on: the CHILD is directly built on top of it (its base model/framework/codebase, or the \
paradigm it extends).
- technique_source: the CHILD's core idea or technique is taken or adapted from it ("inspired by", \
"following X, we ...").
- component: a key ingredient used inside the CHILD's method (teacher model, pretrained encoder, \
specific module), but not the source of the idea.
- alternative: a different approach to the same problem that the CHILD contrasts itself with, or \
concurrent work doing something similar.
- problem_motivation: establishes the problem/goal the CHILD addresses.
- baseline_only: only compared against in experiments.
- dataset_metric_tool: dataset, benchmark, metric, solver, optimizer, library.
- background: generic background, survey-style or laundry-list citation.

directness (0-3): 3 = the CHILD could not exist in its form without it; 2 = explicitly built on / \
key ingredient; 1 = relevant but replaceable; 0 = incidental.
relevance (0-3): how relevant the candidate is to the TARGET paper's research line (not just the CHILD)."""


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+([,.;:)\]])", r"\1", s)
    s = re.sub(r"([(\[])\s+", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def validate_evidence(quote: str, sentences: list[str]) -> tuple[str | None, str]:
    """Map an LLM quote back to a verbatim source sentence.

    Returns (sentence, status) with status in {"exact", "repaired", "rejected"}.
    The returned sentence is always one of `sentences` (never LLM text).
    """
    q = _norm(quote or "")
    q = re.sub(r"\s*(\.\.\.|…)\s*$", "", q)          # prompts truncate long sentences with " ..."
    q = q.strip(" .\"'")
    if not q or not sentences:
        return None, "rejected"
    for s in sentences:
        if q in _norm(s):
            return s, "exact"
    qt = _tokens(q)
    best, best_score = None, 0.0
    for s in sentences:
        st = _tokens(s)
        if not st or not qt:
            continue
        score = len(qt & st) / len(qt)  # fraction of the quote's words found in the sentence
        if score > best_score:
            best, best_score = s, score
    if best is not None and best_score >= 0.8:
        return best, "repaired"
    return None, "rejected"


class HeuristicClassifier:
    name = "heuristic"

    def classify(self, child: PaperMeta, target: PaperMeta, cands: list[Candidate]) -> dict[str, Classification]:
        return {c.bib_key: heuristic_classification(c) for c in cands}


class LLMClassifier:
    name = "llm"

    def __init__(self, llm: LLM, batch_size: int = 12, max_contexts: int = 6, max_chars: int = 450):
        self.llm = llm
        self.batch_size = batch_size
        self.max_contexts = max_contexts
        self.max_chars = max_chars
        self.stats = {"exact": 0, "repaired": 0, "rejected": 0, "missing": 0}

    def _ctx(self, cand: Candidate) -> list[str]:
        ranked = sorted(cand.contexts, key=lambda c: -SECTION_WEIGHT.get(c.section_kind, 0.5))
        return [c.sentence for c in ranked[: self.max_contexts]]

    def build_prompt(self, child: PaperMeta, target: PaperMeta, cands: list[Candidate]) -> str:
        lines = [
            "TARGET paper (whose lineage we trace):",
            f"  Title: {target.title}",
            f"  Abstract: {(target.abstract or '')[:1500]}",
            "",
        ]
        if child.key != target.key:
            lines += [
                "CHILD paper (the citing paper analysed now; an ancestor of the TARGET):",
                f"  Title: {child.title}",
                f"  Abstract: {(child.abstract or '')[:1500]}",
                "",
            ]
        else:
            lines += ["CHILD paper: the TARGET itself.", ""]
        lines += [ROLE_HELP, "", "Candidates (sentences are from the CHILD paper, with section labels):"]
        sect = {c.bib_key: {x.sentence: x.section_kind for x in c.contexts} for c in cands}
        for i, c in enumerate(cands, 1):
            year = c.meta.year if c.meta and c.meta.year else ""
            lines.append(f"[C{i}] \"{c.title}\" ({year})")
            for s in self._ctx(c):
                txt = s if len(s) <= self.max_chars else s[: self.max_chars] + " ..."
                lines.append(f"   - ({sect[c.bib_key].get(s, '?')}) {txt}")
        lines += [
            "",
            'Return: {"results": [{"id": "C1", "role": "...", "directness": 0, "relevance": 0, '
            '"inherited": "<what the CHILD took from it>", '
            '"evidence": "<verbatim sentence or clause from the list above>"}, ...]} '
            "with one entry per candidate.",
            "",
            'Format of "inherited" (lineage roles only, otherwise ""): a lower-case noun phrase of at most 15 words '
            'that completes the sentence "<CHILD> builds on <CANDIDATE>: ___". Name the concrete thing taken '
            "(a model, module, loss, representation, training recipe), not a judgement. No trailing period, do not "
            "start with \"it\" or a verb, do not repeat the CHILD's name.",
            'Good: "per-frame independent noise levels for autoregressive video diffusion". '
            'Bad: "It builds on this work." / "Important foundation." / "Uses VGGT.".',
        ]
        return "\n".join(lines)

    def classify(self, child: PaperMeta, target: PaperMeta, cands: list[Candidate]) -> dict[str, Classification]:
        out: dict[str, Classification] = {}
        for start in range(0, len(cands), self.batch_size):
            chunk = cands[start:start + self.batch_size]
            prompt = self.build_prompt(child, target, chunk)
            try:
                reply = self.llm.complete_json(SYSTEM, prompt)
            except Exception as e:
                log.warning("LLM classification failed (%s); using heuristics for %d candidates", e, len(chunk))
                for c in chunk:
                    out[c.bib_key] = heuristic_classification(c)
                continue
            results = reply.get("results") or []
            by_id = {str(r.get("id", "")).strip("[] "): r for r in results if isinstance(r, dict)}
            for i, c in enumerate(chunk, 1):
                r = by_id.get(f"C{i}")
                if r is None:
                    self.stats["missing"] += 1
                    out[c.bib_key] = heuristic_classification(c)
                    continue
                out[c.bib_key] = self._to_classification(r, c)
        return out

    def _to_classification(self, r: dict, c: Candidate) -> Classification:
        role = str(r.get("role", "background")).strip().lower()
        if role not in ROLES:
            role = "background"
        try:
            directness = max(0, min(3, int(r.get("directness", 0))))
        except (TypeError, ValueError):
            directness = 0
        try:
            relevance = float(r.get("relevance")) if r.get("relevance") is not None else None
        except (TypeError, ValueError):
            relevance = None
        sentences = [x.sentence for x in c.contexts]
        evidence, status = validate_evidence(str(r.get("evidence", "")), sentences)
        self.stats[status] += 1
        inherited = clean_summary(r.get("inherited", ""))
        if evidence is None:
            # Unverifiable claim: never keep it as lineage.
            if role in LINEAGE_ROLES or role == "alternative":
                log.debug("rejecting %s for %r: evidence not found in source", role, c.title)
                role, directness = "background", 0
            evidence = ""
        return Classification(role=role, directness=directness, inherited=inherited, evidence=evidence,
                              relevance=relevance, source="llm")
