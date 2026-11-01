"""Formulas in quotes and titles: find LaTeX, typeset it in the page, never show raw LaTeX.

New runs: the parser wraps every formula from arXiv's HTML in `$…$`.
Older runs (and LLM summaries) may contain bare LaTeX such as `\\mathcal{L}_{\\text{depth}}`; those
fragments are detected heuristically.

`rich(text)` returns HTML where each formula is a `<span class="tex" data-tex="…">` holding a readable
Unicode fallback; the page's script turns it into MathML with Temml. `plain(text)` gives the Unicode
form for places that cannot hold markup (SVG labels, tooltips, page title).
"""

from __future__ import annotations

import html
import re

_FORMAT = r"text|mathrm|mathbf|mathcal|mathit|mathsf|mathtt|boldsymbol|mathbb|mathfrak|mathscr|operatorname|textbf|textit|bm|rm|bf"
_SYMBOLS = {
    "times": "×", "cdot": "·", "in": "∈", "notin": "∉", "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥",
    "neq": "≠", "approx": "≈", "propto": "∝", "rightarrow": "→", "to": "→", "leftarrow": "←", "mapsto": "↦",
    "sim": "~", "pm": "±", "infty": "∞", "sum": "Σ", "prod": "Π", "int": "∫", "partial": "∂", "nabla": "∇",
    "ldots": "…", "dots": "…", "cdots": "⋯", "forall": "∀", "exists": "∃", "cup": "∪", "cap": "∩",
    "subset": "⊂", "subseteq": "⊆", "odot": "⊙", "oplus": "⊕", "otimes": "⊗", "circ": "∘", "top": "⊤",
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ",
    "eta": "η", "theta": "θ", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π",
    "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ",
    "Psi": "Ψ", "Omega": "Ω", "ell": "ℓ", "|": "‖", "%": "%", "{": "{", "}": "}", "_": "_", "&": "&", "#": "#",
}
_SUB_KEYS, _SUP_KEYS = "0123456789+-=()aeoxhklmnpstij", "0123456789+-=()ni"
_SUB = str.maketrans(_SUB_KEYS, "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₕₖₗₘₙₚₛₜᵢⱼ")
_SUP = str.maketrans(_SUP_KEYS, "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ")


def _script(body: str, table, keys: str, mark: str) -> str:
    """x_{12} -> x₁₂ when every character has a Unicode sub/superscript form, else x_12."""
    body = body.strip()
    if body and all(ch in keys for ch in body):
        return body.translate(table)
    return mark + body

_CMD = re.compile(r"\\[A-Za-z]+|\\[%{}_&#]")
_DELIM = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)", re.S)
_OPS = {r"\times", r"\cdot", r"\in", r"\leq", r"\geq", r"\le", r"\ge", r"\approx", r"\to", r"\rightarrow",
        r"\sim", r"\pm", "=", "+", "-", "<", ">", "×"}


def _balanced(s: str) -> bool:
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{" and (i == 0 or s[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or s[i - 1] != "\\"):
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def to_unicode(tex: str) -> str:
    """Best-effort readable form of a LaTeX fragment (used as fallback and for SVG/tooltips)."""
    t = tex
    for _ in range(4):
        t = re.sub(r"\\(?:%s)\s*\{([^{}]*)\}" % _FORMAT, r"\1", t)
        t = re.sub(r"\\(?:hat|widehat)\s*\{([^{}]*)\}", lambda m: m.group(1) + "\u0302", t)
        t = re.sub(r"\\(?:tilde|widetilde)\s*\{([^{}]*)\}", lambda m: m.group(1) + "\u0303", t)
        t = re.sub(r"\\(?:bar|overline)\s*\{([^{}]*)\}", lambda m: m.group(1) + "\u0304", t)
        t = re.sub(r"\\(?:vec)\s*\{([^{}]*)\}", lambda m: m.group(1) + "\u20d7", t)
        t = re.sub(r"\\(?:dot)\s*\{([^{}]*)\}", lambda m: m.group(1) + "\u0307", t)
        t = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", t)
        t = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"√(\1)", t)
    t = re.sub(r"\\([A-Za-z]+|[%{}_&#|])", lambda m: _SYMBOLS.get(m.group(1), m.group(1) if len(m.group(1)) > 1 else ""),
               t)
    t = re.sub(r"\\[,;:! ]", " ", t)

    t = re.sub(r"_\{([^{}]*)\}|_(\w)", lambda m: _script(m.group(1) if m.group(1) is not None else m.group(2),
                                                         _SUB, _SUB_KEYS, "_"), t)
    t = re.sub(r"\^\{([^{}]*)\}|\^(\w)", lambda m: _script(m.group(1) if m.group(1) is not None else m.group(2),
                                                           _SUP, _SUP_KEYS, "^"), t)
    t = t.replace("{", "").replace("}", "")
    return re.sub(r"\s{2,}", " ", t).strip()


def _is_texy(tok: str) -> bool:
    core = tok.strip(".,;:()[]")
    if not core:
        return False
    if _CMD.search(core):
        return True
    return bool(re.search(r"[A-Za-z0-9)\]}]_\{|[A-Za-z0-9)\]}]\^\{|^[A-Za-z]_[A-Za-z0-9]$|^[A-Za-z]\^[A-Za-z0-9]$", core))


def split_math(text: str) -> list[tuple[bool, str]]:
    """[(is_math, chunk)], math chunks are LaTeX without delimiters."""
    out: list[tuple[bool, str]] = []
    pos = 0
    for m in _DELIM.finditer(text):
        if m.start() > pos:
            out.extend(_split_bare(text[pos:m.start()]))
        tex = next(g for g in m.groups() if g is not None)
        out.append((True, tex.strip()))
        pos = m.end()
    if pos < len(text):
        out.extend(_split_bare(text[pos:]))
    return [(m, s) for m, s in out if s]


def _split_bare(text: str) -> list[tuple[bool, str]]:
    """Find undelimited LaTeX runs (older runs, LLM summaries)."""
    if "\\" not in text and "_" not in text and "^" not in text:
        return [(False, text)]
    toks = re.split(r"(\s+)", text)
    words = toks[0::2]
    texy = [_is_texy(w) for w in words]
    # pull operands into runs around binary operators: "2 \times 2", "x = y"
    for i, w in enumerate(words):
        if w in _OPS and ((i > 0 and texy[i - 1]) or (i + 1 < len(words) and texy[i + 1]) or w.startswith("\\")):
            texy[i] = True
            for j in (i - 1, i + 1):
                if 0 <= j < len(words) and len(words[j].strip(".,;:")) <= 6 and not re.search(r"[a-z]{3,}", words[j]):
                    texy[j] = True
    out: list[tuple[bool, str]] = []
    buf: list[str] = []
    is_m = False

    def flush():
        nonlocal buf
        if not buf:
            return
        s = "".join(buf)
        if is_m:
            core = s.rstrip()
            trail = ""
            while core and core[-1] in ".,;:":
                trail = core[-1] + trail
                core = core[:-1]
            lead = ""
            while core and core[0] in "([" and not _balanced_paren(core):
                lead += core[0]
                core = core[1:]
            if core and _balanced(core):
                if lead:
                    out.append((False, lead))
                out.append((True, core))
                out.append((False, trail + s[len(s.rstrip()):]))
            else:
                out.append((False, s))
        else:
            out.append((False, s))
        buf = []

    for i, w in enumerate(words):
        sep = toks[2 * i + 1] if 2 * i + 1 < len(toks) else ""
        if texy[i] != is_m and buf:
            # keep the separator outside math runs
            flush()
        is_m = texy[i]
        if is_m:
            nxt_is_m = i + 1 < len(words) and texy[i + 1]
            buf.append(w + (sep if nxt_is_m else ""))
            if not nxt_is_m:
                flush()
                is_m = False
                buf = [sep] if sep else []
        else:
            buf.append(w + sep)
    flush()
    merged: list[tuple[bool, str]] = []
    for m, s in out:
        if merged and not m and not merged[-1][0]:
            merged[-1] = (False, merged[-1][1] + s)
        else:
            merged.append((m, s))
    return merged


def _balanced_paren(s: str) -> bool:
    return s.count("(") == s.count(")") and s.count("[") == s.count("]")


def rich(text: str | None) -> str:
    """Escaped HTML with formulas as Temml placeholders (readable Unicode until typeset)."""
    if not text:
        return ""
    parts = []
    for is_m, chunk in split_math(text):
        if is_m:
            parts.append(f'<span class="tex" data-tex="{html.escape(chunk, quote=True)}">'
                         f"{html.escape(to_unicode(chunk))}</span>")
        else:
            parts.append(html.escape(tidy(chunk)))
    return "".join(parts)


def plain(text: str | None) -> str:
    """Unicode-only form (SVG text, tooltips, <title>)."""
    if not text:
        return ""
    return tidy("".join(to_unicode(c) if m else c for m, c in split_math(text)))


def tidy(s: str) -> str:
    """Spaces LaTeXML leaves inside citation brackets: '[ 12 , 14 ]' -> '[12, 14]'."""
    t = re.sub(r"\[\s+([^\]]*?)\s+\]", r"[\1]", s)
    t = re.sub(r"\(\s+([^)]*?)\s+\)", r"(\1)", t)
    t = re.sub(r"\s+([,;:])", r"\1", t)
    t = re.sub(r"\s+\.(?=\s|$)", ".", t)
    return t
