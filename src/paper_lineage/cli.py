"""`lineage` command-line interface."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console

from .arxiv import normalize_arxiv_id
from .classify import HeuristicClassifier, LLMClassifier
from .engine import Config, Engine
from .llm import LLMConfig, OpenAICompatLLM, make_llm
from .render import render_html, render_json, render_mermaid
from .render.tree import build_renderable
from .source import PaperSource

app = typer.Typer(add_completion=False, help=__doc__)
err = Console(stderr=True)

FORMATS = ("tree", "mermaid", "json", "html")
EXT = {"mermaid": "mmd", "json": "json", "html": "html"}


@app.command()
def main(
    paper: str = typer.Argument("", help="arXiv url or id, e.g. https://arxiv.org/abs/2507.07982"),
    max_depth: int = typer.Option(4, "--max-depth", help="Hard limit on generations traced back."),
    max_nodes: int = typer.Option(30, "--max-nodes", help="Max papers read (expanded), including the target."),
    parents_per_node: int = typer.Option(4, "--parents-per-node", help="Max lineage parents kept per paper."),
    alternatives: int = typer.Option(2, "--alternatives", help="Max 'done differently' branches per node."),
    min_relevance: float = typer.Option(0.86, "--min-relevance",
                                        help="Min SPECTER2 similarity to the target for ancestors."),
    spine: str = typer.Option("auto", "--spine", help="auto: highlight a main path only if one dominates; none."),
    descendants: int = typer.Option(0, "--descendants", help="Also show N papers that build on the target."),
    concurrent: int = typer.Option(0, "--concurrent", help="Also show N similar concurrent papers."),
    fmt: List[str] = typer.Option(["tree"], "--format", "-f", help="tree|mermaid|json|html (repeatable)."),
    output: Optional[Path] = typer.Option(None, "--output", "-o",
                                          help="Output path stem for file formats (default lineage-<id>)."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Heuristics only (no LLM calls)."),
    llm_provider: str = typer.Option(os.environ.get("LINEAGE_LLM_PROVIDER", "api"), "--llm",
                                     help="api: OpenAI-compatible endpoint (LINEAGE_LLM_* env); "
                                          "codex: your ChatGPT subscription via the Codex CLI (`codex login`)."),
    evidence: bool = typer.Option(False, "--evidence", help="Print evidence quotes in the tree view."),
    cache_dir: Path = typer.Option(Path(os.environ.get("LINEAGE_CACHE_DIR", "~/.cache/paper-lineage")),
                                   "--cache-dir"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log every keep/drop decision."),
    check_llm: bool = typer.Option(False, "--check-llm", help="Send one tiny request to the LLM gateway and exit."),
    full: bool = typer.Option(False, "--full", help="tree/mermaid: draw every kept paper, not just the main chains."),
    render: Optional[Path] = typer.Option(None, "--render",
                                          help="Re-render a saved `-f json` run (no network) instead of running."),
):
    """Trace the citation-grounded lineage of an arXiv paper."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if llm_provider.lower() not in ("api", "codex", "chatgpt"):
        raise typer.BadParameter("--llm must be api or codex")
    if check_llm and llm_provider.lower() != "api":
        llm = make_llm(llm_provider)
        try:
            reply = llm.complete_json("Reply with a JSON object only.", 'Return {"ok": true}')
        except Exception as e:
            err.print(f"[red]LLM check failed:[/red] {e}")
            raise typer.Exit(1) from None
        err.print(f"[green]LLM OK[/green] via Codex CLI (ChatGPT subscription), model "
                  f"{llm.model or 'Codex default'}: {reply}")
        raise typer.Exit(0)
    if check_llm:
        cfg = LLMConfig.from_env()
        if cfg is None:
            err.print("[red]LINEAGE_LLM_BASE_URL and LINEAGE_LLM_MODEL must be set.[/red]")
            raise typer.Exit(2)
        try:
            reply = OpenAICompatLLM(cfg).complete_json(
                "Reply with a JSON object only.", 'Return {"ok": true, "model": "<your model name>"}')
        except Exception as e:
            err.print(f"[red]LLM check failed:[/red] {e}")
            raise typer.Exit(1) from None
        err.print(f"[green]LLM OK[/green] at {cfg.base_url} with model {cfg.model}: {reply}")
        raise typer.Exit(0)
    if render is not None:
        import json as _json

        from .graph import LineageGraph
        graph = LineageGraph.from_dict(_json.loads(render.read_text(encoding="utf-8")))
        if spine == "none":
            graph.choose_spine("none")
        _emit(graph, fmt, output or render.with_suffix(""), evidence, full)
        return
    if not paper:
        raise typer.BadParameter("give an arXiv url or id (or use --check-llm / --render)")
    bad = [f for f in fmt if f not in FORMATS]
    if bad:
        raise typer.BadParameter(f"unknown format(s) {bad}; choose from {FORMATS}")
    if spine not in ("auto", "none"):
        raise typer.BadParameter("--spine must be auto or none")
    try:
        arxiv_id = normalize_arxiv_id(paper)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None

    source = PaperSource(str(cache_dir.expanduser()))
    if no_llm:
        classifier = HeuristicClassifier()
    else:
        llm = make_llm(llm_provider, cache=source.cache)
        if llm is None:
            err.print("[yellow]No LLM configured (set LINEAGE_LLM_BASE_URL and LINEAGE_LLM_MODEL, "
                      "optionally LINEAGE_LLM_API_KEY, or use --llm codex with a ChatGPT subscription). "
                      "Falling back to --no-llm heuristics.[/yellow]")
            classifier = HeuristicClassifier()
        else:
            classifier = LLMClassifier(llm)
    config = Config(max_depth=max_depth, max_nodes=max_nodes, parents_per_node=parents_per_node,
                    alternatives=alternatives, min_relevance=min_relevance, spine=spine,
                    descendants=descendants, concurrent=concurrent)
    with err.status("starting...") as status:
        engine = Engine(source, classifier, config, progress=lambda m: status.update(m))
        graph = engine.run(arxiv_id)
    llm = getattr(classifier, "llm", None)
    if llm is not None and getattr(llm, "calls", 0):
        graph.stats["llm_calls"] = llm.calls
        graph.stats["llm_tokens"] = dict(llm.usage)

    from .quality import check as quality_check
    issues = quality_check(graph)
    graph.stats["quality_issues"] = len(issues)
    if issues:
        err.print(f"[yellow]{len(issues)} output-quality issue(s); run with -v to list them[/yellow]")
    if verbose:
        for i in issues:
            err.print(f"[yellow]quality:[/yellow] {i}")
        for d in graph.log:
            err.print(f"[dim]{d}[/dim]")
    _emit(graph, fmt, output or Path(f"lineage-{arxiv_id.replace('/', '_')}"), evidence, full)


def _emit(graph, fmt: List[str], stem: Path, evidence: bool, full: bool) -> None:
    for f in fmt:
        if f == "tree":
            Console().print(build_renderable(graph, show_evidence=evidence, full=full))
            continue
        if f == "mermaid":
            text = render_mermaid(graph, full=full)
        else:
            text = {"json": render_json, "html": render_html}[f](graph)
        path = stem if stem.suffix == "." + EXT[f] else Path(f"{stem}.{EXT[f]}")
        path.write_text(text, encoding="utf-8")
        err.print(f"wrote {path}")


def run() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
