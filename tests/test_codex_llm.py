"""The ChatGPT-subscription backend (Codex CLI), tested against a fake `codex` executable."""

import json
import stat
import sys

import pytest

from paper_lineage.http import DiskCache
from paper_lineage.llm import CodexCLILLM, OpenAICompatLLM, make_llm

FAKE = r'''#!PYTHON
import json, os, sys
args = sys.argv[1:]
prompt = sys.stdin.read()
log = os.environ["FAKE_CODEX_LOG"]
with open(log, "a") as fh:
    fh.write(json.dumps({"args": args, "prompt": prompt, "cwd_files": os.listdir(args[args.index("-C") + 1])}) + "\n")
if os.environ.get("FAKE_CODEX_MODE") == "logged_out":
    sys.stderr.write("ERROR: unexpected status 401 Unauthorized: Missing bearer authentication\n")
    sys.exit(1)
with open(args[args.index("-o") + 1], "w") as fh:
    fh.write('Here you go:\n```json\n{"results": [{"id": "P1", "intro": "ok"}]}\n```')
'''


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    exe = tmp_path / "codex"
    exe.write_text(FAKE.replace("PYTHON", sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_CODEX_LOG", str(log))
    return str(exe), log


def _calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_codex_backend_runs_exec_read_only_and_parses_the_final_message(fake_codex, tmp_path):
    exe, log = fake_codex
    llm = CodexCLILLM(binary=exe, model="gpt-x", effort="low", cache=DiskCache(str(tmp_path / "cache")))
    out = llm.complete_json("SYSTEM RULES", "USER PROMPT")
    assert out == {"results": [{"id": "P1", "intro": "ok"}]}           # fences/prose around JSON tolerated
    (call,) = _calls(log)
    a = call["args"]
    assert a[0] == "exec" and a[-1] == "-"                              # prompt via stdin
    assert a[a.index("--sandbox") + 1] == "read-only" and "--skip-git-repo-check" in a
    assert a[a.index("-m") + 1] == "gpt-x" and 'model_reasoning_effort="low"' in a
    assert call["cwd_files"] == []                                     # runs in an empty folder
    assert call["prompt"].startswith("SYSTEM RULES") and call["prompt"].rstrip().endswith("USER PROMPT")
    assert llm.calls == 1
    assert llm.complete_json("SYSTEM RULES", "USER PROMPT") == out      # cached: no second run
    assert len(_calls(log)) == 1


def test_codex_backend_reports_a_missing_login(fake_codex, monkeypatch):
    exe, log = fake_codex
    monkeypatch.setenv("FAKE_CODEX_MODE", "logged_out")
    with pytest.raises(RuntimeError, match="codex login"):
        CodexCLILLM(binary=exe).complete_json("s", "u")
    assert len(_calls(log)) == 1                                       # no pointless retries


def test_codex_backend_without_the_cli_says_how_to_install():
    with pytest.raises(RuntimeError, match="npm i -g @openai/codex"):
        CodexCLILLM(binary="/nonexistent/codex").complete_json("s", "u")


def test_make_llm_picks_the_provider(monkeypatch):
    monkeypatch.delenv("LINEAGE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("LINEAGE_LLM_MODEL", raising=False)
    assert make_llm("api") is None                                     # nothing configured
    assert isinstance(make_llm("codex"), CodexCLILLM)
    monkeypatch.setenv("LINEAGE_LLM_PROVIDER", "codex")
    assert isinstance(make_llm(), CodexCLILLM)
    monkeypatch.setenv("LINEAGE_LLM_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("LINEAGE_LLM_MODEL", "m")
    assert isinstance(make_llm("api"), OpenAICompatLLM)
    with pytest.raises(ValueError):
        make_llm("bard")
