"""Minimal OpenAI-compatible chat-completions client (works with most local API gateways)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .http import DiskCache

log = logging.getLogger(__name__)


class LLM(Protocol):
    model: str

    def complete_json(self, system: str, user: str) -> dict: ...


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    timeout: float = 180.0
    max_tokens: int = 4096

    @classmethod
    def from_env(cls) -> LLMConfig | None:
        base = os.environ.get("LINEAGE_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        model = os.environ.get("LINEAGE_LLM_MODEL")
        key = os.environ.get("LINEAGE_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "none"
        if not base or not model:
            return None
        return cls(base_url=base, api_key=key, model=model)


def extract_json(text: str) -> dict:
    """Parse a JSON object from a model reply (tolerates code fences / prose around it)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else {"results": v}
    except json.JSONDecodeError:
        pass
    a, b = t.find("{"), t.rfind("}")
    if a >= 0 and b > a:
        v = json.loads(t[a:b + 1])
        return v if isinstance(v, dict) else {"results": v}
    raise ValueError(f"no JSON object in reply: {text[:200]!r}")


class OpenAICompatLLM:
    def __init__(self, cfg: LLMConfig, cache: DiskCache | None = None):
        self.cfg = cfg
        self.model = cfg.model
        self.cache = cache
        url = cfg.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        self.url = url
        self._json_mode = True
        self.client = httpx.Client(timeout=cfg.timeout, trust_env=True)
        self.calls = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def complete_json(self, system: str, user: str) -> dict:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ck = hashlib.sha256(json.dumps([self.model, messages]).encode()).hexdigest()
        if self.cache:
            hit = self.cache.get("llm", ck)
            if hit is not None:
                return hit
        last: Exception | None = None
        for attempt in range(4):
            body: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens,
            }
            if self._json_mode:
                body["response_format"] = {"type": "json_object"}
            try:
                r = self.client.post(self.url, json=body, headers={"Authorization": f"Bearer {self.cfg.api_key}"})
            except httpx.HTTPError as e:
                last = e
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 400 and self._json_mode and "response_format" in r.text:
                self._json_mode = False
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(2 ** (attempt + 1))
                continue
            if r.status_code != 200:
                raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:500]}")
            data = r.json()
            self.calls += 1
            for k in self.usage:
                self.usage[k] += (data.get("usage") or {}).get(k, 0) or 0
            content = data["choices"][0]["message"].get("content") or ""
            try:
                out = extract_json(content)
            except (ValueError, json.JSONDecodeError) as e:
                last = e
                continue
            if self.cache:
                self.cache.set("llm", ck, out)
            return out
        raise RuntimeError(f"LLM call failed: {last}")


class ScriptedLLM:
    """Test double: returns canned replies, chosen by a substring of the user prompt."""

    def __init__(self, replies: dict[str, dict] | None = None, default: dict | None = None):
        self.model = "scripted"
        self.replies = replies or {}
        self.default = default or {"results": []}
        self.prompts: list[str] = []

    def complete_json(self, system: str, user: str) -> dict:
        self.prompts.append(user)
        for needle, reply in self.replies.items():
            if needle in user:
                return reply
        return self.default
