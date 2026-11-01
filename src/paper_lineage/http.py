"""HTTP client with on-disk caching, per-host rate limiting, and retry/backoff."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)

USER_AGENT = "paper-lineage/0.1 (https://github.com/; research tool)"


class DiskCache:
    """Tiny content-addressed JSON cache."""

    def __init__(self, root: str | Path | None):
        self.root = Path(root).expanduser() if root else None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, key: str) -> Path | None:
        if not self.root:
            return None
        h = hashlib.sha256(key.encode()).hexdigest()
        return self.root / namespace / h[:2] / f"{h}.json"

    def get(self, namespace: str, key: str, max_age: float | None = None) -> Any | None:
        p = self._path(namespace, key)
        if not p or not p.exists():
            return None
        if max_age is not None and time.time() - p.stat().st_mtime > max_age:
            return None
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, namespace: str, key: str, value: Any) -> None:
        p = self._path(namespace, key)
        if not p:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value))
        tmp.replace(p)


class HttpError(RuntimeError):
    def __init__(self, status: int, url: str, body: str = ""):
        super().__init__(f"HTTP {status} for {url}: {body[:200]}")
        self.status = status
        self.url = url


class Http:
    """GET/POST with cache, per-host minimum interval, and retries on 429/5xx/network errors."""

    def __init__(
        self,
        cache: DiskCache,
        min_interval: dict[str, float] | None = None,
        max_retries: int = 6,
        timeout: float = 60.0,
        ttl: float = 30 * 86400,
    ):
        self.cache = cache
        self.min_interval = min_interval or {}
        self.max_retries = max_retries
        self.ttl = ttl
        verify: Any = True
        for env in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            if os.environ.get(env) and Path(os.environ[env]).exists():
                verify = os.environ[env]
                break
        self.client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            verify=verify,
            trust_env=True,
        )
        self._last: dict[str, float] = {}

    def _wait(self, host: str) -> None:
        gap = self.min_interval.get(host, 0.0)
        if gap <= 0:
            return
        elapsed = time.time() - self._last.get(host, 0.0)
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last[host] = time.time()

    def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
        as_json: bool = True,
        cache: bool = True,
        ttl: float | None = None,
        ok_statuses: tuple[int, ...] = (200,),
    ) -> Any:
        """Returns parsed JSON (as_json) or text. Non-ok statuses raise HttpError (404 included)."""
        cache_key = json.dumps([method, url, params, json_body], sort_keys=True)
        ns = "http"
        if cache:
            hit = self.cache.get(ns, cache_key, max_age=ttl if ttl is not None else self.ttl)
            if hit is not None:
                if "error" in hit:
                    raise HttpError(hit["error"], url)
                return hit["body"]
        host = urlparse(url).netloc
        delay = 2.0
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._wait(host)
            try:
                r = self.client.request(method, url, params=params, json=json_body, headers=headers)
            except httpx.HTTPError as e:  # network-level
                last_exc = e
                log.debug("network error %s on %s (attempt %d)", e, url, attempt)
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if r.status_code in ok_statuses:
                body = r.json() if as_json else r.text
                if cache:
                    self.cache.set(ns, cache_key, {"body": body})
                return body
            if r.status_code == 429 or r.status_code >= 500:
                retry_after = r.headers.get("retry-after")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
                log.debug("HTTP %s on %s, retrying in %.1fs", r.status_code, url, wait)
                time.sleep(wait)
                delay = min(delay * 2, 60)
                last_exc = HttpError(r.status_code, url, r.text)
                continue
            if r.status_code == 404 and cache:
                self.cache.set(ns, cache_key, {"error": 404})
            raise HttpError(r.status_code, url, r.text)
        raise last_exc or RuntimeError(f"request failed: {url}")

    def get(self, url: str, **kw: Any) -> Any:
        return self.request("GET", url, **kw)

    def post(self, url: str, **kw: Any) -> Any:
        return self.request("POST", url, **kw)
