"""Results-aware semantic caching in front of a live SPARQL endpoint.

Downstream use case of SQuARE: an incoming query is embedded with a
results-aware encoder, compared with the queries in a small FIFO cache, and
the cached answer of the most similar query is reused when the cosine
similarity is at least theta. This is the decision rule of the paper (RQ2):
top-1 cosine >= theta, FIFO cache of size K.

The paper shows that this rule alone is not a safe cache: on Wikidata up to
46% of the queries receive an unrelated answer. The default mode is therefore
"shadow": on every hit the query is still executed, and the Jaccard overlap
between the live answer and the reused answer is logged. "serve" mode returns
the cached answer without executing, and is meant only for endpoints where the
shadow results show that reuse is reliable.

Torch and transformers are imported only when a neural encoder is loaded, so
the cache, the endpoint client and the tests run without them.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Answer canonicalization
# Same logic as datasets/scripts/build_rq3_from_endpoint.py, which built the
# SQuARE labels, so J here means the same as J in the paper.
# ---------------------------------------------------------------------------


def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip())


def query_hash(q: str) -> str:
    return hashlib.sha256(normalize_query(q).encode("utf-8")).hexdigest()


def canonical_cell(binding: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(binding.get("type", "")),
        str(binding.get("value", "")),
        str(binding.get("xml:lang", binding.get("lang", ""))),
        str(binding.get("datatype", "")),
    )


def canonical_row(binding: dict[str, Any], vars_: Sequence[str]) -> str:
    row = [
        canonical_cell(binding[v]) if v in binding else ("", "", "", "")
        for v in vars_
    ]
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"))


def answer_set(payload: dict[str, Any]) -> Optional[frozenset[str]]:
    """Canonical answer set of a SPARQL JSON result, or None for non-SELECT."""
    bindings = payload.get("results", {}).get("bindings")
    if bindings is None:
        return None
    vars_ = payload.get("head", {}).get("vars", [])
    return frozenset(canonical_row(b, vars_) for b in bindings)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Endpoint client
# ---------------------------------------------------------------------------


@dataclass
class ExecResult:
    status: str  # success | empty | truncated | error | unsupported
    answers: frozenset[str] = frozenset()
    elapsed_s: float = 0.0
    error: Optional[str] = None


@dataclass
class Endpoint:
    """A SPARQL endpoint with polite defaults for public services."""

    name: str
    url: str
    max_rows: Optional[int] = None  # known result cap; a full page is marked truncated
    timeout_s: float = 60.0
    min_interval_s: float = 1.0  # rate limit between requests
    retries: int = 1
    user_agent: str = (
        "SQuARE-semantic-cache-demo/0.1 "
        "(+https://github.com/dice-group/SPARQL-Embeddings)"
    )
    _last_call: float = field(default=0.0, repr=False)

    def _wait(self) -> None:
        delay = self.min_interval_s - (time.monotonic() - self._last_call)
        if delay > 0:
            time.sleep(delay)
        self._last_call = time.monotonic()

    def execute(self, query: str) -> ExecResult:
        import requests  # lazy, so tests can run with a stub endpoint

        headers = {
            "Accept": "application/sparql-results+json",
            "User-Agent": self.user_agent,
        }
        params = {"query": query}
        error = None
        for attempt in range(self.retries + 1):
            self._wait()
            started = time.monotonic()
            try:
                resp = requests.get(self.url, params=params, headers=headers,
                                    timeout=self.timeout_s)
                if resp.status_code in (414, 431):  # URL too long
                    resp = requests.post(self.url, data=params, headers=headers,
                                         timeout=self.timeout_s)
                resp.raise_for_status()
                elapsed = time.monotonic() - started
                answers = answer_set(resp.json())
                if answers is None:
                    return ExecResult("unsupported", elapsed_s=elapsed,
                                      error="no results.bindings (not a SELECT result)")
                # Virtuoso sends this header when it cuts a result at its row cap.
                # Confirm the exact signal of each endpoint before trusting J.
                capped = "X-SPARQL-MaxRows" in resp.headers
                if capped or (self.max_rows is not None and len(answers) >= self.max_rows):
                    return ExecResult("truncated", answers, elapsed)
                return ExecResult("success" if answers else "empty", answers, elapsed)
            except Exception as exc:  # network, HTTP or JSON error
                error = f"{type(exc).__name__}: {exc}"
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 8))
        return ExecResult("error", error=error)


class Executor(Protocol):
    def execute(self, query: str) -> ExecResult: ...


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

Encoder = Callable[[Sequence[str]], np.ndarray]  # texts -> L2-normalized rows


def load_encoder(source: str, device: str = "cpu") -> Encoder:
    """Load a fine-tuned results-aware encoder.

    `source` is a local checkpoint folder written by the training pipeline
    (results/raw/<graph>/<model>/tuned/seed_<seed>/best_model) or a Hugging
    Face model id holding the same files. The folder must contain
    result_aware_model_config.json, which stores pooling, prefix and max
    length. There is deliberately no zero-shot fallback: zero-shot encoders
    reuse almost every query at theta = 0.8 (paper, Table 4).
    """
    path = Path(source)
    if not path.exists():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(source))
    cfg_file = path / "result_aware_model_config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(
            f"{cfg_file} not found. Pass a fine-tuned checkpoint, not a base model.")
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))

    from result_aware.model import HFEmbeddingModel  # imports torch (lazy)

    model = HFEmbeddingModel(cfg, device=device, checkpoint_path=path)
    model.eval()

    def encode(texts: Sequence[str]) -> np.ndarray:
        emb = model.encode(list(texts), batch_size=16, show_progress=False)
        emb = emb.float().numpy()
        return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)

    return encode


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@dataclass
class Entry:
    query: str
    embedding: np.ndarray
    answers: frozenset[str]


class SemanticCache:
    """FIFO cache with a top-1 cosine reuse rule (paper, Section 5.3)."""

    def __init__(self, capacity: int = 128, theta: float = 0.8) -> None:
        self.capacity = capacity
        self.theta = theta
        self.entries: deque[Entry] = deque()

    def lookup(self, embedding: np.ndarray) -> tuple[Optional[Entry], float]:
        if not self.entries:
            return None, float("-inf")
        mat = np.stack([e.embedding for e in self.entries])
        sims = mat @ embedding
        best = int(np.argmax(sims))
        return self.entries[best], float(sims[best])

    def add(self, query: str, embedding: np.ndarray, answers: frozenset[str]) -> None:
        if len(self.entries) >= self.capacity:
            self.entries.popleft()
        self.entries.append(Entry(query, embedding, answers))


@dataclass
class Decision:
    query: str
    hit: bool
    similarity: Optional[float]
    cached_query: Optional[str]
    status: str  # status of the live execution, or "served" in serve mode
    jaccard: Optional[float]  # live vs reused answer; None if not measured
    live_s: Optional[float]  # time of the live execution


def run_stream(
    queries: Sequence[str],
    encoder: Encoder,
    executor: Executor,
    cache: SemanticCache,
    mode: str = "shadow",
    on_decision: Optional[Callable[[Decision], None]] = None,
) -> list[Decision]:
    """Replay a query stream through the cache.

    shadow: every query is executed; hits are scored by J(live, reused).
    serve:  hits return the cached answer without executing (not scored).
    Queries whose live result is an error, truncated or unsupported are never
    cached and are excluded from scoring.
    """
    if mode not in ("shadow", "serve"):
        raise ValueError("mode must be 'shadow' or 'serve'")
    decisions = []
    for q in queries:
        emb = encoder([q])[0]
        entry, sim = cache.lookup(emb)
        hit = entry is not None and sim >= cache.theta
        if hit and mode == "serve":
            d = Decision(q, True, sim, entry.query, "served", None, None)
        else:
            res = executor.execute(q)
            j = None
            if hit and res.status in ("success", "empty"):
                j = jaccard(res.answers, entry.answers)
            d = Decision(q, hit, sim if entry else None, entry.query if hit else None,
                         res.status, j, res.elapsed_s if res.status != "error" else None)
            # The cache only learns from complete live answers.
            if not hit and res.status in ("success", "empty"):
                cache.add(q, emb, res.answers)
        decisions.append(d)
        if on_decision:
            on_decision(d)
    return decisions


def summarize(decisions: Sequence[Decision], pos: float = 0.8, neg: float = 0.1) -> dict:
    """Paper metrics over scorable queries: SHR, Acc., CHR, FHR, mean J.

    A query is scorable when its live execution returned a complete answer.
    saved_s is the live time of correct hits, i.e. what serve mode would save
    on those queries without serving a wrong answer.
    """
    scorable = [d for d in decisions if d.status in ("success", "empty")]
    hits = [d for d in scorable if d.hit]
    correct = [d for d in hits if d.jaccard is not None and d.jaccard >= pos]
    wrong = [d for d in hits if d.jaccard is not None and d.jaccard <= neg]
    n = len(scorable)
    return {
        "queries": len(decisions),
        "scorable": n,
        "skipped": {s: sum(d.status == s for d in decisions)
                    for s in ("truncated", "error", "unsupported", "served")},
        "SHR": len(hits) / n if n else None,
        "Acc": len(correct) / len(hits) if hits else None,
        "CHR": len(correct) / n if n else None,
        "FHR": len(wrong) / n if n else None,
        "mean_J_hits": float(np.mean([d.jaccard for d in hits])) if hits else None,
        "saved_s_correct_hits": float(sum(d.live_s or 0.0 for d in correct)),
        "total_live_s": float(sum(d.live_s or 0.0 for d in scorable)),
    }


def decision_to_json(d: Decision) -> str:
    return json.dumps(asdict(d), ensure_ascii=False)
