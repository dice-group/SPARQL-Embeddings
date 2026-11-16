"""Offline tests for downstream/semantic_cache (no models, no network)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "downstream" / "semantic_cache"))

from semantic_cache import (  # noqa: E402
    ExecResult,
    SemanticCache,
    answer_set,
    canonical_row,
    jaccard,
    run_stream,
    summarize,
)


def payload(rows, vars_=("x",)):
    return {"head": {"vars": list(vars_)},
            "results": {"bindings": [{v: {"type": "uri", "value": r} for v in vars_}
                                     for r in rows]}}


class StubEndpoint:
    def __init__(self, answers, status=None):
        self.answers, self.status, self.calls = answers, status or {}, []

    def execute(self, q):
        self.calls.append(q)
        st = self.status.get(q, "success")
        return ExecResult(st, answer_set(payload(self.answers[q])), 0.5)


def stub_encoder(vectors):
    def enc(texts):
        v = np.array([vectors[t] for t in texts], dtype=float)
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    return enc


def test_canonical_row_matches_label_builder():
    path = ROOT / "datasets" / "scripts" / "build_rq3_from_endpoint.py"
    pytest.importorskip("requests")
    pytest.importorskip("tqdm")
    spec = importlib.util.spec_from_file_location("builder", path)
    builder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = builder  # needed for its dataclasses
    spec.loader.exec_module(builder)
    b = {"x": {"type": "literal", "value": "Paris", "xml:lang": "en"},
         "y": {"type": "literal", "value": "1", "datatype": "xsd:int"}}
    for vars_ in (["x", "y"], ["y", "x", "z"]):
        assert canonical_row(b, vars_) == builder.canonical_row(b, vars_)


def test_jaccard():
    assert jaccard(frozenset("ab"), frozenset("bc")) == pytest.approx(1 / 3)
    assert jaccard(frozenset(), frozenset()) == 1.0


def test_shadow_mode_scores_hits_and_executes_everything():
    vec = {"a": [1, 0], "b": [1, 0.1], "c": [0, 1]}
    ans = {"a": ["1", "2"], "b": ["1", "2"], "c": ["9"]}
    ep = StubEndpoint(ans)
    ds = run_stream(["a", "b", "c"], stub_encoder(vec), ep,
                    SemanticCache(capacity=8, theta=0.8), mode="shadow")
    assert [d.hit for d in ds] == [False, True, False]
    assert ds[1].cached_query == "a" and ds[1].jaccard == 1.0
    assert ep.calls == ["a", "b", "c"]
    s = summarize(ds)
    assert s["SHR"] == pytest.approx(1 / 3) and s["CHR"] == pytest.approx(1 / 3)
    assert s["Acc"] == 1.0 and s["FHR"] == 0.0


def test_wrong_hit_counts_as_false_hit():
    vec = {"a": [1, 0], "b": [1, 0.05]}
    ep = StubEndpoint({"a": ["1"], "b": ["7"]})
    s = summarize(run_stream(["a", "b"], stub_encoder(vec), ep, SemanticCache(theta=0.8)))
    assert s["FHR"] == 0.5 and s["CHR"] == 0.0 and s["Acc"] == 0.0


def test_serve_mode_skips_execution_on_hit():
    vec = {"a": [1, 0], "b": [1, 0.1]}
    ep = StubEndpoint({"a": ["1"], "b": ["1"]})
    ds = run_stream(["a", "b"], stub_encoder(vec), ep, SemanticCache(theta=0.8), mode="serve")
    assert ep.calls == ["a"] and ds[1].status == "served"


def test_truncated_results_are_not_cached_or_scored():
    vec = {"a": [1, 0], "b": [1, 0.1]}
    ep = StubEndpoint({"a": ["1"], "b": ["1"]}, status={"a": "truncated"})
    ds = run_stream(["a", "b"], stub_encoder(vec), ep, SemanticCache(theta=0.8))
    assert not ds[1].hit  # "a" never entered the cache
    assert summarize(ds)["scorable"] == 1


def test_fifo_capacity():
    cache = SemanticCache(capacity=2)
    for i, q in enumerate("abc"):
        cache.add(q, np.eye(3)[i], frozenset())
    assert [e.query for e in cache.entries] == ["b", "c"]
