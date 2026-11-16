#!/usr/bin/env python3
r"""Replay a SPARQL query log through a results-aware semantic cache.

Examples (run from the repository root):

  # SQuARE test queries against the DBpedia mirror SQuARE was built on
  python downstream/semantic_cache/replay.py --endpoint dbpedia-dice \
      --encoder results/raw/dbpedia/e5-base-v2/tuned/seed_42/best_model

  # live Wikidata on QLever, 100 test queries, encoder from Hugging Face
  python downstream/semantic_cache/replay.py --endpoint wikidata-qlever \
      --encoder <hf-user>/<squared-wikidata-encoder> --sample 100

  # your own log: one query per blank-line-separated block, or a JSON list
  python downstream/semantic_cache/replay.py --endpoint dbpedia-public \
      --encoder <checkpoint> --log my_queries.txt

Outputs in --out: decisions.jsonl (one line per query) and summary.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from semantic_cache import (  # noqa: E402
    Endpoint,
    SemanticCache,
    decision_to_json,
    load_encoder,
    normalize_query,
    run_stream,
    summarize,
)


def test_queries(graph: str) -> list[str]:
    """Unique queries of the SQuARE test split (never training queries)."""
    path = ROOT / "datasets" / f"{graph}-splits" / f"{graph}-test.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    seen: dict[str, str] = {}

    def add(q):
        if isinstance(q, dict):
            q = q.get("query")
        if isinstance(q, str) and q.strip():
            seen.setdefault(normalize_query(q), q.strip())

    for item in data:
        add(item.get("anchor_query"))
        for key in ("positive_queries", "negative_queries"):
            for c in item.get(key, []) or []:
                add(c)
    return [seen[k] for k in sorted(seen)]


def log_queries(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        items = json.loads(text)
    else:
        items = [b for b in text.split("\n\n")]
    out, seen = [], set()
    for q in items:
        q = q.get("query", "") if isinstance(q, dict) else q
        n = normalize_query(q)
        if n and n not in seen:  # exact repeats are a plain cache's job
            seen.add(n)
            out.append(q.strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True,
                    help="name in endpoints.yaml, or a SPARQL endpoint URL with --graph")
    ap.add_argument("--graph", choices=["dbpedia", "wikidata"],
                    help="needed when --endpoint is a URL")
    ap.add_argument("--encoder", required=True,
                    help="fine-tuned checkpoint folder or Hugging Face model id")
    ap.add_argument("--theta", type=float, default=0.8,
                    help="reuse threshold (paper: 0.8; better: the validation-selected "
                         "value from scripts/analysis/calibrate_thresholds.py)")
    ap.add_argument("--capacity", type=int, default=128)
    ap.add_argument("--mode", choices=["shadow", "serve"], default="shadow",
                    help="shadow executes every query and scores hits (default)")
    ap.add_argument("--log", type=Path, help="query log; default: SQuARE test split")
    ap.add_argument("--sample", type=int, default=50,
                    help="number of queries to replay (default 50; 0 = all)")
    ap.add_argument("--seed", type=int, default=100, help="stream order seed")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("results/semantic_cache"))
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "endpoints.yaml").read_text(encoding="utf-8"))
    if args.endpoint in cfg:
        e = cfg[args.endpoint]
        graph = e["graph"]
        endpoint = Endpoint(args.endpoint, e["url"], max_rows=e.get("max_rows"),
                            min_interval_s=e.get("min_interval_s", 1.0))
        tier = e.get("tier", "live")
    else:
        if not args.graph:
            ap.error("--graph is required when --endpoint is a URL")
        graph, tier = args.graph, "live"
        endpoint = Endpoint("custom", args.endpoint)
    if args.mode == "serve":
        print("warning: serve mode returns cached answers without checking them; "
              "the paper finds up to 46% unrelated hits on Wikidata at theta=0.8.",
              file=sys.stderr)

    queries = log_queries(args.log) if args.log else test_queries(graph)
    random.Random(args.seed).shuffle(queries)
    if args.sample:
        queries = queries[: args.sample]

    encoder = load_encoder(args.encoder, device=args.device)
    cache = SemanticCache(capacity=args.capacity, theta=args.theta)

    run = args.out / f"{args.endpoint if args.endpoint in cfg else 'custom'}_{args.mode}_seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    with (run / "decisions.jsonl").open("w", encoding="utf-8") as f:
        def log(d):
            f.write(decision_to_json(d) + "\n")
            f.flush()
            mark = "HIT " if d.hit else "miss"
            j = f"J={d.jaccard:.2f}" if d.jaccard is not None else ""
            print(f"{mark} {d.status:<10} {j}", file=sys.stderr)

        decisions = run_stream(queries, encoder, endpoint, cache, args.mode, log)

    summary = summarize(decisions)
    summary.update(endpoint=endpoint.url, tier=tier, graph=graph, encoder=args.encoder,
                   theta=args.theta, capacity=args.capacity, mode=args.mode,
                   seed=args.seed, log=str(args.log) if args.log else f"SQuARE {graph} test")
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
