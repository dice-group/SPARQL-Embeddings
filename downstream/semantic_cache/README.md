# Downstream use case: results-aware semantic caching

This folder shows, end to end, how a results-aware SPARQL encoder trained on
SQuARE can sit in front of a real SPARQL endpoint and decide when a cached
answer can be reused for a new query from a query log.

> **Status:** the code is tested offline with stub encoders and endpoints
> (`pytest tests/test_semantic_cache.py`). It has **not been run against the live
> endpoints yet**, so this README reports no numbers. The commands below are
> the ones to run.

## How it works

```text
incoming query ──► encoder (query text only) ──► top-1 cosine vs. cached queries
                                                      │
                          similarity ≥ θ ─────────────┼──────────── similarity < θ
                                │                                        │
                   reuse the cached answer                    execute on the endpoint,
                   (shadow mode: also execute                 add (query, answer) to the
                    and measure J(live, reused))              FIFO cache (K = 128)
```

This is the decision rule of the paper (Section 5.3): top-1 cosine ≥ θ, FIFO
cache of size K = 128. Answers are compared with the same canonical rows as
the SQuARE labels, so J means the same thing as in the paper.

## Why the default is shadow mode

The paper finds that embedding similarity alone is not a safe cache rule: at
θ = 0.8, fine-tuning lowers the false hit rate on DBpedia from 0.74 to 0.05,
but on Wikidata up to 46% of the queries still receive an unrelated answer.
So the default `--mode shadow` executes every query anyway and checks each
reuse decision against the live answer. `--mode serve` skips execution on a
hit and is only for endpoints where shadow runs show reliable reuse.

## Endpoints

| Name (`endpoints.yaml`) | Endpoint | Tier | What it shows |
|---|---|---|---|
| `dbpedia-dice` | https://dbpedia.data.dice-research.org/sparql | labels match | DBpedia 2022-12 mirror used to build SQuARE; live J should agree with the stored labels |
| `wikidata-dice` | https://wikidata.data.dice-research.org/sparql | labels match | Wikidata mirror used to build SQuARE |
| `dbpedia-public` | https://dbpedia.org/sparql | live | current DBpedia; stored labels do not apply, J is measured live |
| `wikidata-qlever` | https://qlever.dev/api/wikidata (UI: https://qlever.dev/wikidata) | live | current Wikidata on QLever |

The two public services are shared, so the defaults are polite: 1 request per
second, a 60 s timeout, a named User-Agent and 50 queries per run. Results
that hit a known row cap are marked `truncated` and are neither cached nor
scored. The row caps and the QLever API URL in `endpoints.yaml` must be
confirmed before reporting numbers.

## Run it

```bash
pip install -e . && pip install requests pyyaml huggingface_hub

# 1. A fine-tuned encoder: a local checkpoint from the training pipeline ...
python scripts/run_experiment.py --models e5-base-v2 --datasets dbpedia --seeds 42
#    ... or a Hugging Face model id with the same files (result_aware_model_config.json).

# 2. Optional: the validation-selected threshold instead of θ = 0.8
python scripts/analysis/calibrate_thresholds.py --dataset dbpedia --models e5-base-v2 --seeds 42

# 3. Replay the SQuARE test queries (never training queries)
python downstream/semantic_cache/replay.py --endpoint dbpedia-dice \
    --encoder results/raw/dbpedia/e5-base-v2/tuned/seed_42/best_model --sample 0
python downstream/semantic_cache/replay.py --endpoint dbpedia-public \
    --encoder results/raw/dbpedia/e5-base-v2/tuned/seed_42/best_model
python downstream/semantic_cache/replay.py --endpoint wikidata-qlever \
    --encoder results/raw/wikidata/e5-base-v2/tuned/seed_42/best_model

# 4. Or replay your own query log (JSON list of queries, or blocks separated by a blank line)
python downstream/semantic_cache/replay.py --endpoint dbpedia-public \
    --encoder <checkpoint> --log my_log.json
```

Each run writes `results/semantic_cache/<endpoint>_<mode>_seed<seed>/`:

- `decisions.jsonl`: per query, hit or miss, similarity, the reused query,
  live status, J(live, reused) and live execution time.
- `summary.json`: SHR, hit accuracy, CHR, FHR (as in the paper), mean J of
  hits, and the live time of correct hits (`saved_s_correct_hits`), which is
  what serve mode would save without serving a wrong answer.

Exact repeats are removed from the stream, since a plain query cache already
handles them. The stream order is fixed by `--seed`.

## Plan

1. **Check the setup.** Confirm the QLever API URL and the row caps of
   dbpedia.org (Virtuoso) and QLever with a few known queries.
2. **Labels match tier.** Run `dbpedia-dice` and `wikidata-dice` on all test
   queries. Live J should agree with the stored SQuARE labels; a mismatch
   points to an endpoint or snapshot change.
3. **Live tier.** Run `dbpedia-public` and `wikidata-qlever` in shadow mode
   on samples of the test queries, five stream orders each. Report the same
   metrics as Table 4 of the paper, plus the time saved by correct hits.
4. **Threshold.** Compare θ = 0.8 with the validation-selected θ from
   `calibrate_thresholds.py`.
5. **Own logs.** Replay recent logs from the DICE endpoints (`--log`) to show
   the cache on real traffic beyond LSQ.
6. **Share.** Publish the fine-tuned encoders on Hugging Face so the demo runs
   with `--encoder <hf-id>` and no training.

## Files

```text
downstream/semantic_cache/
├── semantic_cache.py   # endpoint client, encoder loader, FIFO cache, replay loop, metrics
├── replay.py           # command-line replay over a query log
├── endpoints.yaml      # endpoint URLs, tiers, row caps, rate limits
└── README.md
tests/test_semantic_cache.py   # offline tests (stub encoder and endpoint)
```
