# Syntax-supervised control

Tests whether the retrieval gain comes from the **answer-overlap labels** or just from
fine-tuning on SPARQL. The same encoders are trained with the same protocol; only the labels
of the training split differ.

| | Answer labels (main runs) | Syntax labels (control) |
|---|---|---|
| Positives per anchor | stored positives, J >= 0.8 | the same number, highest algebra similarity |
| Negatives per anchor | 20, J <= 0.1, nonzero-overlap first | 20 from the low-similarity zone, the same number of "close" ones first |
| Splits, anchors, loss, mining, hyperparameters | `configs/experiment.yaml` | identical (see "Differences" below) |
| Checkpoint selection, evaluation | validation/test with answer overlap | identical (valid/test files unchanged) |

Algebra similarity is the `algebra_similarity` baseline of `scripts/baselines/nonneural.py`
(multiset Jaccard over algebra nodes, parent>child edges, variable-abstracted triple patterns
and RDF terms). The low-similarity zone is every training query with similarity <= 0.5 x the
anchor's weakest syntax positive (`--neg-ratio`). Queries rdflib cannot parse (0 on DBpedia,
7 on Wikidata) fall back to the TF-IDF token bag.

## Setup

- **Encoders:** E5-base-v2, Qwen3-Embedding and CodeBERT (a text embedder, an LLM-based
  embedder and a code model).
- **Seeds:** 42, 43 and 44.
- **Training and checkpoint selection** exactly as in the main runs (`configs/experiment.yaml`):
  20 epochs, validation after every epoch, checkpoint with the highest validation nDCG@10.

That is 9 runs per graph, 18 in total. The answer-label side reuses the checkpoints of the
main runs (same encoders, same seeds), so nothing is retrained.

## Run

From the repository root, on the GPU machine that holds the main-run checkpoints in `results/`:

```bash
pip install -r scripts/baselines/requirements.txt   # rdflib, numpy, networkx

# 1+2+3. DBpedia on GPU 0 (labels -> training -> evaluation)
CUDA_VISIBLE_DEVICES=0 GRAPHS=dbpedia bash scripts/syntax_control/run_syntax_control.sh

# Wikidata on GPU 1, in parallel
CUDA_VISIBLE_DEVICES=1 GRAPHS=wikidata bash scripts/syntax_control/run_syntax_control.sh

# Final evaluation over both graphs (once both are trained)
STEP=eval bash scripts/syntax_control/run_syntax_control.sh
```

Defaults: `MODELS="e5-base-v2 qwen3-embedding codebert"`, `SEEDS="42 43 44"`, `GRAPHS="dbpedia wikidata"`.
Any of them, plus `EPOCHS` and `EVAL_EVERY`, can be set in front of the command.
Steps: `STEP=check | labels | train | eval | all`. Training is resumable: rerun the same
command after an interruption.

What each step does:

1. **Labels** (about 2 min, CPU): `build_syntax_labels.py` writes
   `datasets/<graph>-splits/<graph>-train-syntax.json` and `label_stats_<graph>.json`.
   The output is deterministic (same file for any `PYTHONHASHSEED`); existing files are reused.
2. **Training**: `scripts/run_experiment.py` with `configs/experiment_syntax_control.yaml`
   writes checkpoints to `results_syntax_control/raw/<graph>/<model>/tuned/seed_<s>/`.
   With `EPOCHS`/`EVAL_EVERY` set, a copy `configs/experiment_syntax_control.run.yaml` is used.
3. **Evaluation**: `eval_syntax_control.py` loads both checkpoint sets for the same encoders and
   seeds, ranks all test queries, and computes stored-label (Table 2) and complete-label
   (Table 3) metrics with paired bootstrap intervals (answer - syntax). Outputs:
   - `control_results.json`: all numbers, per encoder and averaged
   - `control_latex.txt`: table rows and the numbers for the paper paragraph

## Differences from the main runs

`configs/experiment_syntax_control.yaml` differs from `configs/experiment.yaml` only in the
training files (`*-train-syntax.json`), the output folder (`results_syntax_control/`), the
seeds (42-44) and the disabled result-caching evaluation. Batch size, explicit negatives,
learning rate, schedule, `max_length`, hard-negative mining, validation frequency and
checkpoint selection are identical, so the control is matched to the main runs. It covers
3 encoders and 3 seeds instead of 10 and 5; the answer-label side of the comparison uses the
same 3 encoders and seeds.

## Label statistics (already computed)

| | DBpedia | Wikidata |
|---|---|---|
| Training anchors / queries | 3,737 / 3,781 | 6,734 / 7,743 |
| Syntax positives / negatives | 31,831 / 74,740 | 29,867 / 134,680 |
| Median similarity of weakest positive | 0.73 | 0.73 |
| Median similarity of strongest negative | 0.33 | 0.33 |
| Share of an anchor's syntax positives that are answer positives (mean) | 13% | 21% |
| Syntax negatives that are answer positives | 0.01% | 0.03% |

## Results

`experiment_results/syntax_control/control_results.json` and `control_latex.txt` hold the
numbers reported in the paper (Table 3 and Section 4.3).
