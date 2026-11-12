# Results-Aware SPARQL Embeddings (SQuARE)

Code and data for the paper **"Contrastive Learning for Results-Aware SPARQL Embeddings"**.

Two SPARQL queries can look almost the same and return different answers, or look different and return the same answers. We execute a query log **once, offline**, label query pairs by the **Jaccard overlap of their answer sets**, and fine-tune pretrained encoders with a contrastive loss on these labels. At inference, the encoder needs **only the query text**.

## ▶ Interactive demos

[![Animated preview of the interactive demos](docs/media/demo_preview.gif)](https://hzahera.github.io/SPARQL-Embeddings/)

| | |
|---|---|
| **[Demo 1 · Results-aware embeddings](https://hzahera.github.io/SPARQL-Embeddings/embeddings.html)** | Why text similarity is the wrong signal, and a map of real SQuARE test queries: text-based embeddings next to results-aware ones. |
| **[Demo 2 · Retrieval and caching](https://hzahera.github.io/SPARQL-Embeddings/tasks.html)** | Real queries where text similarity retrieves the wrong query, and how fine-tuning changes retrieval and result-cache decisions. |

Runs in the browser, nothing to install. Built from [`demo/`](demo/) into [`docs/`](docs/) and served with GitHub Pages.

## Key results

Ten pretrained encoders (text, code, general LM, SPARQL), before and after results-aware fine-tuning. Mean over encoders; fine-tuned = mean of five seeds.

| | DBpedia | Wikidata |
|---|---|---|
| **Query retrieval**, Hits@1 (complete labels): TF-IDF | 0.58 | 0.37 |
| zero-shot encoders | 0.49 | 0.30 |
| **results-aware fine-tuned encoders** | **0.70** | **0.43** |
| Encoders that improve | 10 / 10 | 10 / 10 |
| Control: 3 encoders trained on **syntax** labels (vs. answer labels) | 0.52 (vs. 0.72) | 0.36 (vs. 0.43) |
| **Result caching**, hit accuracy at θ = 0.8: zero-shot → fine-tuned | 0.15–0.33 → **0.59–0.79** | 0.01–0.10 → **0.21–0.37** |

A fixed similarity threshold is still not enough for safe reuse (see the paper, RQ2): ranking quality and calibration are distinct problems.

## The SQuARE resource

| | DBpedia | Wikidata |
|---|---|---|
| Unique queries (train / valid / test) | 3,781 / 473 / 473 | 7,743 / 968 / 968 |
| Positive pairs (J ≥ 0.8) | 39,430 | 37,208 |
| Negative pairs (J ≤ 0.1) | 93,480 | 168,360 |

Splits are built from connected components of positive pairs, so no query, positive pair or answer set is shared across splits. Files: `datasets/{dbpedia,wikidata}-splits/` (`*-train/valid/test.json` for training and retrieval, `*-test-rq3.json` with complete pair labels for caching).

How the dataset is built, and how to reproduce the splits: [`datasets/DATASET.md`](datasets/DATASET.md).

```json
{"anchor_query": "SELECT ...",
 "positive_queries": [{"query": "SELECT ...", "answer_jaccard": 1.0}],
 "negative_queries": [{"query": "SELECT ...", "answer_jaccard": 0.04}]}
```

## Quick start

```bash
pip install -r requirements.txt && pip install -e .
python scripts/validate_data.py --dataset dbpedia          # sanity checks on the splits

# one model
python scripts/run_experiment.py --models e5-base-v2 --datasets dbpedia --seeds 42 43 44 45 46

# full benchmark (10 encoders x 2 graphs x 5 seeds), then the paper tables
python scripts/run_experiment.py --models all --datasets dbpedia wikidata --seeds 42 43 44 45 46
python scripts/aggregate_results.py && python scripts/generate_tables.py
```

Each run evaluates zero-shot, fine-tunes, selects the checkpoint on validation nDCG@10, and evaluates retrieval, alignment and caching on test. Runs resume; use `--force` to recompute. Gated models (Gemma) need `huggingface-cli login`.

## Environment

Python 3.10, CUDA 12.8.

```bash
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins the exact versions used for the paper. The packages in its last block are only needed to rebuild the dataset from the SPARQL endpoints and to run the notebooks.

The checkpoints were trained under transformers 5.16.1 (DBpedia: BGE, CodeBERT, GraphCodeBERT, GTE, MPNet, Qwen2.5-Coder, Qwen3-Embedding, SPBERT) or 4.57.3 (DBpedia E5 and Gemma, and all Wikidata runs). Under the pinned 5.16.1, every fine-tuned checkpoint reproduces its stored test results exactly. Three Wikidata zero-shot baselines (Gemma, CodeBERT, Qwen2.5-Coder) were scored under 4.57.3 and differ by one test anchor (±0.0012 Hits@1) under 5.16.1. To reproduce them exactly, run `pip install transformers==4.57.3 tokenizers==0.22.1 safetensors==0.7.0`.

## Reproducing the paper

All commands run from the repository root with `export PYTHONPATH="$PWD/src:$PYTHONPATH"`.
Scripts write to `results/`; the files behind the paper are in
[`experiment_results/`](experiment_results/) (same layout, see its README). To run the
analysis steps on them without retraining: `ln -s experiment_results results`.

| Paper | Command | Output |
|---|---|---|
| Training, Table 3 (stored labels), Table 4 (caching), Spearman correlation (Sec. 5.2) | `python scripts/run_experiment.py --models all --datasets dbpedia wikidata --seeds 42 43 44 45 46`, then `python scripts/aggregate_results.py` and `python scripts/generate_tables.py` | `results/raw/`, `results/aggregated/`, `results/tables/` |
| Non-neural baselines (Tables 2, 3) | `bash scripts/baselines/run_baselines.sh` | `scripts/baselines/results_<graph>.json` |
| Per-anchor scores (input of Table 3 and the CIs) | `python scripts/analysis/per_anchor_scores.py --dataset <graph>` | `results/per_anchor/` |
| Table 2 (complete labels), paired CIs, seen/unseen templates (Sec. 5.2) | `python scripts/analysis/paired_analysis.py --dataset <graph>` | `results/analysis/paired_analysis_<graph>.json` |
| Syntax-supervised control (Sec. 5.2) | `MODELS="e5-base-v2 qwen3-embedding codebert" SEEDS="42 43 44" bash scripts/syntax_control/run_syntax_control.sh` | `results_syntax_control/`, `control_results.json` |
| Caching intervals (Sec. 5.3) | `python scripts/analysis/cache_ci.py` | `results/analysis/cache_ci.json` |
| Threshold selection and risk-coverage curves (Sec. 5.3) | `python scripts/analysis/calibrate_thresholds.py --dataset <graph>` | `results/analysis/calibration_<graph>.json` |
| Checkpoint selection (Sec. 5.1) | `python scripts/analysis/checkpoint_selection.py` | `results/analysis/checkpoint_selection.json` |
| Inference time (Sec. 5.1) | `python scripts/analysis/timing_inference.py --dataset wikidata` | `results/analysis/timing_inference_wikidata.json` |
| Dataset statistics (Table 1, Sec. 4) | `datasets/dataset_statistics.ipynb`; templates and seen/unseen anchors: `cd scripts/audit && python audit_splits.py ../../datasets/<graph>-splits <graph> audit_<graph>.json` | `datasets/statistics/`, `scripts/audit/audit_<graph>.json` |
| Label threshold analysis (Sec. 3.1, released with the resource) | `LT_MODE=full jupyter nbconvert --to notebook --execute scripts/analysis/label_thresholds.ipynb` (needs the execution cache) | `results/analysis/label_thresholds/`; generated files in `experiment_results/analysis/label_thresholds/` |
| Labeling time (Sec. 4) | `python scripts/analysis/timing_labeling.py --dataset <graph>` (needs the execution cache) | `results/analysis/timing_labeling_<graph>.json` |

Training was run on one NVIDIA A100 80 GB per run.

## Repository layout

```text
src/result_aware/   training (loss.py, mining.py, trainer.py), evaluation, caching simulation
configs/            experiment.yaml (all hyperparameters) and one YAML per encoder
scripts/            run_experiment.py, aggregate_results.py, generate_tables.py
  baselines/        non-neural baselines (TF-IDF, BM25, algebra similarity, ...)
  syntax_control/   control: same training with syntax-based labels
  analysis/         thresholds, timing, paired bootstrap, per-anchor scores
  audit/            split leakage audit
datasets/           SQuARE splits and statistics
demo/, docs/        interactive demo (sources and single-file build)
tests/              unit tests (pytest -q, no model downloads)
```

## Method in brief

InfoNCE (temperature 0.05) with in-batch candidates, 4 execution-verified explicit negatives per anchor and known positives masked. From epoch 2, every two epochs, the current model's 8 hardest **labeled** negatives are mined (75% hard / 25% random). AdamW, lr 1e-5, 20 epochs, batch 4 x 8 accumulation. Validation and test queries are never seen by the miner, and the cache threshold is never tuned on test. To add an encoder: `python scripts/add_model.py --key <name> --hf-id <hf-model> ...`, no code changes.

## License

Code: MIT (see [`LICENSE`](LICENSE)). Data: CC BY-SA 4.0, since the DBpedia answer sets derive from DBpedia (CC BY-SA); Wikidata is CC0. Queries derive from LSQ; its attribution requirements apply.

## Dataset and citation

The SQuARE dataset, including the answer caches, is archived on Zenodo: https://doi.org/10.5281/zenodo.23923443. Download and file formats: [`datasets/DATASET.md`](datasets/DATASET.md). To cite the code or the dataset, see [`CITATION.cff`](CITATION.cff) (GitHub shows it under "Cite this repository").
