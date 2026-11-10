# SQuARE dataset: how it is built

SQuARE is built in three stages per graph. Stages 1 and 2 select which LSQ queries
enter the dataset; Stage 3 executes them in full, computes all labels and creates the
splits. Every label in the released files comes from Stage 3.

All scripts are in `datasets/scripts/dataset_construction/<g>/` (`<g>` = `dbpedia` or
`wikidata`).

```
Stage 1  01_query_collection_<g>.ipynb    LSQ log -> executable, non-empty queries
           -> <g>-splits/sources/successful_queries_<g>.json
Stage 2  02_triplet_mining_<g>.ipynb      first labeling pass (first 200 rows per query)
           -> <g>-splits/sources/<g>-triplets-dataset.json
Stage 3  rebuild_<g>_clean_splits.py      full execution, Jaccard labels, splits, leakage checks
           -> <g>-splits/unique_queries.json            (frozen query list)
           -> <g>-splits/query_execution_cache.jsonl    (answers of every query)
           -> <g>-splits/<g>-{train,valid,test}.json    (released splits)
           -> <g>-splits/split_report.json
```

`<g>-splits/sources/` is the working folder of Stages 1 and 2. It is not part of the
repository (it is git-ignored); see "Released intermediate files" below.

## Download (Zenodo)

The released dataset, including the answer caches, is archived on Zenodo:
https://doi.org/10.5281/zenodo.23923443 (data license: CC BY-SA 4.0).

All files are flat in the record. The split and label files keep their names; three files
carry a graph prefix there, and the answer caches are gzipped:

| File on Zenodo | Path in this repository |
|---|---|
| `<g>-{train,valid,test}.json`, `<g>-{valid,test}-rq3.json`, `<g>-train-syntax.json` | `datasets/<g>-splits/` (same name) |
| `successful_queries_<g>.queries-only.json.gz` | `datasets/<g>-splits/` (same name) |
| `<g>-unique_queries.json` | `datasets/<g>-splits/unique_queries.json` |
| `<g>-split_report.json` | `datasets/<g>-splits/split_report.json` |
| `<g>-query_execution_cache.jsonl.gz` | `datasets/<g>-splits/query_execution_cache.jsonl` (unpacked) |

After downloading the record into the repository root:

```bash
g=wikidata                                   # or dbpedia
mkdir -p datasets/$g-splits
mv $g-*.json successful_queries_$g.queries-only.json.gz datasets/$g-splits/
mv datasets/$g-splits/$g-unique_queries.json datasets/$g-splits/unique_queries.json
mv datasets/$g-splits/$g-split_report.json   datasets/$g-splits/split_report.json
gunzip -c $g-query_execution_cache.jsonl.gz > datasets/$g-splits/query_execution_cache.jsonl
```

Which files you need: training uses `<g>-train.json` and `<g>-valid.json`; evaluation with
complete labels uses `<g>-test-rq3.json`; the answer caches (unpacked 1.7 GB and 22 GB) are
only needed to recompute labels, apply other thresholds or rebuild the splits.

## File formats

**Splits** (`<g>-{train,valid,test}.json`, `<g>-train-syntax.json`): JSON list of anchor records.

```json
{"anchor_query": "SELECT ...",
 "positive_queries": [{"query": "SELECT ...", "answer_jaccard": 1.0}],
 "negative_queries": [{"query": "SELECT ...", "answer_jaccard": 0.0}]}
```

**Complete labels** (`<g>-{valid,test}-rq3.json`): JSON object with

- `metadata`: counts and label definitions;
- `queries`: list of `{"id", "query_hash", "query", "n_answers"}`, where `id` is the position in the list;
- `positive_pairs`, `negative_pairs`: lists of `[id_i, id_j, J]` covering all unordered pairs.
  A pair in neither list has intermediate overlap (`0.1 < J < 0.8`).

**Answer cache** (`query_execution_cache.jsonl` (on Zenodo: `<g>-query_execution_cache.jsonl.gz`)): one JSON object per line.

```json
{"query": "SELECT ...", "query_hash": "7c5d0f73...", "status": "success",
 "answers": ["[[\"uri\",\"http://www.wikidata.org/entity/Q55606385\",\"\",\"\"]]"],
 "n_answers": 1, "error": null, "elapsed_s": 0.12}
```

- `query_hash`: SHA-256 of the query text with whitespace collapsed.
- `answers`: the distinct result rows. Each row is a JSON string holding one
  `[type, value, language, datatype]` tuple per projected variable, in projection order;
  an unbound variable is `["","","",""]`. Variable names and row order are not stored.
- `status`: `success` or `error`. Failed queries are not part of the splits.

**Query lists** (`unique_queries.json`): JSON list of SPARQL strings.
(`successful_queries_<g>.queries-only.json.gz`): gzipped JSON list of `{"sparql_query": ...}`.

## Stage 1: query collection

Pages through the LSQ SPARQL endpoint with `LIMIT 1000 OFFSET i*1000` (endpoint order,
no random sampling) and keeps the queries that return at least one row on the target
endpoint. DBpedia: `SELECT` queries only, without `COUNT` and `dbpo:wikipageid`. All
queries of the released dataset are `SELECT` queries on both graphs.

## Stage 2: first labeling pass

Reads the Stage 1 list in order and keeps the first `MAX_QUERIES` distinct queries
(compared after stripping each line; DBpedia 50,000, Wikidata 60,000, i.e. all of its
51,700). Each query is executed with `LIMIT 200`, and pairs are compared by the Jaccard
overlap of these rows (J >= 0.8 positive, J <= 0.1 negative). Anchors with at least one
positive are kept with up to 20 positives and 20 negatives. Only the *selection of
queries* is used later. Negatives are sorted by ascending J with a stable sort, so they
are the earliest J = 0 queries of the input; since every positive is itself an anchor,
the retained queries are the anchors plus a few such early queries.

## Stage 3: rebuild

Extracts every unique query from the Stage 2 file, removes scalar aggregates (aggregates
without `GROUP BY`), executes each query without a row limit, canonicalizes rows (term
type, value, language tag, datatype), computes J for all pairs that share a row,
builds connected components of positive pairs, assigns whole components to
train/validation/test (80/10/10 % of queries, largest component first, ties shuffled
with seed 42), builds anchor records within each split, and asserts that no query,
positive pair or answer set crosses splits.

## Recorded counts of the released version

| | DBpedia | Wikidata |
|---|---:|---:|
| Stage 1 output (non-empty queries) | 254,045 | 51,700 |
| Stage 2 input (first distinct queries) | 50,000 | 51,700 |
| Queries in the Stage 2 file (`unique_queries.json`) | 4,727 | 10,749 |
| Scalar aggregates removed in Stage 3 | 0 | 1,058 |
| Executed in Stage 3 | 4,727 | 9,691 |
| Successful, non-empty (final queries) | 4,727 | 9,679 (12 errors) |

Every query of `unique_queries.json` occurs in the Stage 2 input of its graph (checked
after whitespace normalization: 4,727/4,727 and 10,749/10,749). The DBpedia Stage 1 list
comes from a larger run of the Stage 1 notebook than its default settings.

DBpedia was built with an earlier version of the Stage 3 script that had no
scalar-aggregate filter; the DBpedia query list contains no scalar aggregates, so the
current script yields the same result.

## Released intermediate files

| File | Content |
|---|---|
| `<g>-splits/successful_queries_<g>.queries-only.json.gz` | Stage 1 output of the released version, query texts only (`[{"sparql_query": ...}]`, in the original order) |
| `<g>-splits/unique_queries.json` | Stage 2 output: the frozen query list, input of Stage 3 |

The original Stage 1 files also stored the result rows of every query (DBpedia 92 GB),
which are not needed by Stage 2 and are omitted here. The Stage 2 output files
(`<g>-triplets-dataset.json`) of this version were not retained; `unique_queries.json`
contains all their queries.

## Reproducing the released splits

Stage 3 runs from the frozen query list:

```bash
cd datasets/<g>-splits
mkdir -p ../repro_<g>
python ../scripts/dataset_construction/<g>/rebuild_<g>_clean_splits.py \
    --input unique_queries.json --output-dir ../repro_<g>
# compare ../repro_<g>/<g>-{train,valid,test}-clean.json with <g>-{train,valid,test}.json
```

This re-executes all queries, so results depend on the endpoint's graph version
(DBpedia 2022-12 snapshot, Wikidata truthy dump of 2025-01-22). The execution caches
(`query_execution_cache.jsonl`) are too large for this repository and are archived on
Zenodo (see "Download" above). With a cache, copy it into the output directory first
(`cp query_execution_cache.jsonl ../repro_<g>/`); Stage 3 then needs no endpoint
access and reproduces the released splits exactly.

## Rerunning Stage 2 on the released Stage 1 list

```bash
cd datasets/<g>-splits
mkdir -p sources
gunzip -c successful_queries_<g>.queries-only.json.gz > sources/successful_queries_<g>.json
# then run ../scripts/dataset_construction/<g>/02_triplet_mining_<g>.ipynb
```

Stage 2 then reads the same first 50,000 (DBpedia) or 51,700 (Wikidata) queries as the
released version. Its labels come from live `LIMIT 200` executions, so the selected
queries depend on the endpoint's graph version.

## Rerunning Stage 1

LSQ is paged with `LIMIT`/`OFFSET` without `ORDER BY`, and endpoints change over time,
so rerunning Stage 1 produces a *new* query list, not the released one.