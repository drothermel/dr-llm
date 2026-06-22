# HumanEval Pool Extraction

This note documents the one-off workflow used to extract HumanEval-related LLM
code generation attempts from historical `dr-llm` pool projects.

## Goal

The goal is to build a broad first-pass analysis dataset for parsing and
clustering attempted HumanEval code generations. The extracted table keeps the
raw, unnormalized model output plus enough provenance to compare generations
against the decoder input description that produced them.

The current workflow intentionally favors inspectable artifacts over a polished
library API:

1. Explore candidate pool schemas and row payloads.
2. Dump candidate pool rows to per-pool JSONL gzip files.
3. Extract decoder/code-attempt rows into one Parquet table.
4. Split that Parquet table into one Parquet file per HumanEval task id for
   targeted per-function analysis.
5. Use the inspection notebook for task-local exact duplicate checks over raw
   model outputs.

For the richer encoder-decoder trace workflow over these same dumped pool
artifacts, see `rich-trace-extraction.md`.

## Scripts

The implementation lives in `scripts/`:

- `humaneval_pool_extract_common.py` contains shared models, policy helpers,
  Docker project state restoration, row streaming, and extraction logic.
- `explore_humaneval_pool_policy.py` reports schema, row count, HumanEval
  policy hits, decoder candidate counts, and description availability.
- `dump_humaneval_candidate_pool_rows.py` streams selected pools to one
  `.jsonl.gz` file per pool and writes a manifest.
- `extract_humaneval_code_attempts.py` reads the dump and writes a unified
  Parquet file. It uses the existing `nl-code` parsed HumanEval cache, when
  available, to backfill official prompt text for decoder rows whose pool
  payload does not store the prompt directly.
- `split_humaneval_attempts_by_task.py` reads the unified Parquet file and
  writes one `.parquet` file per `human_eval_task_id` under `per_elem/`, plus
  one deduplicated JSONL file per task for quick raw-output inspection.

The interactive inspection notebook lives in `nbs/inspect/`:

- `humaneval_exact_dedupe.py` provides a dropdown over the per-task Parquet
  files, a run button to load one task, exact-match duplicate metrics for
  `raw_code_output`, a top-100 repeat-count chart, and dataframes for the
  loaded generations and repeated raw outputs.

Parquet writing is intentionally handled with an ephemeral dependency:

```bash
uv run --with pyarrow python scripts/extract_humaneval_code_attempts.py <dump-dir>
```

## Scope And Policy

The first extraction targets these pool projects:

- all pools in `code_comp_t1`
- all pools in `code_comp_v0`
- `nl_latents:nl_latents`

The HumanEval identity policy is exact:

- include IDs matching `human_eval/HumanEval/<n>`
- exclude `humaneval_pro/HumanEvalPro/<n>` and all other datasets
- prefer structured IDs in pool key columns and metadata lineage over
  free-text payload matches

The Parquet extraction includes only decoder/direct code-generation attempts:

- `response_json.text` for decoder rows
- `response_json.decoded_code` for `nl_latents` rows if exact HumanEval
  identity is present
- encoder rows are not included as standalone attempts

The decoder input description is kept in `decoder_input_description`.
Description sources are recorded in `decoder_input_description_source` and can
come from:

- `metadata.source_text`
- `metadata.source_sample_payload.text`
- `request.prompt`
- `humaneval_cache.prompt`

## Current Artifacts

The current run wrote artifacts outside the repository:

```text
/Users/daniellerothermel/drotherm/data/code-comp/dr-llm-humaneval-pool-dumps/20260621_manual
```

Important files:

- `policy_summary.json`
- `manifest.json`
- one `*.jsonl.gz` dump per selected pool
- `humaneval_code_attempts.parquet`
- `humaneval_code_attempts_preview.csv`
- `per_elem/human_eval-<n>-decode.parquet`
- `per_elem/human_eval-<n>-decode-dedup.jsonl`
- `per_elem/manifest.json`

The artifact directory is about 1.4 GB after the per-task split. The unified
Parquet file is about 102 MB, and `per_elem/` is about 224 MB. These files are
analysis outputs and should not be committed.

The `per_elem/` split uses the standard `.parquet` extension. Each file keeps
the same columns as `humaneval_code_attempts.parquet`, but contains only rows
for one canonical HumanEval task id. For example:

```text
per_elem/human_eval-0-decode.parquet
```

Each per-task split also writes a deduplicated JSONL file for quick human
inspection. Every line has the shape:

```json
{"out": "<raw model output>", "count": 1}
```

The JSONL preserves raw, unparsed generation text and sorts rows by descending
exact-match count.

The notebook uses the same raw, unparsed `raw_code_output` column. Its exact
duplicate counts are therefore exact raw generation string matches, before any
parsing to separate code from dialogue, Markdown fences, or explanatory text.

## Validation Results

The completed run produced:

- 26 dumped pools
- no missing dump files
- no dump row-count mismatches against the manifest
- 203,407 extracted HumanEval code attempts
- 203,407 non-empty `raw_code_output` values
- 203,407 non-empty `decoder_input_description` values
- zero extracted `HumanEvalPro`, `MbppPro`, `ClassEval`, or other dataset IDs
- 163 per-task Parquet files in `per_elem/`
- 163 per-task deduplicated JSONL files in `per_elem/`
- 172,454 task-local unique raw outputs across the deduplicated JSONL files

Description source counts:

```text
metadata.source_text      193,548
humaneval_cache.prompt      9,856
request.prompt                 3
```

Rows by extracted pool:

```text
code_comp_t1/budget_dec_v0_size6                  146,326
code_comp_t1/budget_dec_v0                         24,363
code_comp_t1/budget_dec_v0_output_only             10,190
code_comp_v0/official_decoder_t0                    9,856
code_comp_t1/dec_v0_orig                            4,074
code_comp_t1/dec_v0_orig_docstring                  4,074
code_comp_t1/dec_v0_orig_docstring_output_only      4,072
code_comp_t1/decoder_t2_smoke_3                       225
code_comp_t1/decoder_t1_smoke_3                       224
code_comp_v0/reexport_seed_decoder_simple               1
code_comp_v0/tde_20260510_0345                          1
code_comp_v0/tds_20260510_0344                          1
```

The `nl_latents` pool was dumped but contributed zero rows under the exact
HumanEval policy. It remains useful in the raw dump for auditability.

## Reproduction Commands

Policy exploration:

```bash
uv run python scripts/explore_humaneval_pool_policy.py \
  --sample-limit 100 \
  --output /Users/daniellerothermel/drotherm/data/code-comp/dr-llm-humaneval-pool-dumps/20260621_manual/policy_summary.json
```

Raw dump:

```bash
uv run python scripts/dump_humaneval_candidate_pool_rows.py \
  --output-dir /Users/daniellerothermel/drotherm/data/code-comp/dr-llm-humaneval-pool-dumps/20260621_manual \
  --batch-size 2000
```

Parquet extraction:

```bash
uv run --with pyarrow python scripts/extract_humaneval_code_attempts.py \
  /Users/daniellerothermel/drotherm/data/code-comp/dr-llm-humaneval-pool-dumps/20260621_manual
```

Per-task split:

```bash
uv run --with pyarrow python scripts/split_humaneval_attempts_by_task.py \
  /Users/daniellerothermel/drotherm/data/code-comp/dr-llm-humaneval-pool-dumps/20260621_manual/humaneval_code_attempts.parquet
```

Exact duplicate inspection notebook:

```bash
uv run marimo edit nbs/inspect/humaneval_exact_dedupe.py
```

All scripts preserve the original running/stopped state of Docker-managed
`dr-llm` projects.

## Current State

The scripts are suitable for the current manual analysis pass. They should stay
under `scripts/` unless this becomes a supported public workflow. If the output
schema becomes a product contract, the next cleanup should add focused tests for
the policy helpers and decide whether to make `pyarrow` a formal dependency.
