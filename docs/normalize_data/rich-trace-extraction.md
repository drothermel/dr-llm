# Rich Trace Extraction

This note documents the second-stage rich-trace workflow over the dumped pool
JSONL artifacts. It is intentionally parallel to the simpler HumanEval code
attempt extraction so the first-pass `humaneval_code_attempts.parquet` stays
stable and easy to inspect.

For the parallel DSPy HumanEval eval-attempt workflow, see
`dspy-rich-trace-extraction.md`.

## Goal

The goal is to recover a high-confidence subset of encoder-decoder traces from
the pool data with enough context to study:

- encoder prompt
- encoder input
- encoder output
- decoder prompt
- decoder input
- decoder output
- encoder and decoder model metadata

The current implementation chooses trace fidelity over maximum size. Rows that
do not satisfy the clean rule are kept in a messy artifact with reasons so they
can be audited or promoted by later rules.

## Clean Rule

A decoder candidate is clean only when it is a full encoder chain:

- the row is decoder-like by pool name or `dec_*` key columns
- the decoder row has `response_json.text`
- the decoder row has rendered `request_json.prompt`
- `metadata_json.source_kind == "encoder_sample"`
- `metadata_json.source_sample_id` is shaped like
  `encoder_pool/<source_pool>/<source_sample_id>`
- the referenced encoder row exists in the dumped pool rows
- the encoder row has rendered `request_json.prompt`
- the encoder row has `response_json.text`

This rule was chosen because the raw pool dumps preserve what was actually sent
to and returned from the LLMs. It is more reliable than reconstructing prompts
from current or historical prompt template code.

Rows that fail the clean rule are classified into messy categories such as
`missing_decoder_prompt`, `missing_encoder_row`, or
`missing_encoder_source_kind`. They are not discarded.

## Scripts

The implementation lives in `scripts/`:

- `rich_trace_extract_common.py` contains shared models, classification logic,
  prompt flattening, rich row construction, and split helpers.
- `split_rich_trace_candidate_rows.py` reads the dumped pool JSONL files and
  writes clean/messy candidate records plus manifests.
- `extract_rich_trace_rows.py` reads clean candidate records and writes a
  normalized rich Parquet table.
- `split_rich_traces_by_dataset_task.py` splits the rich Parquet by dataset and
  task id.

The scripts read the existing dump directory:

```text
/Users/daniellerothermel/drotherm/data/code-comp/dr-llm-humaneval-pool-dumps/20260621_manual
```

They do not start Docker or query live pool databases.

## Artifact Layout

The rich trace split writes:

```text
rich_trace_split/
  manifest.json
  summary.json
  clean/full_encoder_chain.jsonl.gz
  messy/messy_decoder_candidates.jsonl.gz
```

The rich extraction writes:

```text
rich_traces/
  rich_trace_attempts.parquet
  by_dataset/
    manifest.json
    human_eval/all.parquet
    human_eval/HumanEval__0.parquet
    ...
```

The clean JSONL stores paired records with both the decoder row and the
resolved encoder row. The rich Parquet flattens each pair into one row with
stable columns for prompts, outputs, IDs, template/config IDs, usage, cost, and
latency.

## Rich Row Shape

Important columns in `rich_trace_attempts.parquet` include:

- IDs: `attempt_id`, `dataset`, `task_id`, `data_sample_id`,
  `project_name`, `decoder_pool_name`, `decoder_sample_id`,
  `encoder_pool_name`, `encoder_sample_id`
- prompt/input/output fields: `enc_prompt_json`, `enc_prompt_text`,
  `enc_input`, `enc_output`, `dec_prompt_json`, `dec_prompt_text`,
  `dec_input`, `dec_output`
- template/config fields: `enc_prompt_template_id`,
  `dec_prompt_template_id`, `enc_llm_config_id`, `dec_llm_config_id`
- model metadata: encoder and decoder `provider`, `model`,
  `finish_reason`, `usage_json`, `cost_json`, and `latency_ms`
- provenance: `rich_extraction_level`, `source_kind`, `source_sample_id`,
  `source_pool_name`

`enc_input` is nullable in this first pass. The actual rendered encoder prompt
is preserved in `enc_prompt_text` and `enc_prompt_json`.

## Reproduction Commands

Classify clean and messy candidates:

```bash
uv run python scripts/split_rich_trace_candidate_rows.py \
  /Users/daniellerothermel/drotherm/data/code-comp/dr-llm-humaneval-pool-dumps/20260621_manual
```

Extract rich Parquet:

```bash
uv run --with pyarrow python scripts/extract_rich_trace_rows.py \
  /Users/daniellerothermel/drotherm/data/code-comp/dr-llm-humaneval-pool-dumps/20260621_manual
```

Split by dataset and task:

```bash
uv run --with pyarrow python scripts/split_rich_traces_by_dataset_task.py \
  /Users/daniellerothermel/drotherm/data/code-comp/dr-llm-humaneval-pool-dumps/20260621_manual
```

## Current Run Results

The scripts were run against the current dump:

- `181,328` full encoder-chain rows
- `274,376` messy decoder candidates retained for later audit
- dataset: `human_eval`
- task count: `163`
- contributing decoder pools:
  - `code_comp_t1/budget_dec_v0_size6`: `146,326`
  - `code_comp_t1/budget_dec_v0`: `24,363`
  - `code_comp_t1/budget_dec_v0_output_only`: `10,190`
  - `code_comp_t1/decoder_t2_smoke_3`: `225`
  - `code_comp_t1/decoder_t1_smoke_3`: `224`

Messy decoder candidate counts by primary classification:

- `missing_decoder_output`: `210,614`
- `missing_decoder_prompt`: `51,539`
- `missing_encoder_source_kind`: `12,222`
- `missing_encoder_source_id`: `1`

Output sizes from this run:

- `rich_trace_split/`: `609M`
- `rich_traces/`: `462M`
- `rich_traces/by_dataset/`: `164` Parquet files (`human_eval/all.parquet`
  plus one file per task)

Rows such as official decoder, docstring-only decoder, non-HumanEval decoder,
response-only, and `nl_latents` decoded-code rows are intentionally messy in
this first pass.

## Verification

Use the focused helper tests and script checks:

```bash
uv run pytest tests/scripts/test_rich_trace_extract_common.py -v
uv run ruff check scripts/rich_trace_extract_common.py \
  scripts/split_rich_trace_candidate_rows.py \
  scripts/extract_rich_trace_rows.py \
  scripts/split_rich_traces_by_dataset_task.py \
  tests/scripts/test_rich_trace_extract_common.py
uv run ty check scripts/rich_trace_extract_common.py \
  scripts/split_rich_trace_candidate_rows.py \
  scripts/extract_rich_trace_rows.py \
  scripts/split_rich_traces_by_dataset_task.py
```

After running the artifact scripts, validate that:

- `rich_trace_split/manifest.json.clean_count` matches the rich Parquet row
  count
- `rich_trace_split/summary.json.classification_counts.full_encoder_chain`
  matches the rich Parquet row count
- `rich_traces/by_dataset/manifest.json.total_rows` matches the rich Parquet
  row count
- required rich columns are non-empty for every row:
  `enc_prompt_text`, `enc_output`, `dec_prompt_text`, `dec_input`,
  `dec_output`
