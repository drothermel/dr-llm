# DSPy Rich Trace Extraction

This note documents the DSPy HumanEval rich-trace workflow over the canonical
sessionized DSPy corpus. It is parallel to the pool-based rich trace extraction
and does not read the existing aggregate performance CSVs.

## Goal

The goal is to recover HumanEval eval attempts with enough context to compare
against the pool-derived rich trace table:

- encoder prompt, input, and output when the attempt is encoder-decoder
- decoder or direct prompt, input, and output
- model, usage, cost, and finish metadata
- DSPy provenance and `test_pass_rate`

The current implementation uses parsed eval reports from the canonical corpus
because they preserve prompt messages, generated outputs, attempts, pass rates,
and generation-call linkage in one place.

## Source And Policy

The source corpus is:

```text
/Users/daniellerothermel/drotherm/data/code-comp/dspy-exps/v0
```

The workflow reads:

```text
parsed_eval_reports/*.eval_report.json
```

It intentionally does not use
`../nl-code/data/humaneval-dspy-sample-performance/*.csv` as source data. Those
files are useful sample-level performance summaries, but they do not preserve
the full prompt/input/output trace needed for rich extraction.

A clean eval attempt is:

- non-skipped
- has `raw_completed_code`
- has `test_pass_rate`
- has linked `generation_call_ids`
- direct rows include a `direct_code_from_stub` generation call
- encoder-decoder rows include `encode_code_spec` and `decode_code_spec`
  generation calls

Rows that fail this rule are kept in a messy JSONL artifact with a primary
classification and all classification reasons.

## Scripts

The implementation lives in `scripts/`:

- `dspy_rich_trace_extract_common.py` contains parsed-report models,
  classification logic, DSPy field parsing, rich row construction, and artifact
  path helpers.
- `split_dspy_rich_trace_candidate_rows.py` reads parsed eval reports and
  writes clean/messy records plus manifests.
- `extract_dspy_rich_trace_rows.py` reads clean records and writes the rich
  Parquet table.
- `split_dspy_rich_traces_by_dataset_task.py` splits the rich Parquet by
  dataset and task id.

## Artifact Layout

The split step writes:

```text
dspy_rich_trace_split/
  manifest.json
  summary.json
  clean/eval_attempts.jsonl.gz
  messy/eval_attempts.jsonl.gz
```

The extraction step writes:

```text
dspy_rich_traces/
  dspy_rich_trace_attempts.parquet
  by_dataset/
    manifest.json
    human_eval/all.parquet
    human_eval/HumanEval__0.parquet
    ...
```

## Rich Row Shape

The Parquet table is a superset of the pool rich-trace shape. Direct rows keep
encoder fields null. Encoder-decoder rows fill both encoder and decoder fields.

Important compatibility columns include:

- `attempt_id`, `dataset`, `task_id`, `data_sample_id`, `run_id`
- `enc_prompt_json`, `enc_prompt_text`, `enc_input`, `enc_output`
- `dec_prompt_json`, `dec_prompt_text`, `dec_input`, `dec_output`
- encoder and decoder `provider`, `model`, `finish_reason`, `usage_json`,
  `cost_json`, and `latency_ms`
- `rich_extraction_level`

DSPy-specific columns include:

- `dspy_session_id`
- `dspy_report_path`
- `dspy_attempt_id`
- `dspy_generation_type`
- `dspy_dataset_index`
- `dspy_repeat_index`
- `dspy_test_pass_rate`
- `dspy_raw_completed_code`
- `dspy_extracted_code`
- `dspy_generation_call_ids`
- `dspy_encoder_call_id`
- `dspy_decoder_call_id`
- `dspy_encoder_call_output`
- `dspy_decoder_call_output`

`dec_output` is `raw_completed_code`, so the primary output column remains the
code-like generation. The raw DSPy decoder call output with structured field
markers is preserved separately in `dspy_decoder_call_output`.

## Reproduction Commands

Classify clean and messy eval attempts:

```bash
uv run python scripts/split_dspy_rich_trace_candidate_rows.py \
  /Users/daniellerothermel/drotherm/data/code-comp/dspy-exps/v0
```

Extract rich Parquet:

```bash
uv run --with pyarrow python scripts/extract_dspy_rich_trace_rows.py \
  /Users/daniellerothermel/drotherm/data/code-comp/dspy-exps/v0
```

Split by dataset and task:

```bash
uv run --with pyarrow python scripts/split_dspy_rich_traces_by_dataset_task.py \
  /Users/daniellerothermel/drotherm/data/code-comp/dspy-exps/v0
```

## Current Run Results

The scripts were run against the current corpus:

- `14` parsed eval reports
- `7,564` eval attempts
- `6,625` clean eval attempts
- `4,270` clean encoder-decoder attempts
- `2,355` clean direct attempts
- `939` messy eval attempts retained for later audit
- `163` HumanEval task IDs in parsed eval attempts
- `157` HumanEval task IDs in the clean rich Parquet

Clean task IDs exclude these parsed-attempt task IDs because they do not have
clean linked generation-call traces in this pass:

- `HumanEval/14`
- `HumanEval/15`
- `HumanEval/83`
- `HumanEval/100`
- `HumanEval/130`
- `HumanEval/139`

Messy primary reasons:

- `missing_generation_call_ids`: `699`
- `skipped_attempt`: `240`

Output sizes from this run:

- `dspy_rich_trace_split/`: `105M`
- `dspy_rich_traces/`: `46M`
- `dspy_rich_traces/by_dataset/`: `158` Parquet files
  (`human_eval/all.parquet` plus one file per clean task)

## Verification

Use focused helper tests and script checks:

```bash
uv run pytest tests/scripts/test_dspy_rich_trace_extract_common.py -v
uv run ruff check scripts/dspy_rich_trace_extract_common.py \
  scripts/split_dspy_rich_trace_candidate_rows.py \
  scripts/extract_dspy_rich_trace_rows.py \
  scripts/split_dspy_rich_traces_by_dataset_task.py \
  tests/scripts/test_dspy_rich_trace_extract_common.py
uv run ty check scripts/dspy_rich_trace_extract_common.py \
  scripts/split_dspy_rich_trace_candidate_rows.py \
  scripts/extract_dspy_rich_trace_rows.py \
  scripts/split_dspy_rich_traces_by_dataset_task.py
```

After running the artifact scripts, validate that:

- `dspy_rich_trace_split/manifest.json.clean_count` matches the rich Parquet
  row count
- `dspy_rich_trace_split/summary.json.classification_counts` clean counts sum
  to the rich Parquet row count
- `dspy_rich_traces/by_dataset/manifest.json.total_rows` matches the rich
  Parquet row count
- required decoder fields are non-empty for every clean row:
  `dec_prompt_text`, `dec_input`, `dec_output`, `dspy_test_pass_rate`
- encoder-decoder rows have non-empty `enc_prompt_text` and `enc_output`
- direct rows have null encoder prompt/output fields
