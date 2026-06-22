# Normalize Data

This directory collects the current notes for normalizing HumanEval/code-comp
analysis data across the data sources used so far.

## Purpose

This area is for work toward the normalization goal described in
[GOAL.md](GOAL.md): ingesting, parsing, normalizing, standardizing, and
documenting backups, pools, logs, and other artifacts from previous
natural-language/code encoder-decoder experiments.

The current work keeps source-specific extraction rules explicit while shaping
outputs toward compatible Parquet tables with stable provenance, prompt, input,
output, model, usage, cost, and task identity columns.

This is not yet a single canonical dataset contract. The current docs describe
the extraction passes that exist today and the validation results from the most
recent local runs. `PLAN.md` is reserved for the higher-level unification plan.

## Current State

- `humaneval-pool-extraction.md` documents the first-pass pool dump workflow
  that extracts broad HumanEval decoder/direct code attempts from historical
  pool data.
- `rich-trace-extraction.md` documents the pool-derived rich trace workflow
  that pairs clean decoder rows with their encoder rows and writes normalized
  encoder-decoder trace artifacts.
- `dspy-rich-trace-extraction.md` documents the DSPy parsed-report workflow
  that extracts direct and encoder-decoder HumanEval eval attempts into a shape
  compatible with the pool rich trace table, with DSPy-specific provenance and
  evaluation fields.

## Source Notes

- [HumanEval pool extraction](humaneval-pool-extraction.md)
- [Pool rich trace extraction](rich-trace-extraction.md)
- [DSPy rich trace extraction](dspy-rich-trace-extraction.md)

## Planning

- [Goal](GOAL.md)
- [Unification plan](PLAN.md)
