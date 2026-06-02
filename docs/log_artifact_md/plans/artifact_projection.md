# Artifact Projection Design-Discussion Prep Plan

## Purpose

This document plans the investigation needed to design the blob-storage portion
of the artifact projection. It is preparation for a design discussion, not a
final storage specification and not an implementation plan for the streaming log
or metadata graph.

The working assumption is that the streaming log and metadata projection have
their own concrete designs. This plan focuses on the artifact-store questions
that should be informed by existing data, plus the few storage defaults that can
be sketched early and refined after measurement.

## Known Inputs

- `docs/log_artifact_md/goal.md`: overall redesign goal and three-stage shape.
- `docs/log_artifact_md/streaming_log.md`: event log as durable source of truth.
- `docs/log_artifact_md/artifact_store.md`: artifact store purpose and Zarr v3
  direction.
- `docs/log_artifact_md/md_graph_projection.md`: metadata references,
  provenance, graph/fact projection shape.
- `docs/log_artifact_md/current_system/pool_streaming_log_mapping.md`: current
  pool behavior and the concerns that need explicit event/projection treatment.

## Existing Data Sources To Inspect

Start with local, non-invasive sources:

- `.dr_llm/generation_logs/generation_transcripts.jsonl`
- `.llm_pool/generation_logs/generation_transcripts.jsonl`
- `logs/nbs_hit_providers-20260511-170020.json`

Then inspect pool-backed data when a local project or DSN is available:

- persisted `request_json` and `response_json` sizes
- `metadata_json` shape and size
- provider/model/run distribution
- response stats already parsed by `src/dr_llm/pool/response_stats.py`
- any pool rows with raw provider payloads, reasoning traces, generated code, or
  validation-like artifacts

## Measurements To Collect

Collect summary tables rather than exhaustive payload dumps.

- Count records by source file, event type, provider, model, stage, and mode.
- Measure encoded event byte sizes: min, p50, p90, p99, max.
- Measure likely artifact fields separately where possible:
  - request payload bytes
  - response payload bytes
  - raw response text bytes
  - stdout/stderr bytes from headless providers
  - reasoning traces and reasoning detail bytes
  - generated-code or validation-artifact bytes, if present
- Count preview-only fields versus full raw-payload fields, such as
  `response_text_preview` versus `response_text`.
- Count truncation envelopes from the generation log sink.
- Identify repeated payload patterns that may benefit from content addressing or
  deduplication.
- Identify common read shapes in current inspection flows:
  - one artifact by ID
  - all artifacts for a run/pool-like slice
  - response text plus raw provider payload
  - request/response pairs for debugging
  - batch reads for notebooks or validation passes

## Data-Driven Design Questions

Use the measurements to prepare recommendations for these storage decisions.

- **Physical layout:** candidate shard size, chunk size, group hierarchy, and
  path structure.
- **Vartext format:** byte encoding, offset/length pointer shape, null handling,
  and whether JSON text and arbitrary bytes need separate lanes.
- **Checksum policy:** artifact-level versus shard-level checksums, verification
  cost, and when corruption should be reported.
- **Read API:** minimum retrieval operations for design discussion, including
  single-artifact read, batch read, decoded-text read, and raw-byte read.
- **Retention and compaction:** whether v1 should support only append/finalize
  plus rebuild, or also plan for garbage collection and re-sharding.
- **Operational limits:** expected local disk use, shard count, object-store
  compatibility constraints, and how large individual payloads can be before
  special handling is needed.

## Mostly-Independent Defaults To Sketch

These can be proposed before the measurement pass, but should be treated as
provisional.

- **Compression:** choose a conservative default codec and level for text-heavy
  payloads, then revisit after measuring entropy and read/write cost.
- **Path convention:** define an abstract layout that works for both local files
  and future object storage, without baking in a local-only assumption.
- **Batch retrieval:** design the reader around batch lookup from metadata
  artifact references, since analysis workflows are likely to request many
  related payloads at once.
- **Immutable lifecycle:** use write-in-progress state plus finalized shard
  state, with readers only allowed to consume finalized shards.
- **Layout versioning:** include a projection/layout version in artifact
  references so rebuilt layouts can coexist with older projections.

## Discussion Outputs

The design discussion should leave with concrete answers or explicit follow-up
owners for:

- initial artifact categories and which fields become artifacts versus metadata
- target shard size and chunk size ranges
- pointer/reference fields required by metadata
- checksum and verification policy for v1
- read API minimum surface
- retention, rebuild, and coexistence expectations
- local-first storage path and object-store compatibility constraints
- measurements that must be rerun before implementation

## Non-Goals For This Plan

- Define the streaming event envelope.
- Define the metadata graph schema.
- Add Zarr, NATS, or storage dependencies.
- Implement artifact writing, reading, compaction, or verification.
- Migrate existing pool rows out of Postgres JSONB.
