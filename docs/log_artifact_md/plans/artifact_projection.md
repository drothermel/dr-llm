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
- `docs/log_artifact_md/plans/streaming_log.md`: concrete NATS JetStream event
  stream and object-store implementation plan.
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

## Initial Investigation Findings

Investigation date: 2026-06-02.

The local log corpus is small but useful for event and raw-provider payload
shape:

- `.dr_llm/generation_logs/generation_transcripts.jsonl`: 24,427 records,
  13.8 MB.
- `.llm_pool/generation_logs/generation_transcripts.jsonl`: 412 records,
  0.2 MB.
- `logs/nbs_hit_providers-20260511-170020.json`: 5 provider/model records after
  flattening the top-level mapping.
- Across those sources, encoded record sizes were small: p50 570 bytes, p90 692
  bytes, p99 966 bytes, max 14,435 bytes.
- Main event types were `provider.raw_response` (16,468), `llm_call.started`
  (4,186), `llm_call.succeeded` (3,636), and `llm_call.failed` (549).
- No generation-log truncation envelopes were present in this corpus.
- Full raw-payload-like fields were much more common than preview-only fields:
  24,092 full fields versus 2,520 preview fields.
- Repeated artifact-like field values were common in the logs, but many are
  tiny repeated stubs such as empty stderr, short stdin, empty raw JSON, or
  repeated synthetic test responses. This supports content hashing for identity,
  but does not by itself prove that aggressive deduplication is worth v1
  complexity.

The local pool-backed corpus is more representative for storage volume:

- Running local projects exposed 107 readable pools across 7 projects with
  `pool_catalog`; one running project had no `pool_catalog`.
- Those pools contained 1,309,637 sample rows and 1,295,363 completed rows.
- Aggregate JSON text payload volume was about 7.6 GB:
  - request JSON: 347.7 MB
  - response JSON: 5,185.3 MB
  - metadata JSON: 2,110.3 MB
- Response payloads are the primary artifact-store pressure. They are usually
  kilobyte-scale but have a meaningful tail:
  - 19,141 responses exceeded 16 KiB.
  - 2,660 responses exceeded 64 KiB.
  - 449 responses exceeded 256 KiB.
  - 1 response exceeded 1 MiB.
- Metadata is sometimes artifact-like under the current pool model:
  - 7,042 metadata blobs exceeded 16 KiB.
  - 551 metadata blobs exceeded 64 KiB.
  - 27 metadata blobs exceeded 256 KiB.
  - 2 metadata blobs exceeded 1 MiB.
- Requests are generally small:
  - only 1 request exceeded 16 KiB.
  - no request exceeded 64 KiB.
- Largest observed payloads:
  - response JSON max: 1,117,971 bytes.
  - metadata JSON max: 1,593,225 bytes.
  - request JSON max: 42,165 bytes.
- Several individual pools are already hundreds of MB to more than 1 GB of JSON
  text payloads. The largest measured pool was about 1.44 GB of request,
  response, and metadata text.

Compression observations from the local logs:

- Whole event records compressed with zlib level 6 to about 59% at p50 and 68%
  at p90.
- Medium response objects compressed well, but very small fields often expanded
  under per-value compression.
- Per-artifact compression should therefore avoid tiny independent compression
  units; compression should happen at chunk/shard granularity or only above a
  size threshold.

## Streaming-Log Decisions Now Available

The streaming-log implementation plan now answers several upstream questions
that were previously reserved for later design work.

- **Projection input:** the artifact projector consumes `DRLLM_EVENTS`, not pool
  tables, JSONL logs, or `DRLLM_WORK`. Work queue messages are operational state
  and are not needed for projection correctness.
- **Large-payload source:** large raw payload bytes live first in the permanent
  `DRLLM_PAYLOADS` NATS Object Store. That bucket is part of the source log, not
  the final artifact projection. The artifact projection resolves event
  `payload_refs` and rewrites those bytes into its own immutable layout.
- **Payload reference minimum shape:** every large payload reference should
  already provide `role`, `object_key`, `sha256`, `size_bytes`, `content_type`,
  `encoding`, and `compression`. These fields become the minimum source
  metadata needed by the artifact projector.
- **Publish ordering:** payload objects exist before the event that references
  them is durably published. The artifact projector can therefore treat a
  missing referenced object as corruption, partial bootstrap, or operational
  failure rather than as an expected race.
- **Identity and provenance:** `event_id` identifies the specific appended event,
  while `idempotency_key` identifies the logical fact where the event type
  defines idempotent behavior. Artifact projection should preserve both, plus
  `run_id`, `work_id`, `attempt_id`, `causation_id`, `correlation_id`, `source`,
  producer metadata, and event `schema_version` when present.
- **Delivery semantics:** projection consumers are at-least-once. The artifact
  projector must be idempotent across redelivery and replay, using event ID,
  idempotency key, payload role, object key, and content hash to detect already
  projected payloads.
- **Imported pool facts:** pool imports are reconstructed snapshot facts. The
  artifact projection should preserve reconstructed provenance but should not
  infer missing lifecycle events.
- **Rebuild posture:** because `DRLLM_EVENTS` plus `DRLLM_PAYLOADS` are the
  permanent source, artifact layout changes should be handled by replaying into
  a new projection/layout version rather than mutating old finalized shards.

## Artifact Decisions Now Mostly Answerable

These answers are now strong enough to carry into an implementation plan, though
the physical Zarr details still need a final design pass.

- **Artifact categories from streaming refs:** project all event `payload_refs`
  with large/raw roles into artifact storage in v1. This includes
  `request_json`, `response_json`, `stdout`, `stderr`, `generated_code`,
  `validation_report`, `metrics`-like payloads, and `error_detail` when those
  payloads are externalized by the streaming log. Oversized metadata remains an
  additional artifact candidate when it appears in imported pool facts or later
  event schemas.
- **Inline event payloads:** keep ordinary inline event payload fields in the
  metadata projection. The artifact projector should not extract small inline
  facts by default. It may spill inline metadata-like blobs above a configured
  threshold once the metadata projection defines which summaries stay queryable.
- **Artifact identity:** use a logical artifact identity that includes source
  event identity, payload role, object key, and content hash. Content hash alone
  is not enough because the same bytes may play different semantic roles or
  appear in different provenance contexts.
- **Pointer metadata:** artifact references emitted to or consumed by metadata
  should include source fields from the payload ref, source event provenance,
  logical artifact ID, projection/layout version, shard identifier, shard
  relative key/path, lane, offset, length, and integrity fields.
- **Checksum layering:** preserve the streaming-log `sha256` for the exact bytes
  stored in `DRLLM_PAYLOADS`. Add artifact-projection integrity checks for the
  projected logical bytes and finalized shard/chunk data. Do not collapse source
  identity and physical shard integrity into one checksum.
- **Dedupe posture:** content addressing is required for identity and duplicate
  detection. Physical deduplication in the artifact projection should remain
  optional for v1; repeated payloads can share identity metadata without adding a
  compaction/dedupe writer path.
- **Checkpointing posture:** projection progress should be tracked separately
  from shard finalization. A consumer acknowledgment should happen only after the
  artifact write, manifest/pointer update, and any needed projection checkpoint
  are durable enough for replay to be idempotent.
- **Lifecycle:** readers should only consume finalized immutable shards and their
  committed pointer/manifest data. In-progress shards are writer-owned and
  recoverable by replay.

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
- **Projection handoff:** whether artifact references are exposed to the
  metadata projection through a manifest table/file, derived projection events,
  or direct metadata-writer integration.
- **Checkpoint storage:** where artifact projector progress is recorded, and how
  checkpoints relate to finalized shard manifests and JetStream consumer
  acknowledgments.

## Initial Storage Implementation Suggestions

These are starting recommendations for the full artifact storage implementation
plan. They should be revisited after measuring a larger corpus and after the
streaming-log and metadata schemas are fixed.

- **Artifact categories:** treat all large/raw streaming-log `payload_refs` as
  first-class artifact candidates, including request JSON, response JSON,
  stdout/stderr, reasoning traces, generated code, validation outputs, metrics
  payloads, error details, and oversized metadata. Small request facts may still
  remain inline in event/metadata payloads when the streaming log does not
  externalize them.
- **Metadata spill threshold:** introduce a threshold-based rule for high-contact
  metadata versus low-contact artifacts. Based on current data, 16 KiB is a
  reasonable discussion threshold for moving rarely queried metadata blobs into
  artifact storage while retaining searchable summary fields in metadata.
- **Shard sizing:** design v1 around many finalized shards rather than one file
  per artifact. A target finalized shard range around 64-256 MiB uncompressed is
  plausible for the measured 7.6 GB local corpus: it would produce tens to low
  hundreds of shards, not thousands.
- **Large-object handling:** support individual artifacts above 1 MiB without a
  special fallback path. The measured corpus already contains response and
  metadata blobs around 1-1.6 MiB.
- **Vartext lanes:** store text-like artifacts as UTF-8 bytes plus pointer
  metadata. Keep separate logical lanes for structured JSON bytes, plain text,
  and binary bytes so readers can decode safely without guessing from path names.
- **Pointer fields:** include artifact ID, source event ID, source idempotency
  key, payload role, source object key, source content hash, content type,
  encoding, source compression, logical byte length, shard ID, shard path/group,
  byte offset, byte length, compression/layout version, and checksum. Metadata
  should reference artifacts by logical artifact ID and content hash, not
  physical path alone.
- **Compression:** compress chunks or shards, not every tiny artifact. Use a
  conservative text-friendly default and skip or batch very small values because
  the measured tiny fields often expand when compressed individually.
- **Checksums:** checksum each artifact's uncompressed bytes for identity and
  each finalized shard/chunk for corruption detection. This separates semantic
  dedupe/provenance from physical integrity checks.
- **Read API:** prioritize batch dereference from metadata query results. The
  first reader should support `read_artifact`, `read_artifacts`, decoded text,
  raw bytes, and request/response-pair retrieval.
- **Retention and rebuild:** make v1 append/finalize/rebuild-first. Do not add
  garbage collection or compaction while `DRLLM_EVENTS` and `DRLLM_PAYLOADS`
  remain the permanent source of truth and the metadata layer can point to a
  selected artifact layout version.
- **Migration posture:** current pool JSONB should be treated as a measurement
  source and possible backfill source, not as the long-term storage model.
  Existing rows show why the artifact projection should also consider oversized
  metadata, not just response blobs.

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

After the initial measurements, the strongest mostly-independent defaults are:

- local-first path layout with object-store-compatible relative keys
- finalized immutable shards only visible to readers
- chunk/shard-level compression rather than per-field compression
- batch-oriented reads as the default ergonomic path
- content hashes everywhere, but no mandatory v1 deduplication pass
- projection consumes `DRLLM_EVENTS` and resolves `DRLLM_PAYLOADS`; it never
  depends on `DRLLM_WORK`
- logical artifact identity includes provenance plus content hash, not content
  hash alone
- source payload checksums and projected shard checksums are separate integrity
  layers

## Discussion Outputs

The design discussion should leave with concrete answers or explicit follow-up
owners for the remaining physical-layout and handoff details:

- target shard size and chunk size ranges
- read API minimum surface
- metadata handoff mechanism for projected artifact references
- projection checkpoint storage and recovery behavior
- finalized shard manifest format
- local-first storage path and object-store compatibility constraints
- measurements that must be rerun before implementation

## Suggested Next Design Decisions

- Decide the first shard target size range. Start discussion at 64-256 MiB
  uncompressed, then tune by Zarr chunking constraints.
- Decide the v1 threshold for spilling oversized inline metadata-like payloads
  into artifacts. Start discussion at 16 KiB.
- Decide whether content-addressed dedupe stays identity-only in v1 or also
  becomes a physical storage optimization. Current data and the streaming-log
  object-key design support identity-only first.
- Decide the minimum artifact reader API before implementing writer internals,
  because notebook/debug workflows likely need batch request/response retrieval
  more than low-level shard inspection.
- Decide whether metadata receives artifact references from a manifest,
  projector-emitted derived events, or a direct projection service boundary.
- Decide where projection checkpoints live and what must be durable before
  acknowledging a consumed event.

## Non-Goals For This Plan

- Define the streaming event envelope beyond the contract already established in
  `docs/log_artifact_md/plans/streaming_log.md`.
- Define the metadata graph schema.
- Add Zarr, NATS, or storage dependencies.
- Implement artifact writing, reading, compaction, or verification.
- Migrate existing pool rows out of Postgres JSONB.
