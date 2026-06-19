# Artifact Projection Design-Discussion Prep Plan

## Purpose

This document records the v1 implementation plan for the blob-storage portion
of the artifact projection. The first implementation step is to update this
document to match the concrete v1 design, then pause for review before adding
dependencies or code.

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

## Artifact Decisions Now Locked For V1

These answers are now strong enough for implementation and for the metadata
projection to design against.

- **Artifact categories from streaming refs:** project selected large/raw
  payload roles into artifact storage in v1. Always project `request_json`,
  `response_json`, `stdout`, `stderr`, `generated_code`, `validation_report`,
  `metrics_payload`, `metrics`, and `error_detail`. Project `metadata_json`
  only when the source payload is at least the configured metadata spill
  threshold. Do not project `pool_schema` in v1.
- **Inline event payloads:** keep ordinary inline event payload fields in the
  metadata projection. The artifact projector should not extract small inline
  facts by default. It may spill inline metadata-like blobs above a configured
  threshold once the metadata projection defines which summaries stay queryable.
- **Artifact identity:** use a logical artifact identity that includes source
  logical fact identity, payload role, object key, and content hash. Content
  hash alone is not enough because the same bytes may play different semantic
  roles or appear in different provenance contexts. Preserve
  `source_ref.event_id` as provenance, not as the primary artifact identity
  input.
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
  committed pointer/manifest data. In-progress shard bytes are writer-owned, but
  accepted open references are durable in the sidecar so duplicate detection does
  not depend on shard finalization timing.

## Metadata-Facing Artifact Contract

The metadata projection should consume artifact references from finalized
artifact manifests and the rebuildable artifact index. It should not read
`DRLLM_PAYLOADS` directly, depend on pool rows, or know how the artifact writer
stages bytes.

The v1 metadata-facing reference is `ArtifactReference` with these fields:

- `artifact_id`: stable logical artifact identifier with an `art_` prefix.
- `projection_version`: artifact projection version, starting at
  `artifact-v1`.
- `source_ref`: durable source event and payload-reference identity:
  - `event_id`: event that introduced the payload reference.
  - `event_type`: source event type.
  - `schema_version`: source event schema version.
  - `idempotency_key`: logical fact key from the source event.
  - `payload_role`: source semantic role, such as `request_json`,
    `response_json`, `stdout`, `stderr`, `generated_code`,
    `validation_report`, `metrics_payload`, or `error_detail`.
  - `object_key`: immutable object key in `DRLLM_PAYLOADS`.
  - `sha256`: SHA-256 of the exact bytes in `DRLLM_PAYLOADS`.
  - `size_bytes`: byte length recorded on the source payload reference.
  - `content_type`: media type from the source payload reference.
  - `encoding`: text encoding, usually `utf-8`, or `binary` for non-text
    bytes.
  - `compression`: compression recorded on the source payload reference.
- `event_context`: optional run/work/attempt/causation/correlation/source,
  producer, and event metadata context copied from the source event.
- `logical_sha256`: SHA-256 of the bytes exposed by the artifact projection
  after any supported source decompression.
- `size_bytes`: logical byte length exposed to artifact readers.
- `lane`: one of `json`, `text`, or `binary`.
- `shard_id`: finalized shard identifier.
- `shard_uri`: object-store-compatible relative path to the finalized shard.
- `offset`: byte offset within the lane array.
- `length`: byte length within the lane array.
- `created_at`: artifact projection write timestamp.
- `schema_version`: artifact reference schema version.

`artifact_id` is derived from source logical provenance plus content, not from
content alone. Use canonical JSON for the hash input so part boundaries are
unambiguous:

```text
art_<sha256(canonical_json({
  "projection_version": "artifact-v1",
  "source_idempotency_key": source_ref.idempotency_key,
  "payload_role": source_ref.payload_role,
  "source_object_key": source_ref.object_key,
  "source_sha256": source_ref.sha256
}))>
```

The same bytes can therefore appear as distinct artifacts when they have
different semantic roles or provenance. Physical deduplication is not required
for v1.

## V1 Role Policy

The v1 projector uses a small explicit policy instead of projecting every
payload reference.

- Always project raw or low-contact payload roles: `request_json`,
  `response_json`, `stdout`, `stderr`, `generated_code`, `validation_report`,
  `metrics_payload`, `metrics`, and `error_detail`.
- Project `metadata_json` only when `size_bytes >= 16384`.
- Ignore `pool_schema`; schema facts belong in metadata and are often repeated
  during pool imports.
- Ignore events with no artifact-bearing payload refs, but checkpoint and
  acknowledge them so replay progress is explicit.

## Concrete Implementation Plan

The artifact projection should be implemented as a completely new package,
`src/dr_llm/artifact_projection/`, with no imports from `dr_llm.pool`,
`dr_llm.sampling`, or current generation-log sinks. The only upstream contracts
are the streaming-log event envelope and payload-reference shape.

Implementation steps:

1. **Update this plan document first.** Keep this metadata-facing contract
   current, including the concrete role policy, artifact identity formula,
   storage layout, index tables, ack/error behavior, CLI surface, test plan, and
   quality gate. Pause for review after this documentation change before adding
   dependencies or code.
2. **Add storage dependencies.** Add `zarr>=3,<4` and `numpy>=2,<3` for
   immutable Zarr v3 shard stores. Use the standard-library `sqlite3` module for
   the local mutable sidecar index. Do not add a Postgres dependency for
   artifact projection state.
3. **Create Pydantic models.** Define `ArtifactProjectionConfig`,
   `ArtifactSourceRef`, `ArtifactEventContext`, `PayloadArtifactSource`,
   `ArtifactReference`, `ShardManifest`, `ProjectionCheckpoint`, and
   `ProjectionError`. Use `extra="forbid"` for manifest and reference models so
   schema drift fails fast.
4. **Create the sidecar index.** Store finalized artifact references, open
   artifact references, finalized shard records, open shard state, projection
   checkpoints, and projection errors in `index/artifacts.sqlite3`. Treat this
   index as the normal-write idempotency authority and as rebuildable from
   finalized manifests, not as the permanent source of artifact bytes.
5. **Create the shard writer and storage backend.** Append payload bytes into
   lane-specific open-shard buffers, snapshot open shards into immutable
   contents at finalization, and delegate durable publishing to a
   `ShardStorageBackend`. The local backend stages Zarr v3 stores in
   writer-owned directories, writes manifest JSONL files plus finalized marker
   JSON, then atomically moves finalized outputs into place. Readers must never
   read staging shards.
6. **Create the reader.** Support `read_artifact`, `read_artifacts`,
   `read_text`, `read_json`, and `read_bytes` by looking up references in the
   sidecar index and reading finalized Zarr lane arrays by offset and length.
7. **Create the projector.** Consume `DRLLM_EVENTS` with explicit JetStream
   message acknowledgment control, ignore events without artifact-bearing
   `payload_refs`, fetch referenced bytes from `DRLLM_PAYLOADS`, verify source
   hashes and sizes, write artifacts, update manifests and the sidecar index,
   and acknowledge only after the durable projection outcome is recorded. Do not
   use the current `replay_events()` helper for projector delivery because it
   auto-acknowledges messages.
8. **Create CLI commands.** Add `dr-llm artifact-projection init`, `run`,
   `flush`, `inspect`, `rebuild-index`, `verify`, and `read`.
9. **Test the layer.** Add unit and storage tests first, then integration tests
   against Docker NATS once the streaming-log implementation exists.

Default storage layout:

```text
<artifact-root>/
  layouts/artifact-v1/
    shards/<shard_id>.zarr/
    manifests/<shard_id>.jsonl
    manifests/<shard_id>.finalized.json
    index/artifacts.sqlite3
    staging/<writer_session>/<shard_id>/
```

Default artifact root is `.dr_llm/artifacts`. Each finalized shard stores one
one-dimensional `uint8` Zarr array per used lane under `lanes/json`,
`lanes/text`, and `lanes/binary`. Text-like artifacts are UTF-8 bytes unless
their source reference states another encoding. JSON artifacts are stored as
bytes and decoded by readers, not queried inside Zarr.

Default physical parameters:

- target finalized shard size: 128 MiB uncompressed
- chunk size: 8 MiB
- large-object behavior: a single artifact may exceed the target shard size
- projected compression: Zarr chunk/shard compression, not per-artifact
  compression
- source compression support: `none` only in v1; unsupported source compression
  is recorded as a durable projection error
- default durable consumer: `drllm_artifact_projection_v1`
- default environment prefix: `DR_LLM_ARTIFACT_PROJECTION_`

SQLite sidecar tables:

- `artifact_references`: one row per finalized logical artifact reference,
  unique by `artifact_id`.
- `open_artifact_references`: one row per accepted but not yet finalized
  logical artifact reference, unique by `artifact_id`.
- `shards`: finalized shard metadata and finalized marker state.
- `open_shards`: open shard state used to group accepted open references.
- `projection_checkpoints`: projection progress keyed by projection version and
  durable consumer.
- `projection_errors`: durable recoverable projection errors, including source
  event/ref identity, error type, message, and timestamp.

The sidecar uses WAL mode, foreign keys, and explicit transactions. Duplicate
identical `artifact_id` references are no-ops across both open and finalized
references. A duplicate `artifact_id` with conflicting source or physical fields
is a durable projection error. The sidecar may store selected `source_ref`
values in flat SQL columns for indexing, but the durable JSON contract remains
the nested `ArtifactReference` and `ProjectionError` model shape.

Checkpoint and recovery rules:

- Consumer delivery is at least once; projection must be idempotent.
- A duplicate event/ref with the same `artifact_id` and identical source fields
  is a no-op.
- A duplicate `artifact_id` with conflicting source fields is a hard projection
  error.
- Event acknowledgments happen only after artifact writes, open/finalized
  reference updates, and checkpoint/error records are durable.
- Missing source objects, hash mismatches, size mismatches, and unsupported
  source compression are recorded as `ProjectionError` rows, checkpointed,
  acknowledged, and reported by `verify` instead of blocking later events.
- On restart, remove partial finalized outputs without finalized markers, resume
  open staging state when safe, and rebuild missing index rows from finalized
  manifests.

This is a hard-cutover design. Pool rows may still be imported into the
streaming log upstream, but artifact projection code must not contain pool
backfill, pool compatibility, lease, progress, or JSONB-storage logic.

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
- **Projection handoff:** answered for v1. Metadata consumes finalized artifact
  manifests and a rebuildable sidecar index.
- **Checkpoint storage:** answered for v1. Checkpoints live in the SQLite
  sidecar index and are recoverable against finalized manifests.

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
  per artifact. Use 128 MiB uncompressed as the v1 target. The earlier
  64-256 MiB range remains useful for later tuning, but the implementer should
  start with 128 MiB.
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
- **Metadata handoff:** use finalized shard manifests plus the rebuildable
  SQLite sidecar index. Do not publish derived artifact events in v1, and do not
  write directly to the metadata database from the artifact projector.
- **Clean cutover boundary:** the artifact projection package must not import or
  call pool, sampling, worker-lease, or generation JSONL code. Pool-equivalent
  results come from the combination of streaming-log import, artifact manifests,
  and metadata projection.
- **Migration posture:** current pool JSONB should be treated as a measurement
  source and possible backfill source, not as the long-term storage model.
  Existing rows show why the artifact projection should also consider oversized
  metadata, not just response blobs.

## Clean-Code Implementation Constraints

- Keep responsibilities narrow: `ArtifactRolePolicy` decides whether a ref is an
  artifact, `ArtifactProjector` orchestrates event processing, `ArtifactStore`
  owns logical artifact acceptance and reference promotion, `OpenShard` builds
  in-memory shard contents, `ShardWriter` orchestrates rotation and
  finalization, `ShardStorageBackend` owns durable publishing and finalized
  reads, and `ArtifactIndex` persists sidecar state.
- Use intention-revealing names. Avoid vague manager-style classes and avoid
  abbreviations in public model fields.
- Keep projector flow top-down and delegate low-level hashing, lane selection,
  SQLite persistence, and Zarr writes to named helpers.
- Prefer exceptions for invalid or corrupt internal states. Convert recoverable
  source/projection failures into durable `ProjectionError` rows at the
  projector boundary.
- Use Pydantic `BaseModel` with `extra="forbid"` for durable reference,
  manifest, checkpoint, and error models.
- Write focused tests before each major behavior slice where practical:
  identity, role policy, index idempotency, shard read/write, and projector ack
  behavior.

## CLI Surface

Add a new Typer sub-app registered as `dr-llm artifact-projection`:

- `init`: create directories and SQLite schema.
- `run`: consume `DRLLM_EVENTS`; include `--batch-size`, `--flush-on-exit`, and
  `--max-messages` options.
- `flush`: finalize the current staged shard.
- `inspect`: print root, version, shard counts, artifact counts, checkpoint, and
  error counts.
- `rebuild-index`: rebuild SQLite from finalized manifests.
- `verify`: verify manifests, index consistency, source/logical hashes, shard
  presence, and durable projection errors.
- `read`: read one artifact by ID as bytes, text, or JSON.

## Test Plan

Unit tests:

- Artifact ID is stable and changes on role, object key, source hash, or source
  idempotency key changes.
- Role policy projects selected roles, thresholds `metadata_json`, and ignores
  `pool_schema`.
- Lane selection maps JSON, text, and binary payloads correctly.
- SQLite index handles open/finalized insert, duplicate no-op, conflict error,
  checkpoint update, and rebuild from finalized manifests.
- Open-shard snapshots do not perform filesystem publishing, local shard storage
  writes finalized Zarr arrays, and reader returns bytes, text, and JSON.
- Projector with fake message and payload store acknowledges only after durable
  success or durable error.

Integration tests:

- With Docker NATS, publish an event with `response_json`, run the projector, and
  read the artifact back.
- Redelivered duplicate event/ref is idempotent.
- Missing payload ref records a projection error and does not block later
  events.
- `rebuild-index` restores readable artifacts from finalized manifests.

Quality gate for library-code implementation:

1. `uv run ruff format`
2. `uv run ruff check --fix .`
3. Manually fix any remaining lint issues in the repository.
4. `uv run ty check`
5. `uv run pytest tests/ -v -m "not integration"`
6. `./scripts/run-tests-local.sh`

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

## Remaining Discussion Outputs

The design discussion should leave with concrete answers or explicit follow-up
owners for the remaining physical-layout and handoff details:

- read API minimum surface
- local-first storage path and object-store compatibility constraints
- measurements that must be rerun before implementation

## Suggested Next Design Decisions

- Decide the v1 threshold for spilling oversized inline metadata-like payloads
  into artifacts. Start discussion at 16 KiB.
- Decide whether content-addressed dedupe stays identity-only in v1 or also
  becomes a physical storage optimization. Current data and the streaming-log
  object-key design support identity-only first.
- Decide the minimum artifact reader API before implementing writer internals,
  because notebook/debug workflows likely need batch request/response retrieval
  more than low-level shard inspection.

## Non-Goals For This Plan

- Define the streaming event envelope beyond the contract already established in
  `docs/log_artifact_md/plans/streaming_log.md`.
- Define the metadata graph schema.
- Add Zarr, NATS, or storage dependencies.
- Implement artifact writing, reading, compaction, or verification.
- Migrate existing pool rows out of Postgres JSONB.
