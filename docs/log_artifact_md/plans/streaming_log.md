# Streaming Log Implementation Plan

## Purpose

Build a new NATS JetStream based streaming-log system in parallel with the
existing pool implementation. The current pool system remains unchanged until a
future hard cutover. The new system should not reuse the pool storage,
coordination, worker, or JSONL logging implementation as architectural
constraints. It should use the pool system only as:

- A behavioral reference for work submission, claiming, retries, completion,
  errors, and progress.
- A migration source for importing existing pool data into the new permanent
  log.

The target final state is a clean async-native streaming-log foundation that can
support metadata and artifact projections without depending on old pool tables.

## Permanent Log Model

The permanent log is composed of two NATS JetStream backed resources:

1. Ordered event streams for durable lifecycle and provenance facts.
2. Immutable NATS Object Store buckets for large raw payloads that belong to
   those facts.

The Object Store is part of the permanent log substrate. It is not the future
artifact projection. Artifact projections are derived later by consuming the
event stream and resolving payload references.

Small structured facts may be stored inline in event payloads. Large raw
payloads, including requests, responses, stdout, stderr, generated code,
validation output, metrics payloads, and error details, should be stored in the
permanent object bucket and referenced from events.

## NATS Resources

The first implementation should create these JetStream resources:

- `DRLLM_EVENTS`: append-only permanent event stream.
- `DRLLM_WORK`: operational work-queue stream for claimable work.
- `DRLLM_PAYLOADS`: NATS Object Store bucket for immutable permanent payload
  objects.

`DRLLM_EVENTS` is archival truth. `DRLLM_WORK` is operational queue state and
can be rebuilt from durable submitted-work facts if needed. Acknowledging work
messages must never be the only record that work happened.

Recommended subjects:

- `drllm.events.>` for permanent events.
- `drllm.work.>` for claimable work.

Use explicit publish acknowledgments for all event and work publishes. Use
durable pull consumers with explicit acknowledgments for work processing and
projection consumption.

## Event Envelope

Every event should use a language-neutral JSON envelope validated by Pydantic in
the Python implementation.

Required envelope fields:

- `event_id`: globally unique event identifier.
- `event_type`: strict event type string.
- `schema_version`: integer event schema version.
- `occurred_at`: UTC timestamp for when the fact occurred.
- `producer`: producer identity and version metadata.
- `idempotency_key`: deterministic key for safe duplicate handling.
- `payload`: JSON object for inline facts.
- `payload_refs`: list of object references for large or raw payloads.

Optional envelope fields:

- `run_id`
- `work_id`
- `attempt_id`
- `causation_id`
- `correlation_id`
- `source`
- `metadata`

`event_id` identifies this specific appended event. `idempotency_key` identifies
the logical fact and should be stable across retries of the same publish.
Projection code should deduplicate by idempotency key where the event type
defines idempotent behavior.

## Payload Reference Contract

Large raw payloads must be written to `DRLLM_PAYLOADS` before publishing the
event that references them.

Each payload reference should include:

- `role`: semantic role, such as `request_json`, `response_json`, `stdout`,
  `stderr`, `generated_code`, `validation_report`, or `error_detail`.
- `object_key`: immutable object key in `DRLLM_PAYLOADS`.
- `sha256`: content hash of the exact stored bytes.
- `size_bytes`: byte length.
- `content_type`: media type, such as `application/json` or `text/plain`.
- `encoding`: text or binary encoding, such as `utf-8`.
- `compression`: compression algorithm if used, otherwise `none`.

Object keys should be content-addressed and deterministic, for example:

```text
sha256/<first-two-hex>/<full-sha256>
```

If the object already exists with the same hash, the writer may reuse it. If an
object exists at the same key with different content, that is a hard integrity
error.

## Event Types for V1

Import events:

- `pool_import_started`
- `pool_sample_imported`
- `pool_import_completed`
- `pool_import_failed`

Work lifecycle events:

- `work_submitted`
- `attempt_started`
- `provider_request_prepared`
- `provider_response_received`
- `attempt_succeeded`
- `attempt_failed`
- `work_retry_scheduled`
- `work_completed`
- `work_cancelled`

Operational events:

- `producer_started`
- `producer_stopped`
- `streaming_log_error`

The first implementation should keep the event set small and stable. Metadata
and artifact projection plans may add derived event types later, but they should
not require changing these core lifecycle facts.

## Async Python Implementation

Implement the first version in async Python using the official `nats-py`
client. Do not build a synchronous compatibility layer for the existing worker
runtime. The new worker path should be async-native.

Recommended package layout:

```text
src/dr_llm/streaming_log/
  __init__.py
  config.py
  events.py
  payloads.py
  client.py
  bootstrap.py
  workers.py
  ingest_pools.py
  cli.py
```

Core responsibilities:

- `config.py`: Pydantic settings for NATS URL, stream names, bucket name,
  durable consumer names, ack wait, max delivery, fetch batch size, and inline
  payload threshold.
- `events.py`: event type enum, producer model, event envelope model, and
  idempotency helpers.
- `payloads.py`: payload serialization, hashing, object key construction, and
  payload reference model.
- `client.py`: async connection manager, event publisher, payload writer, work
  publisher, and replay consumer helpers.
- `bootstrap.py`: idempotent creation or validation of JetStream streams,
  consumers, and object bucket.
- `workers.py`: async worker runtime using JetStream pull consumers.
- `ingest_pools.py`: pool snapshot import logic.
- `cli.py`: Typer commands for bootstrap, inspect, ingest-pool, and worker
  execution.

## Publish Ordering and Atomicity

For events with object payloads:

1. Serialize and hash payload bytes.
2. Write missing payload objects to `DRLLM_PAYLOADS`.
3. Build the event envelope with payload references.
4. Publish the event to `DRLLM_EVENTS`.
5. Require JetStream publish acknowledgment.

For submitted work:

1. Publish `work_submitted` to `DRLLM_EVENTS`.
2. Require publish acknowledgment.
3. Publish the corresponding claimable work message to `DRLLM_WORK`.
4. Require publish acknowledgment.

This ordering ensures archival truth exists before operational queue state.

If publishing a work message fails after the permanent event is acknowledged,
the work queue can be repaired by replaying `work_submitted` events and
republishing missing operational work messages.

## Work Queue Semantics

`DRLLM_WORK` should use JetStream pull consumers with explicit acknowledgments.
Workers fetch work when ready, process it, publish permanent lifecycle events,
and acknowledge work only after the terminal event for that processing step is
durably accepted.

The first worker design should replace pool leases with JetStream consumer
delivery and acknowledgment behavior:

- A fetched work message is the worker's active claim.
- Ack wait acts as the lease timeout.
- Workers should send in-progress acknowledgments or equivalent heartbeat
behavior for long-running provider calls if supported by the client.
- If a worker exits before acking, JetStream redelivers the work.
- Retryable failures should leave the message available for redelivery or
  explicitly nack it.
- Terminal failures should publish a terminal failure event before acking work.

The permanent event stream should record attempts. The work stream should only
coordinate available work.

## Pool Import Phase

The first phase of the full switchover is importing existing pools into the new
permanent log.

Add a command:

```text
dr-llm streaming-log ingest-pool --dsn <postgres-dsn> --pool-name <pool>
```

Behavior:

1. Load the pool schema from the existing pool catalog.
2. Emit `pool_import_started`.
3. Iterate all pool samples through read-only pool APIs.
4. For each row, emit `pool_sample_imported`.
5. Emit `pool_import_completed` with counts.

Use snapshot import semantics. Existing pool rows do not contain complete
lifecycle history, so the importer must not fabricate claim, attempt, retry, or
provider-call events. Imported sample facts should be marked as reconstructed
and should preserve the row's current state:

- pool name
- schema
- key values
- sample ID
- sample index
- run ID
- request
- response
- finish reason
- attempt count
- metadata
- created timestamp
- completion state

Importer idempotency keys should be deterministic from:

- source database identity if available
- pool name
- sample ID
- sample index
- row state hash

Repeated imports of unchanged rows should not create conflicting logical facts.

## Projection-Facing Guarantees

Metadata and artifact projectors should be able to rely on these guarantees:

- `DRLLM_EVENTS` is the ordered source of durable facts.
- Large payload bytes referenced by an event exist before that event is
  published.
- Payload references are immutable and content-addressed.
- Work queue messages are not required for projection correctness.
- Consumers may receive events at least once and must deduplicate by event ID or
  idempotency key according to their needs.
- Snapshot-imported pool facts are explicitly identified as reconstructed
  source facts.
- Future canonicalization, artifact extraction, and metadata graph assertions
  should be derived from events rather than from pool tables.

## CLI and Operations

Add a `streaming-log` Typer app under the existing `dr-llm` CLI.

Initial commands:

- `bootstrap`: create or validate streams, consumers, and object bucket.
- `inspect`: print stream, consumer, and object bucket status.
- `ingest-pool`: import one existing pool.
- `ingest-pools`: import multiple discovered pools from a DSN.
- `run-worker`: run the async JetStream-backed worker.

Local development should support a Docker NATS server with JetStream enabled.
Use the official `nats` image and persistent JetStream storage for integration
tests and manual development.

Recommended local server command:

```text
docker run -p 4222:4222 -v dr-llm-nats:/data nats -js -sd /data
```

## Testing Plan

Unit tests:

- Event envelope validation.
- Event type validation.
- Deterministic idempotency key generation.
- Payload hashing and object key construction.
- Payload reference serialization.
- Snapshot import event construction from `PoolSample`.

Integration tests with Docker NATS:

- Bootstrap creates streams and object bucket idempotently.
- Publishing an event requires a JetStream ack.
- Payload object write happens before event publish.
- Event consumers can replay from `DRLLM_EVENTS`.
- Work consumers receive messages from `DRLLM_WORK`.
- Unacked work is redelivered.
- Work is acked only after terminal event publication.
- Repeated pool import is idempotent.

Compatibility tests:

- Existing pool tests continue passing unchanged.
- Imported row counts match source pool row counts.
- Imported complete, error, incomplete, and leased pool states are represented
  faithfully as reconstructed snapshot facts.

Repository quality gate for library changes:

1. `uv run ruff format`
2. `uv run ruff check --fix .`
3. `uv run ty check`
4. `uv run pytest tests/ -v -m "not integration"`
5. `./scripts/run-tests-local.sh`

## Assumptions

- The implementation should optimize for the best final architecture, not for
  compatibility with the current pool implementation.
- The new system is async-native.
- The current pool system remains unchanged until a later hard cutover.
- The stream contract should be language-neutral so Rust or other languages can
  produce and consume events later.
- NATS Object Store is part of the permanent log substrate.
- Artifact and metadata projections are future derived systems built on top of
  this permanent log.
- The work queue is operational state and can be rebuilt from durable events.
- Existing pool ingestion uses snapshot import semantics rather than synthetic
  lifecycle reconstruction.
