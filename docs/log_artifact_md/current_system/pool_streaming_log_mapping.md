# Current Pool System Mapping to Streaming Log Concerns

This document describes how the current pool-based system addresses the same
concerns that a future NATS JetStream logging layer would need to address.
It is not a one-to-one architecture comparison. The current system is based on
Postgres table mutation, leases, and completion state. The proposed streaming
log system would be based on append-only durable events and projections.

The purpose of this mapping is to identify which behaviors already exist as
pool semantics, which guarantees are currently provided by Postgres, and which
parts would need to be redesigned rather than ported directly.

## Event Contract

Current behavior:

Pool records are represented primarily by `PoolSample`. A sample has a
`sample_id`, pool key values, `sample_idx`, optional `run_id`, request JSON,
response JSON, finish reason, attempt count, metadata JSON, and creation time.
The physical samples table is built from the pool schema, so each pool can have
different key columns.

Current guarantees:

The samples table has a primary key on `sample_id` and a unique cell index on
the pool key columns plus `sample_idx`. Request, response, and metadata payloads
are stored as JSONB. Completion is represented by the presence of
`response_json`.

Current limitations:

There is no append-only event contract. State transitions are implicit in row
updates and deletes. Attempt identity, event identity, causality, and schema
versioning are not first-class concepts.

Implication for streaming log design:

The streaming log needs an explicit event envelope with event IDs, attempt IDs,
worker identity, event timestamps, schema versions, subject/routing metadata,
idempotency keys, and payload or artifact-reference conventions.

## Stream Topology

Current behavior:

Pool boundaries are physical database boundaries. Each pool owns separate
runtime-created samples and leases tables whose names are derived from the pool
name.

Current guarantees:

Pool isolation is simple because tables are separate. Queries, leases,
progress, and deletion are scoped to the pool's physical tables.

Current limitations:

The storage layout is coupled to the experiment boundary. Cross-pool analysis,
shared indexing, replay, and schema evolution are harder because facts are
partitioned into separate table sets.

Implication for streaming log design:

JetStream topology needs to decide how subjects, streams, and retention policies
represent experiments, runs, prompt families, datasets, and workers without
recreating one physical storage object per pool.

## Publisher API

Current behavior:

The worker-facing persistence API is `PoolStore`. It inserts samples, claims
leases, updates incomplete requests, completes samples, releases leases, resets
samples, requeues errors, and reports progress.

Current guarantees:

These operations run through Postgres transactions. Completion can be guarded by
lease ownership. Inserts can ignore conflicts. Administrative reset and requeue
operations mutate rows back to an incomplete state.

Current limitations:

The API publishes no durable facts. It mutates the current state directly. Once
a row is updated, the prior state is not recoverable from the pool tables alone.

Implication for streaming log design:

The future publisher API needs to be the worker's durable append path. It should
validate events, serialize them, publish to JetStream, wait for publish
acknowledgment, and expose clear retry and failure behavior.

## Worker Lifecycle

Current behavior:

Workers claim an incomplete sample, process it, complete it with a response or
error, and release the lease. Retry behavior releases a lease without marking
the sample complete until the retry limit is exceeded.

Current guarantees:

Incomplete work is represented by `response_json is null`. Active ownership is
represented by a lease row. Errors are terminal completions with
`finish_reason = "error"` unless requeued administratively.

Current limitations:

Lifecycle transitions are not durable events. There is no replayable history of
claims, attempts, provider calls, retries, failures, or terminal outcomes.
Generation logs record some provider-call observations, but those records do
not drive persistence.

Implication for streaming log design:

The stream needs explicit lifecycle events such as work offered or seeded,
attempt claimed, attempt started, request prepared, provider response observed,
attempt failed, retry scheduled, completion recorded, and cancellation or
administrative reset requested.

## Durability and Idempotency

Current behavior:

Durability comes from Postgres. Idempotency is handled through primary keys,
unique indexes, conflict handling, and guarded updates.

Current guarantees:

Duplicate sample insertion can be ignored. Completion only updates incomplete
rows. Lease-owned completion checks that the lease exists, belongs to the
worker, and has not expired.

Current limitations:

The idempotency model is tied to table state. There is no event-level duplicate
detection, no append-only deduplication key, and no immutable record of
conflicting or repeated actions.

Implication for streaming log design:

The log needs event-level idempotency and projection-level duplicate handling.
Projections should be able to process at-least-once deliveries while producing
logically exactly-once state.

## Consumer Checkpointing

Current behavior:

The current pool system does not have independent event consumers. Readers query
the current Postgres state directly. Some APIs stream query results with
server-side cursoring, but that is not event replay.

Current guarantees:

Query readers see committed database state. Worker coordination is controlled by
the pool tables, not by consumer offsets.

Current limitations:

There is no durable consumer offset, replay position, projection checkpoint,
redelivery policy, or dead-letter path.

Implication for streaming log design:

JetStream consumers need durable names, acknowledgment timing, replay behavior,
redelivery policy, checkpointing strategy, and dead-letter handling for failed
projection events.

## Coordination Semantics

Current behavior:

Coordination is implemented with Postgres lease rows. Claiming uses incomplete
samples, active or expired lease checks, row locking, and lease expiration.
Progress derives from counts of incomplete, complete, error, and leased rows.

Current guarantees:

Multiple workers can safely claim work without claiming the same available
sample concurrently. Expired leases can be reclaimed. Completion can require the
current worker to own an active lease.

Current limitations:

Coordination and storage are coupled. Leases are not historical facts. Progress
is current state only. Requeue and reset operations mutate state instead of
creating provenance-preserving derivations.

Implication for streaming log design:

The design must decide which coordination concepts belong in the stream and
which belong in a projection or separate work-allocation service. Claiming,
lease expiry, retries, cancellation, and progress may need different treatment
than immutable generation facts.

## Serialization and Versioning

Current behavior:

Requests, responses, and metadata are stored as JSONB in the samples table.
Pydantic models validate many Python-facing objects before storage.

Current guarantees:

The current system can store heterogeneous structured payloads, and database
queries can inspect JSONB when needed.

Current limitations:

There is no explicit event-envelope version. Large payloads live inline in
Postgres. Payload structure is partly implicit in provider and request models,
not in a stream schema.

Implication for streaming log design:

The stream needs a versioned envelope and a policy for inline payloads versus
artifact references. It also needs compatibility rules for evolving event
schemas while older events remain replayable.

## Operational Configuration

Current behavior:

Operational setup centers on Postgres. The pool store creates runtime-owned pool
tables and indexes. Local integration tests use Docker-backed Postgres helpers.

Current guarantees:

The local development path is well established for Postgres-backed pools.
Integration tests can exercise the lease, completion, deletion, reader, and
store behavior.

Current limitations:

There is no NATS service configuration, stream bootstrap, subject setup,
retention cleanup, JetStream health check, or local reset workflow.

Implication for streaming log design:

The streaming layer needs local NATS startup and configuration, stream
initialization code, environment settings, health checks, teardown/reset
commands, and CI or integration-test support.

## Test Harness

Current behavior:

The pool implementation has substantial tests for schema creation, insertion,
claiming, completion, lease expiry, release, progress, requeueing, reset,
reader behavior, and worker runtime behavior. Generation logging tests cover
JSONL writes, redaction, and thread safety.

Current guarantees:

The current tests protect the pool state machine and Postgres coordination
behavior.

Current limitations:

The tests do not cover durable event publishing, publish acknowledgment, replay,
redelivery, projection checkpointing, event deduplication, or crash recovery
from an event log.

Implication for streaming log design:

The future test harness should preserve the behavioral intent of the pool tests
while adding JetStream-specific tests for acked publish, consumer replay,
redelivery, duplicate events, projection recovery, and worker crash scenarios.

## Summary

The current pool system already answers many operational questions through
Postgres state: identity, uniqueness, claiming, completion, retries, progress,
and administrative resets. Those behaviors are useful requirements for the new
streaming-log design.

However, the mechanisms differ too much to port directly. The current system is
a mutable state store with leases. The future streaming system should be an
append-only source of truth with derived projections. The most important design
work is therefore not replacing table names with stream subjects; it is making
state transitions, attempts, retries, ownership, failures, payload references,
and projection progress explicit as durable facts.
