# Streaming Log

The streaming log is the durable ingestion layer. Collection workers append
events to a persistent log and receive acknowledgment when those events are
durably accepted. Downstream consumers then project the events into artifact
storage, metadata storage, validation tables, and future analysis indexes.

This changes the current `dr-llm` flow. Today, pool workers claim rows from
Postgres, call a provider, and write completion data back to the same pool
tables. In the new architecture, the worker's primary responsibility is to
produce valid events. Storage and analysis systems consume those events
independently.

The proposed implementation target is NATS JetStream. It is appealing because
it provides durable streams, replay, consumer acknowledgment, independent
durable consumers, redelivery on failure, and support for many producers while
remaining lighter operationally than a Kafka-style stack. That fits a
local-first setup running on a powerful development machine while preserving
patterns that can scale to larger distributed systems.

The log provides the recovery model:

- If a metadata projection changes, it can be rebuilt by replaying events.
- If the artifact layout changes, a new artifact projector can consume the
  same event history.
- If a downstream consumer fails, it can resume from its last acknowledged
  position.

The current repo already has generation log events around provider calls, but
those logs are observational JSONL records. In this architecture, the streaming
log becomes the persistent source of truth for collected facts.

Event producers should bind repeated workflow identity through `EventContext`
instead of passing envelope fields on every publish call. This keeps
event-specific payloads explicit while making shared run, work, attempt,
correlation, and source fields a single construction concern.

Event-specific wire shape should live in public event builder primitives.
Reporters and import recorders should decide which lifecycle facts happened,
then publish builder-produced specs rather than constructing inline payloads,
payload references, idempotency keys, and metadata in the orchestration method.

Workflow-specific reporters should own lifecycle event emission. Source readers
should remain log-agnostic so snapshot acquisition and event recording can be
reused independently by future import and projection systems.
