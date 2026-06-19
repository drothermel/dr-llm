# dr-llm Streaming Log

`dr-llm` now has a NATS JetStream-backed streaming log for recording LLM
production work, replaying execution events, and importing existing
Postgres-backed pools as reconstructed historical facts.

The previous pool-oriented README has moved to
[`POOL_README.md`](POOL_README.md).

## What This Adds

The streaming log is the new event backbone for `dr-llm`:

- work messages are queued in JetStream and consumed by async workers
- producer, attempt, provider, completion, and pool-import facts are published
  as append-only events
- large request/response/schema payloads are stored in JetStream Object Store
  and referenced from events by content hash
- existing pools can be imported into the log without pretending they were
  produced by the new worker path
- event replay can verify lifecycle ordering, payload references, and imported
  sample counts against real data

Core implementation lives under
[`src/dr_llm/streaming_log/`](src/dr_llm/streaming_log/):

| Module | Purpose |
|---|---|
| `config.py` | NATS URL, stream names, subjects, consumers, and payload bucket settings. |
| `bootstrap.py` | Creates and inspects JetStream streams, consumers, and object buckets. |
| `client.py` | Explicit streaming-log primitives for NATS connection, payload storage, event publishing/replay, and work queues. |
| `events.py` | Event envelope, event types, producer metadata, and idempotency helpers. |
| `serialization.py` | Canonical JSON byte serialization shared by event hashing, event publishing, and JSON payload storage. |
| `payloads.py` | Payload hashing, object key construction, prepared payloads, and typed payload references. |
| `work.py` | Queued work messages. |
| `workers.py` | Public worker primitives plus the async JetStream worker entrypoint. |
| `ingest_pools.py` | Snapshot import of existing Postgres pools into streaming-log facts. |
| `cli.py` | `dr-llm streaming-log ...` commands. |

Shared live-demo helpers live in
[`src/dr_llm/demo/streaming_log.py`](src/dr_llm/demo/streaming_log.py). They
create temporary NATS containers when needed, isolate demo stream names,
bootstrap resources, expose connected event/payload/work clients through
`open_streaming_log_demo_runtime(...)`, replay events, and verify payload
hashes and sizes.

## Publishing Events

Event envelopes keep workflow identity fields such as `run_id`, `work_id`,
`attempt_id`, `correlation_id`, and `source` at the top level for replay and
projection. Callers should not repeat those fields on every publish call.
Instead, build an `EventContext` once for the workflow step and publish through
a context-bound publisher:

```python
async with StreamingLogConnection(config) as connection:
    payload_store = StreamingPayloadStore(connection)
    event_log = StreamingEventLog(connection, payload_store)

    context = EventContext.from_work_attempt(work, attempt_id=attempt_id)
    publisher = event_log.with_event_context(context)

    await publisher.publish_event_spec(
        StreamingEventPublishSpec(
            event_type=StreamingLogEventType.attempt_started,
            idempotency_key=idempotency_key(
                "attempt_started", work.work_id, attempt
            ),
            payload={"worker_id": worker_id, "attempt": attempt},
        )
    )
```

Use `event_log.publish_event_spec(...)` directly for context-free
operational events such as producer startup/shutdown. The event wire shape is
unchanged: `EventContext` is only the construction primitive for shared
envelope identity.

JSON event bytes, idempotency hashes, and JSON payload bytes all use the shared
`canonical_json_bytes(...)` primitive. Payload references on an
`EventEnvelope` are validated `PayloadRef` models, so replay consumers can use
`event.payload_refs` directly instead of reparsing raw dictionaries.

## Streaming-Log Primitives

The streaming log intentionally uses explicit clients rather than one broad
facade:

```python
async with StreamingLogConnection(config) as connection:
    payload_store = StreamingPayloadStore(connection)
    event_log = StreamingEventLog(connection, payload_store)
    work_queue = StreamingWorkQueue(connection, event_log)

    await work_queue.submit_work(work)
    stored = await payload_store.read_payload_ref(payload_ref)
```

`StreamingLogConnection` owns NATS lifecycle and JetStream access.
`StreamingPayloadStore` owns Object Store payload writes and reads.
`StreamingEventLog` owns event construction, publishing, subscriptions, and
replay. `StreamingWorkQueue` owns work submission and work-message fetches.

## Worker Primitives

`run_streaming_worker(...)` remains the CLI and demo entrypoint, but worker
processing is built from public primitives so future systems can reuse or
replace one concern at a time:

- `StreamingWorkAttempt` decodes a queued message into work identity and
  delivery-attempt metadata.
- `StreamingWorkProcessor` coordinates one attempt at the workflow level.
- `StreamingWorkExecutor` and `ProviderRegistryStreamingWorkExecutor` isolate
  provider execution from queue handling.
- `StreamingRetryPolicy` converts provider failures into explicit
  `StreamingWorkRetryScheduled` or `StreamingWorkFailed` outcomes.
- `StreamingWorkSucceeded`, `StreamingWorkRetryScheduled`, and
  `StreamingWorkFailed` are the concrete outcome primitives. Use
  `StreamingWorkOutcome` for the full union and
  `StreamingWorkFailureOutcome` for retry-or-failed consumers.
- `StreamingEventPublisher` and `StreamingWorkLifecycleReporter` isolate
  attempt, provider, retry, and completion event emission.
- `StreamingMessageAcknowledger` applies the final `ack` or `nak` decision.
- `StreamingWorkMessageHandler` wires those pieces for one JetStream message.

## Requirements

- Python environment managed by `uv`
- Docker, unless you pass `--nats-url` for an existing NATS JetStream server
- For pool import: an existing Postgres-backed `dr-llm` pool
- For worker execution: at least one real provider available through API keys
  or supported CLI tools

Provider availability is detected from the existing provider registry. The
worker demo does not mock provider calls.

## Streaming-Log CLI

The CLI uses `StreamingLogConfig()` defaults. Set environment variables or use
the Python/demo helpers when you need isolated resource names.

```bash
uv run dr-llm streaming-log bootstrap
uv run dr-llm streaming-log inspect

uv run dr-llm streaming-log ingest-pool \
  --dsn postgresql://postgres:postgres@localhost:5504/dr_llm \
  --pool-name decoder_t1_smoke_3 \
  --sample-limit 3

uv run dr-llm streaming-log run-worker --max-messages 1
```

`ingest-pool` and `ingest-pools` accept `--sample-limit` for quick live checks
against large existing pools. The limit is real ingestion behavior: only that
many source samples are emitted as `pool_sample_imported` events.

## Live Demo Scripts

The demo scripts are the recommended end-to-end verification path because they
use isolated stream names on every run and clean up temporary NATS containers.

### Sync A Project To Postgres

This command creates a temporary Docker Postgres project, seeds a small typed
pool in the source database, syncs it into a separate Postgres-compatible target
database, and reads the synced pool back:

```bash
uv run python scripts/demo-project-sync-postgres.py
```

Prerequisites: Docker must be running and `psql` must be available on `PATH`.

Useful options:

```bash
uv run python scripts/demo-project-sync-postgres.py --help
uv run python scripts/demo-project-sync-postgres.py --keep-projects
```

### Import An Existing Pool

This command should work with the currently running local `code_comp_t1`
project and its `decoder_t1_smoke_3` pool:

```bash
uv run python scripts/demo-streaming-log-pool-import.py \
  --dsn postgresql://postgres:postgres@localhost:5504/dr_llm \
  --pool-name decoder_t1_smoke_3 \
  --sample-limit 3
```

What it verifies:

- the source pool opens and reports real progress counts
- a NATS JetStream server is reachable or created through Docker
- isolated event/work streams and payload buckets bootstrap successfully
- pool import emits `pool_import_started`, `pool_sample_imported`, and
  `pool_import_completed`
- source snapshot failures attempt `pool_import_failed`; if that failure event
  cannot be published, the source error stays primary and the publish failure is
  logged and attached to the exception
- replayed event counts match the imported source rows
- every payload reference can be read back and matches its recorded SHA-256 and
  byte size

Useful options:

```bash
uv run python scripts/demo-streaming-log-pool-import.py --help
uv run python scripts/demo-streaming-log-pool-import.py \
  --dsn postgresql://postgres:postgres@localhost:5504/dr_llm \
  --pool-name decoder_t1_smoke_3 \
  --sample-limit 1 \
  --keep-nats
```

Use `--nats-url nats://host:4222` to target an existing NATS server instead of
starting a temporary Docker container.

### Run One Live Worker Message

```bash
uv run python scripts/demo-streaming-log-worker.py \
  --provider openai \
  --model gpt-4o-mini \
  --prompt "What is 2+2?" \
  --max-retries 0
```

What it verifies:

- provider availability is detected from real environment variables and CLI
  tools
- a model catalog is synced and a real model is selected
- auto-selected providers fall back gracefully if a configured provider fails at
  execution time, for example because of billing or account limits
- one work message is submitted to JetStream
- the streaming worker processes that message through the provider
- replay includes the expected lifecycle events:
  `work_submitted`, `producer_started`, `attempt_started`,
  `provider_request_prepared`, `provider_response_received`,
  `attempt_succeeded`, `work_completed`, and `producer_stopped`
- payload references are readable and hash-verified

Useful options:

```bash
uv run python scripts/demo-streaming-log-worker.py --help
uv run python scripts/demo-streaming-log-worker.py
uv run python scripts/demo-streaming-log-worker.py --provider openai --model gpt-4o-mini
uv run python scripts/demo-streaming-log-worker.py --keep-nats
uv run python scripts/demo-streaming-log-worker.py --nats-url nats://127.0.0.1:4222
```

Fallback only applies to auto-selected providers. If you pass `--provider` or
`--model`, that explicit choice is tested directly and failures are reported
with the replayed `attempt_failed` details.

### Project Artifacts From The Streaming Log

```bash
uv run python scripts/demo-artifact-projection.py
```

What it verifies:

- isolated streaming-log resources bootstrap successfully
- one event with duplicate artifact payload refs projects to one artifact
- the artifact sidecar records no open references after finalization
- the finalized artifact can be read back from the Zarr shard

Useful options:

```bash
uv run python scripts/demo-artifact-projection.py --help
uv run python scripts/demo-artifact-projection.py --keep-nats
uv run python scripts/demo-artifact-projection.py --artifact-root /tmp/dr-llm-artifacts
```

## Local Project And Pool Discovery

Use the existing pool commands to find a live source pool:

```bash
uv run dr-llm project list
uv run dr-llm pool list-dsn --dsn postgresql://postgres:postgres@localhost:5504/dr_llm
```

Then pass the discovered DSN and pool name to
`scripts/demo-streaming-log-pool-import.py`. Prefer a smoke or demo pool and a
small `--sample-limit` while debugging.

## Testing

Recommended quality gate:

```bash
uv run ruff format
uv run ruff check --fix .
uv run ty check
uv run pytest tests/ -v -m "not integration"
./scripts/run-tests-local.sh
```

The local integration runner starts temporary Postgres and NATS resources, runs
integration tests, and cleans up after itself.
