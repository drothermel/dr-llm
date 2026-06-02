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
| `client.py` | Publishes events, submits work, writes payloads, reads payload refs, and replays events. |
| `events.py` | Event envelope, event types, producer metadata, and idempotency helpers. |
| `work.py` | Queued work messages and worker configuration. |
| `workers.py` | Async worker loop that runs real provider requests and emits lifecycle events. |
| `ingest_pools.py` | Snapshot import of existing Postgres pools into streaming-log facts. |
| `cli.py` | `dr-llm streaming-log ...` commands. |

Shared live-demo helpers live in
[`src/dr_llm/demo/streaming_log.py`](src/dr_llm/demo/streaming_log.py). They
create temporary NATS containers when needed, isolate demo stream names, replay
events, and verify payload hashes and sizes.

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
uv run python scripts/demo-streaming-log-worker.py --provider openai --model gpt-4o-mini
uv run python scripts/demo-streaming-log-worker.py --keep-nats
uv run python scripts/demo-streaming-log-worker.py --nats-url nats://127.0.0.1:4222
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
