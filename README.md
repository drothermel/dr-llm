# dr-llm

`dr-llm` is a shared primitive for:
- provider-agnostic LLM calls (API and headless)
- canonical PostgreSQL recording/query storage
- event-sourced multistep sessions with tool-calling
- worker-safe parallel tool execution with queue claiming
- generic typed sample pools for benchmark storage
- isolated per-project databases with backup/restore

It is intentionally domain-neutral so repos like `nl_latents` and `unitbench` can reuse it.

## Core Capabilities

- Unified call interface:
  - `LlmClient.query(LlmRequest) -> LlmResponse`
- Canonical storage (PostgreSQL):
  - runs, calls, request/response payloads, artifacts
- Session runtime:
  - `start`, `step`, `resume`, `cancel`
  - native tool strategy (if provider supports tools) + brokered fallback
- Tool queue + workers:
  - idempotent tool call enqueue
  - concurrent worker claims via `FOR UPDATE SKIP LOCKED`
- Replay:
  - reconstruct message history from `session_events`
- Sample pools:
  - schema-driven typed key dimensions with auto-generated DDL
  - no-replacement acquisition via claims table
  - pending sample lifecycle (claim/promote/fail with `FOR UPDATE SKIP LOCKED`)
  - top-up orchestration: acquire, wait for pending, generate, re-acquire
- Project management:
  - isolated per-project Postgres containers via Docker
  - backup/restore with atomic swap

## Install

```bash
uv add dr-llm
```

Quick verification:

```bash
uv run python -c "import dr_llm"
```

For maintainers, see the release runbook: `docs/releasing.md`.

## Quick Start

### 1. Query a provider (no database required)

```bash
uv run dr-llm query \
  --provider openai \
  --model gpt-4.1 \
  --message "Hello, what's 2+2?" \
  --no-record
```

The `--no-record` flag skips database recording, so you can test providers without Postgres.

### 2. Inspect supported and available providers

```bash
uv run dr-llm providers
uv run dr-llm providers --json
```

`dr-llm providers` renders a human-readable table showing canonical provider names, whether each provider is currently available on this machine, and any missing local requirements. Use `--json` when you want a scriptable output format.

### 3. Start Postgres (for catalog and recording)

```bash
source ./scripts/start-test-postgres.sh
```

This starts a local Postgres container, applies schema migrations, and exports `DR_LLM_DATABASE_URL` and `DR_LLM_TEST_DATABASE_URL` into your shell. Use `source` (not `./`) so the env vars persist.

### 4. Sync and list models

```bash
uv run dr-llm models sync --provider openai
uv run dr-llm models list --provider openai
```

`models sync` now prints a single concise success line by default. Use `--verbose` when you want the full per-provider JSON sync payload. `models list` prints a human-readable header plus bulleted model names by default, and shows 20 results unless you override `--limit`; use `--json` for full metadata output.

### Available Providers

| Provider | Type | Local Requirements |
|---|---|---|
| `openai` | OpenAI API | `OPENAI_API_KEY` |
| `openrouter` | OpenRouter API | `OPENROUTER_API_KEY` |
| `minimax` | MiniMax OpenAI-compat API | `MINIMAX_API_KEY` |
| `anthropic` | Anthropic API | `ANTHROPIC_API_KEY` |
| `google` | Google Gemini API | `GOOGLE_API_KEY` |
| `glm` | GLM (ZAI) API | `ZAI_API_KEY` |
| `codex` | Codex CLI (headless) | `codex` executable |
| `claude-code` | Claude Code CLI (headless) | `claude` executable |
| `claude-code-minimax` | Claude Code via MiniMax | `claude` executable + `MINIMAX_API_KEY` |
| `claude-code-kimi` | Claude Code via Kimi | `claude` executable + `KIMI_API_KEY` |

Headless providers shell out to CLI tools (`codex`, `claude`). The MiniMax and Kimi variants point Claude Code at third-party Anthropic-compatible endpoints and require their corresponding API keys.

Run `uv run dr-llm providers` to see which of the supported providers are currently usable in your shell.

Some providers (MiniMax, Codex, Claude Code, Kimi) use static model lists for `models sync` since they don't expose a `/models` endpoint. The CLI will note when a list may be out of date and link to the provider's docs.

## Configuration

- Required for DB-backed workflows: `DR_LLM_DATABASE_URL`
- Provider API keys: see the table above
- GLM provider defaults to: `https://api.z.ai/api/coding/paas/v4`
- MiniMax API provider defaults to: `https://api.minimax.io/v1`
- Claude headless coding-plan presets:
  - `claude-code-minimax`: routes via `https://api.minimax.io/anthropic`
  - `claude-code-kimi`: routes via `https://api.kimi.com/coding/`

## CLI Reference

```bash
dr-llm providers
dr-llm providers --json

dr-llm models sync
dr-llm models sync --provider openai --verbose
dr-llm models list --provider openai
dr-llm models list --supports-reasoning --json
dr-llm models show --provider openrouter --model openai/o3-mini

dr-llm query \
  --provider openai \
  --model gpt-4.1 \
  --message "hello" \
  --no-record

dr-llm query \
  --provider openai \
  --model gpt-4.1 \
  --reasoning-json '{"effort":"high"}' \
  --message "hello"

dr-llm run start --run-type benchmark
dr-llm run finish --run-id <run_id> --status success
dr-llm run benchmark \
  --workers 128 \
  --total-operations 200000 \
  --warmup-operations 10000 \
  --max-in-flight 128 \
  --operation-mix-json '{"record_call":2,"session_roundtrip":1,"read_calls":1}' \
  --artifact-path .dr_llm/benchmarks/release-baseline.json

dr-llm session start \
  --provider openai \
  --model gpt-4.1 \
  --message "You are helpful" \
  --message "Solve this task"

dr-llm session step --session-id <session_id> --message "next"
dr-llm session resume --session-id <session_id>
dr-llm session cancel --session-id <session_id> --reason "stopped"

# brokered tool calls are queued by default; use workers
dr-llm tool worker run --tool-loader mypkg.tools:register_tools
# optional synchronous override for a single step:
dr-llm session step --session-id <session_id> --inline-tool-execution

dr-llm replay session --session-id <session_id>

dr-llm project create my-experiment
dr-llm project list
dr-llm project use my-experiment    # prints export DR_LLM_DATABASE_URL=...
dr-llm project start my-experiment
dr-llm project stop my-experiment
dr-llm project backup my-experiment
dr-llm project restore my-experiment backups/my-experiment-20260325.sql.gz
dr-llm project destroy my-experiment --yes-really-delete-everything
```

Benchmark command output:
```json
{
  "artifact_path": ".dr_llm/benchmarks/release-baseline.json",
  "failed_operations": 0,
  "operations_per_second": 4231.8,
  "p50_latency_ms": 20.0,
  "p95_latency_ms": 200.0,
  "run_id": "run_abc123",
  "status": "success"
}
```

Reasoning + cost notes:
- OpenAI-compatible adapters now accept `LlmRequest.reasoning` / `--reasoning-json`.
- Reasoning text/details and reasoning token counts are normalized on `LlmResponse`.
- Provider-returned cost fields (e.g. OpenRouter `usage.cost` variants) are normalized into `LlmResponse.cost`.
- These are persisted in `llm_call_responses` alongside standard token usage.

Generation transcript logging (default on):
- `DR_LLM_GENERATION_LOG_ENABLED=true`
- `DR_LLM_GENERATION_LOG_DIR=.dr_llm/generation_logs`
- `DR_LLM_GENERATION_LOG_ROTATE_BYTES=104857600`
- `DR_LLM_GENERATION_LOG_BACKUPS=10`
- `DR_LLM_GENERATION_LOG_REDACT_SECRETS=true`
- `DR_LLM_GENERATION_LOG_MAX_EVENT_BYTES=10485760`

## Python Example

```python
from dr_llm import LlmClient, LlmRequest, Message, PostgresRepository, ToolRegistry

repo = PostgresRepository()
client = LlmClient(repository=repo)

response = client.query(
    LlmRequest(
        provider="openai",
        model="gpt-4.1",
        messages=[Message(role="user", content="hello")],
    )
)
print(response.text)
```

Adapter lifecycle note:
- If you instantiate provider adapters directly, call `adapter.close()` when done (or use context manager form `with ... as adapter:`) to release underlying HTTP connections.

## Pool Example

Pools provide schema-driven sample storage with no-replacement acquisition for benchmarks.

```python
from dr_llm import (
    ColumnType, KeyColumn, PoolSchema, PoolStore, PoolService,
    PoolAcquireQuery, PoolAcquireResult,
)
from dr_llm.pool.models import PoolSample
from dr_llm.storage._runtime import StorageConfig, StorageRuntime

# 1. Declare a pool schema with typed key dimensions
schema = PoolSchema(
    name="my_benchmark",
    key_columns=[
        KeyColumn(name="provider"),
        KeyColumn(name="difficulty", type=ColumnType.integer),
    ],
)

# 2. Connect and create tables
runtime = StorageRuntime(StorageConfig(dsn="postgresql://..."))
store = PoolStore(schema, runtime)
store.init_schema()  # idempotent CREATE TABLE IF NOT EXISTS

# 3. Insert samples
store.insert_samples([
    PoolSample(
        key_values={"provider": "openai", "difficulty": 1},
        sample_idx=0,
        payload={"prompt": "What is 2+2?", "expected": "4"},
    ),
    PoolSample(
        key_values={"provider": "openai", "difficulty": 1},
        sample_idx=1,
        payload={"prompt": "What is 3+3?", "expected": "6"},
    ),
])

# 4. Acquire samples (no-replacement within a run)
result = store.acquire(PoolAcquireQuery(
    run_id="run_001",
    key_values={"provider": "openai", "difficulty": 1},
    n=2,
))
for sample in result.samples:
    print(sample.payload)

# 5. Or use PoolService for automatic top-up generation
service = PoolService(store)
result = service.acquire_or_generate(
    PoolAcquireQuery(
        run_id="run_002",
        key_values={"provider": "openai", "difficulty": 2},
        n=5,
    ),
    generator_fn=lambda key_values, deficit: [
        PoolSample(key_values=key_values, payload={"generated": True})
        for _ in range(deficit)
    ],
)
```

## Testing

```bash
uv run ruff format
uv run ruff check --fix .
uv run ty check
uv run pytest tests/ -v
```

### Integration tests

Integration tests require a running Postgres instance. If the test container (`dr-llm-pg-test` on port 5433) is already running, tests work automatically — `conftest.py` sets the default `DR_LLM_TEST_DATABASE_URL`.

To start the test container from scratch:

```bash
source ./scripts/start-test-postgres.sh
```

Then run integration tests:

```bash
uv run pytest tests/ -v -m integration
```

If integration tests are skipped unexpectedly, include skip reasons:
```bash
uv run pytest tests/ -v -m integration -rs
```

## Demo Scripts

### End-to-end query flow

Creates a project, records queries, verifies backup/restore:

```bash
./scripts/demo-query-flow.sh
```

Requires Docker and at least one of `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.

### Provider discovery demo

Shows all supported canonical providers and which of them are currently available on this machine:

```bash
source ./scripts/start-test-postgres.sh
uv run python scripts/demo-providers.py
```

`scripts/demo-providers.py` exits early if `DR_LLM_DATABASE_URL` is unset, so start the local test Postgres first with `source ./scripts/start-test-postgres.sh` or otherwise export `DR_LLM_DATABASE_URL` before running the demo.

### Pool provider demo

Queries all available LLM providers (API and headless) and stores results in a typed pool:

```bash
uv run python scripts/demo-pool-providers.py
```

Auto-detects available providers by checking API key env vars and CLI tool availability (`claude`, `codex`). For each provider: syncs the model catalog, selects a model, sends a query, and inserts the result into a pool. Prints a summary table at the end.

Options:
```bash
uv run python scripts/demo-pool-providers.py --project-name my-demo --prompt "Explain gravity in one sentence."
```

Requires Docker. Works with any combination of providers — set API keys and/or install CLI tools for the providers you want to test.
