# dr-llm

`dr-llm` is a shared primitive for:
- provider-agnostic LLM calls (API and headless)
- canonical PostgreSQL recording/query storage
- event-sourced multistep sessions with tool-calling
- worker-safe parallel tool execution with queue claiming

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

### 2. Start Postgres (for catalog and recording)

```bash
source ./scripts/start-integration-postgres.sh
```

This starts a local Postgres container, applies schema migrations, and exports `DR_LLM_DATABASE_URL` and `DR_LLM_TEST_DATABASE_URL` into your shell. Use `source` (not `./`) so the env vars persist.

### 3. Sync and list models

```bash
uv run dr-llm models sync --provider openai
uv run dr-llm models list --provider openai
```

Use `--json` on `models list` for full metadata output.

### Available Providers

| Provider | Aliases | Type | API Key Env |
|---|---|---|---|
| `openai` | — | OpenAI API | `OPENAI_API_KEY` |
| `openrouter` | — | OpenRouter API | `OPENROUTER_API_KEY` |
| `minimax` | — | MiniMax OpenAI-compat API | `MINIMAX_API_KEY` |
| `anthropic` | — | Anthropic API | `ANTHROPIC_API_KEY` |
| `google` | — | Google Gemini API | `GOOGLE_API_KEY` |
| `glm` | — | GLM (ZAI) API | `ZAI_API_KEY` |
| `codex` | `codex-cli` | Codex CLI (headless) | `OPENAI_API_KEY` |
| `claude-code` | `claude` | Claude Code CLI (headless) | `ANTHROPIC_API_KEY` |
| `claude-code-minimax` | `claude-minimax` | Claude Code via MiniMax | `MINIMAX_API_KEY` |
| `claude-code-kimi` | `claude-kimi` | Claude Code via Kimi | `KIMI_API_KEY` |

Headless providers shell out to CLI tools (`codex`, `claude`) and redirect API traffic via environment variables. The MiniMax and Kimi variants point Claude Code at third-party Anthropic-compatible endpoints.

Some providers (MiniMax, Codex, Claude Code, Kimi) use static model lists for `models sync` since they don't expose a `/models` endpoint. The CLI will note when a list may be out of date and link to the provider's docs.

## Configuration

- Required for DB-backed workflows: `DR_LLM_DATABASE_URL`
- Provider API keys: see the table above
- GLM provider defaults to: `https://api.z.ai/api/coding/paas/v4`
- MiniMax API provider defaults to: `https://api.minimax.io/v1`
- Claude headless coding-plan presets:
  - `claude-code-minimax`: routes via `https://api.minimax.io/anthropic`
  - `claude-code-kimi`: routes via `https://api.kimi.com/coding/`
- Model catalog overrides: `config/model_overrides.json` (YAML also supported)

## CLI Reference

```bash
dr-llm providers

dr-llm models sync
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

## Testing

```bash
uv run ruff format
uv run ruff check --fix src/ tests/ scripts/
uv run ty check src
uv run pytest tests/ -v
```

Postgres integration tests are env-gated. Start the local test database and run:

```bash
source ./scripts/start-integration-postgres.sh
uv run pytest tests/ -v -m integration
```

The script starts a Postgres container on port 5433, applies migrations, and exports the required env vars.

If integration tests are skipped unexpectedly, include skip reasons:
```bash
uv run pytest tests/ -v -m integration -rs
```

## CI

GitHub Actions workflows:
- `ci`: runs on PRs to `main` and pushes to `main`
  - `quality-unit` job: format check, lint, type-check, non-integration tests
  - `security` job: `uv lock --check` and `uvx pip-audit`
- `integration`: runs on pushes to `main`, manual dispatch, and PRs to `main` only when label `run-integration` is present
  - starts `postgres:16` service and runs `pytest -m integration`

Branch protection recommendation:
- require `ci / quality-unit`
- require `ci / security`
- keep `integration / postgres-integration` non-required for all PRs (opt-in via label, always on `main`)

## Milestone Closeout Artifacts

- Milestone status: `docs/milestones.md`
- Consumer rollout checklist: `docs/consumer-rollout-checklist.md`
- M2b operations checklist: `docs/ops/m2b-hardening-checklist.md`
- Compatibility contract: `docs/compatibility-contract.md`
- Migration guide: `docs/migration-guide.md`
- Integration notes:
  - `docs/integrations/nl_latents.md`
  - `docs/integrations/unitbench.md`
- Example gateways:
  - `examples/nl_latents_gateway.py`
  - `examples/unitbench_gateway.py`
