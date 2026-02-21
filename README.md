# llm-pool

`llm-pool` is a shared primitive for:
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
uv sync
```

## Configuration

- Required for DB-backed workflows: `LLM_POOL_DATABASE_URL`
- Optional provider keys:
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `GOOGLE_API_KEY`
  - `GLM_API_KEY`

## CLI

```bash
llm-pool providers

llm-pool query \
  --provider openai \
  --model gpt-4.1 \
  --reasoning-json '{"effort":"high"}' \
  --message "hello"

llm-pool run start --run-type benchmark
llm-pool run finish --run-id <run_id> --status success

llm-pool session start \
  --provider openai \
  --model gpt-4.1 \
  --message "You are helpful" \
  --message "Solve this task"

llm-pool session step --session-id <session_id> --message "next"
llm-pool session resume --session-id <session_id>
llm-pool session cancel --session-id <session_id> --reason "stopped"

# brokered tool calls are queued by default; use workers
llm-pool tool worker run --tool-loader mypkg.tools:register_tools
# optional synchronous override for a single step:
llm-pool session step --session-id <session_id> --inline-tool-execution

llm-pool replay session --session-id <session_id>
```

Reasoning + cost notes:
- OpenAI-compatible adapters now accept `LlmRequest.reasoning` / `--reasoning-json`.
- Reasoning text/details and reasoning token counts are normalized on `LlmResponse`.
- Provider-returned cost fields (e.g. OpenRouter `usage.cost` variants) are normalized into `LlmResponse.cost`.
- These are persisted in `llm_call_responses` alongside standard token usage.

## Python Example

```python
from llm_pool import LlmClient, LlmRequest, Message, PostgresRepository, ToolRegistry

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

## Testing

```bash
uv run ruff check src tests
uv run pytest
```

Postgres integration tests are env-gated:
- set `LLM_POOL_TEST_DATABASE_URL` (or `LLM_POOL_DATABASE_URL`)
- run `uv run pytest -m integration`
