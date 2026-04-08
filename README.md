# dr-llm

Provider-agnostic LLM primitives: call any model, browse catalogs, run batch experiments with typed sample pools.

Domain-neutral by design — shared across repos like `nl_latents` and `unitbench`.

## Two Flows

**Flow 1 — Standalone (no database):**
Call providers, sync model catalogs, browse available models. File-based catalog cache, zero infrastructure.

**Flow 2 — Pool (Postgres-backed):**
Schema-driven sample pools with no-replacement acquisition, pending sample lifecycle, and per-project isolated databases via Docker.

## Install

```bash
uv add dr-llm
```

## Quick Start

### 1. Query a provider

```bash
uv run dr-llm query \
  --provider openai \
  --model gpt-4.1 \
  --message "Hello, what's 2+2?"
```

No database needed.

### 2. List providers

```bash
uv run dr-llm providers         # human-readable table
uv run dr-llm providers --json  # machine-readable
```

### 3. Sync and browse model catalogs

```bash
uv run dr-llm models sync --provider openai
uv run dr-llm models list --provider openai
uv run dr-llm models show --provider openai --model gpt-4.1
```

Catalog data is cached locally at `~/.dr_llm/catalog_cache/`. No database required.
Human-readable and JSON model listings also include the repo's curated blacklist,
and OpenRouter listings are filtered through the local reasoning-policy allowlist.

## Available Providers

| Provider | Type | Requirements |
|---|---|---|
| `openai` | OpenAI API | `OPENAI_API_KEY` |
| `openrouter` | OpenRouter API | `OPENROUTER_API_KEY` |
| `minimax` | MiniMax Anthropic-compatible API | `MINIMAX_API_KEY` |
| `anthropic` | Anthropic API | `ANTHROPIC_API_KEY` |
| `google` | Google Gemini API | `GOOGLE_API_KEY` |
| `glm` | GLM (ZAI) API | `ZAI_API_KEY` |
| `codex` | Codex CLI (headless) | `codex` executable |
| `claude-code` | Claude Code CLI (headless) | `claude` executable |
| `kimi-code` | Kimi Code API (Anthropic-compatible) | `KIMI_API_KEY` |

Headless providers shell out to CLI tools. `minimax` and `kimi-code` are direct Anthropic-compatible `/messages` API providers.

Some providers use static model lists for `models sync` (no `/models` endpoint). The CLI notes when a list may be out of date and links to docs.

## Python API

### Calling a provider

```python
from dr_llm.llm import build_default_registry
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.messages import Message

registry = build_default_registry()
adapter = registry.get("openai")

response = adapter.generate(
    LlmRequest(
        provider="openai",
        model="gpt-4.1",
        messages=[Message(role="user", content="hello")],
    )
)
print(response.text)
```

### Filling a pool with LLM calls (requires Docker)

The recommended way to populate a pool: define `LlmConfig`s and prompts,
manually construct `PendingSample` rows whose payloads carry the serialized
config and messages, insert them into the pending queue, and let parallel
workers make the actual provider calls. Docker is used to auto-manage a
Postgres project.

```python
from itertools import product

from dr_llm import DbConfig, KeyColumn, PoolSchema, PoolStore
from dr_llm.llm import build_default_registry
from dr_llm.llm.config import LlmConfig
from dr_llm.llm.messages import Message
from dr_llm.pool.db.runtime import DbRuntime
from dr_llm.pool.llm_pool_adapter import make_llm_process_fn
from dr_llm.pool.pending.backend import PoolPendingBackend, PoolPendingBackendConfig
from dr_llm.pool.pending.pending_sample import PendingSample
from dr_llm.project.project_service import create_project
from dr_llm.workers import WorkerConfig, start_workers

# 1. Create a Docker-managed Postgres project
project = create_project("my_eval")

# 2. Define pool schema — keys are IDs, not raw values
schema = PoolSchema(
    name="my_eval",
    key_columns=[KeyColumn(name="llm_config"), KeyColumn(name="prompt")],
)
runtime = DbRuntime(DbConfig(dsn=project.dsn))
store = PoolStore(schema, runtime)
store.ensure_schema()

# 3. Define configs and prompts
llm_configs = {
    "gpt-4.1-mini": LlmConfig(
        provider="openai", model="gpt-4.1-mini", max_tokens=64,
    ),
    "gemini-flash": LlmConfig(
        provider="google", model="gemini-2.5-flash", max_tokens=64,
    ),
}
prompts = {
    "haiku": [Message(role="user", content="Write a haiku about programming.")],
    "math": [Message(role="user", content="What is 17 * 23?")],
}

# 4. Manually build PendingSample rows for the (llm_config x prompt) cross
#    product. Each payload carries the serialized LlmConfig and messages so
#    workers can reconstruct the request.
samples_per_cell = 2
pending_samples: list[PendingSample] = []
for cfg_id, prompt_id in product(llm_configs, prompts):
    payload = {
        "llm_config": llm_configs[cfg_id].model_dump(mode="json"),
        "prompt": [m.model_dump(mode="json") for m in prompts[prompt_id]],
    }
    for sample_idx in range(samples_per_cell):
        pending_samples.append(
            PendingSample(
                key_values={"llm_config": cfg_id, "prompt": prompt_id},
                sample_idx=sample_idx,
                payload=payload,
            )
        )

# 5. Insert the pending rows in one batch
store.pending.insert_many(pending_samples, ignore_conflicts=True)
# 2 configs × 2 prompts × 2 samples = 8 pending rows

# 6. Start workers — they call the real providers
registry = build_default_registry()
process_fn = make_llm_process_fn(registry)
controller = start_workers(
    PoolPendingBackend(
        store,
        config=PoolPendingBackendConfig(max_retries=1),
    ),
    process_fn=process_fn,
    config=WorkerConfig(
        num_workers=4,
        thread_name_prefix="pool-fill",
    ),
)

# 7. Wait for completion
import time
while True:
    snap = controller.snapshot()
    assert snap.backend_state is not None
    if snap.backend_state.status_counts.in_flight == 0:
        break
    time.sleep(1)
controller.stop()
controller.join()

# 8. Acquire samples (no-replacement within a run)
from dr_llm.pool.models import AcquireQuery
result = store.acquire(AcquireQuery(
    run_id="eval_run_1",
    key_values={"llm_config": "gpt-4.1-mini", "prompt": "math"},
    n=2,
))

# 9. Clean up when done
registry.close()
runtime.close()
project.destroy()
```

See `scripts/demo-pool-fill.py` for a complete runnable example.

## CLI Reference

```bash
# Providers
dr-llm providers [--json]

# Model catalog (file-based, no DB needed)
dr-llm models sync [--provider NAME] [--verbose]
dr-llm models list [--provider NAME] [--supports-reasoning] [--model-contains TEXT] [--json]
dr-llm models sync-list [--provider NAME] [--supports-reasoning] [--model-contains TEXT] [--json]
dr-llm models show --provider NAME --model NAME

# Query
dr-llm query --provider NAME --model NAME --message TEXT
dr-llm query --provider openai --model gpt-5-mini --reasoning-json '{"kind":"openai","thinking_level":"high"}' --message TEXT
dr-llm query --provider google --model gemini-2.5-flash --reasoning-json '{"kind":"google","thinking_level":"budget","budget_tokens":512}' --message TEXT
dr-llm query --provider openrouter --model openai/gpt-oss-20b --reasoning-json '{"kind":"openrouter","effort":"high"}' --message TEXT

# Projects (Docker-managed Postgres)
dr-llm project create NAME
dr-llm project list
dr-llm project use NAME
dr-llm project start|stop NAME
dr-llm project backup NAME
dr-llm project restore NAME BACKUP_PATH  # BACKUP_PATH must be .sql.gz
dr-llm project destroy NAME --yes-really-delete-everything
```

## Configuration

Generation transcript logging (default on, used for LLM call debugging):

| Variable | Default |
|---|---|
| `DR_LLM_GENERATION_LOG_ENABLED` | `true` |
| `DR_LLM_GENERATION_LOG_DIR` | `.dr_llm/generation_logs` |
| `DR_LLM_GENERATION_LOG_ROTATE_BYTES` | `104857600` (100MB) |
| `DR_LLM_GENERATION_LOG_BACKUPS` | `10` |
| `DR_LLM_GENERATION_LOG_REDACT_ENABLED` | `true` |

Provider endpoint defaults:
- GLM: `https://api.z.ai/api/coding/paas/v4`
- MiniMax API: `https://api.minimax.io/anthropic/v1/messages`
- Kimi Code API: `https://api.kimi.com/coding/v1/messages`

## Testing

```bash
uv run ruff format && uv run ruff check --fix .
uv run ty check
uv run pytest tests/ -v -m "not integration"
```

### Integration tests (requires Docker)

```bash
./scripts/run-tests-local.sh
```

`pytest` now defaults to `pytest-xdist`, so `uv run pytest tests/ -v -m "not integration"` runs the safe non-integration suite in parallel. `run-tests-local.sh` forces `-n 0`, auto-creates a temporary Docker Postgres project, runs `pytest -m integration`, and destroys it on exit. Pass extra pytest args for targeted runs: `./scripts/run-tests-local.sh -k test_pool_store`.

## Demo Scripts

### Provider discovery (no DB needed)

```bash
uv run python scripts/demo-providers.py
```

Lists all supported providers, syncs and displays model catalogs for each available one.

### Pool provider demo (requires Docker)

```bash
uv run python scripts/demo-pool-providers.py
```

Creates a project, queries every available provider, stores results in a typed pool, prints a summary table. Run with `--help` for options.

### Pool fill worker demo (requires Docker + API keys)

```bash
uv run python scripts/demo-pool-fill.py
```

Auto-creates a Docker Postgres project, manually constructs `PendingSample` rows for an `(llm_config, prompt)` pool with serialized `LlmConfig` and `Message` payloads, inserts them via `store.pending.insert_many`, starts workers that make real LLM calls via `make_llm_process_fn`, prints progress, shows response snippets, and destroys the project on exit. Pass `--dsn` to use an existing database instead. Run with `--help` for options.

### Reasoning and effort demo (live API / CLI checks)

```bash
uv run python scripts/demo_thinking_and_effort.py
```

Exercises the branch's provider-specific reasoning and effort validation against
curated model sets for OpenAI, OpenRouter, Google, Codex, Claude Code, MiniMax,
and Kimi Code. Use `--provider` to limit the run to one provider.

Reasoning configs are validated before dispatch. For example, OpenAI GPT-5
family models use configs like `{"kind":"openai","thinking_level":"high"}`,
Google 2.5 models accept budget configs like
`{"kind":"google","thinking_level":"budget","budget_tokens":512}`, `minimax`
requires `{"kind":"anthropic","thinking_level":"na"}` together with an explicit
`--effort`, `kimi-code` uses Anthropic-compatible reasoning like
`{"kind":"anthropic","thinking_level":"adaptive"}` together with an explicit
`--effort` and `--max-tokens`, and OpenRouter reasoning-capable models use
`{"kind":"openrouter", ...}` with either `enabled` or `effort` depending on the
repo's curated model policy.

See [`OPEN_ROUTER_REASONING_NOTES.md`](/Users/daniellerothermel/drotherm/repos/dr-llm/OPEN_ROUTER_REASONING_NOTES.md)
for the direct API observations that informed the OpenRouter policy layer.
