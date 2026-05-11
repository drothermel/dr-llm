# dr-llm

Provider-agnostic LLM primitives: call any model, browse catalogs, run batch experiments with typed sample pools.

Domain-neutral by design — shared across repos like `nl_latents` and `unitbench`.

## Two Flows

**Flow 1 — Standalone (no database):**
Call providers, sync model catalogs, browse available models. File-based catalog cache, zero infrastructure.

**Flow 2 — Pool (Postgres-backed):**
Schema-driven sample pools with a unified two-table design
(`pool_<name>_samples` + `pool_<name>_leases`), no-replacement acquisition,
and per-project isolated databases via Docker.

## Install

```bash
uv add dr-llm
```

For the optional marimo pool-inspection notebooks in
[`nbs/inspect/`](nbs/inspect), install the notebook extra:

```bash
uv add "dr-llm[notebooks]"
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
and provider orchestrators own any provider-specific catalog policy, such as
OpenRouter's reasoning-policy allowlist.

## Demo Scripts

The README gives the mental model and short commands. The demo scripts are the
source of truth for complete runnable workflows, including exact imports,
provider/model choices, setup, cleanup, and progress output.

| Script | Use it for | Requirements |
|---|---|---|
| `scripts/demo-providers.py` | Discover providers and sync/list model catalogs. | API keys or CLI tools for the providers you want to query. No database. |
| `scripts/demo-pool-providers.py` | Query every available provider and store one result per provider/model in a typed pool. | Docker plus at least one API key or supported CLI tool. |
| `scripts/demo-pool-fill.py` | Seed an `(llm_config, prompt)` grid, fill it with workers, and inspect stored responses. | OpenAI/Google API keys, plus Docker or `--dsn` for Postgres. |
| `scripts/demo_thinking_and_effort.py` | Live-check provider-specific reasoning and effort validation. | API keys or CLI tools for the providers under test. |

```bash
uv run python scripts/demo-providers.py
uv run python scripts/demo-pool-providers.py --help
uv run python scripts/demo-pool-fill.py --help
uv run python scripts/demo_thinking_and_effort.py --provider openai
```

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

Headless providers shell out to CLI tools. `minimax` and `kimi-code` are direct Anthropic-compatible `/messages` API providers. Headless input shapes do not expose `temperature`, `top_p`, or `max_tokens`. `kimi-code` rejects `temperature` and `top_p`; its orchestrator supplies the provider max-token default when callers omit it.

Some provider orchestrators use static fallback catalogs when a provider has no
`/models` endpoint or live discovery is unavailable. The CLI notes when a list
may be out of date and links to docs.

## Python API

The Python API exposes the same provider and pool primitives used by the demo
scripts. Keep README examples small; use the demos above for maintained
end-to-end workflows.

`LlmConfig` and `LlmRequest` are the shared runtime shapes for all providers.
They carry `provider`, `model`, `mode`, reasoning, effort, token limits, and
optional nested `SamplingControls`. Provider-specific authoring configs such as
`OpenAIGpt5Config`, `AnthropicBudgetConfig`, `GoogleBudgetConfig`, and
`CodexGpt54Config` encode provider and model-family constraints, then serialize
to the common `LlmConfig` shape with `.to_llm_config()`.

Provider orchestrators construct requests from stored configs or caller inputs. They apply provider defaults for effort, reasoning, max tokens, and sampling controls before generation. For generic sampling-capable API providers, omitted sampling controls default to `temperature=1.0` and `top_p=0.95`. OpenAI omits those fields unless you set them explicitly. `kimi-code` and headless providers reject those fields entirely.

Use `build_default_registry().get(provider).request_defaults(model)` when
inspecting generic provider requests. It returns the orchestrator-owned defaults
for effort, reasoning, token limits, and supported sampling controls.

### Calling a provider

```python
from dr_llm.llm import Message, build_default_registry

registry = build_default_registry()
orchestrator = registry.get("openai")

response = orchestrator.generate(
    orchestrator.build_request(
        model="gpt-4.1",
        messages=[Message(role="user", content="hello")],
    )
)
print(response.text)
```

### Pool workflows

The recommended way to populate a pool: declare each variant axis (LLM
configs, prompts, datasets, ...), pass them to `seed_llm_grid`, and let
parallel workers make the provider calls. `seed_llm_grid` walks the cross
product, builds per-cell payloads in the shape `make_llm_process_fn` consumes,
and bulk-inserts the unfilled sample rows in one round-trip. After starting
workers, use `drain_pool` to wait until the pool has no incomplete rows while
emitting progress snapshots.

Run the maintained worker example instead of copying a README-sized snippet:

```bash
uv run python scripts/demo-pool-fill.py
uv run python scripts/demo-pool-fill.py --dsn postgresql://postgres:postgres@localhost:5433/dr_llm_test
```

For reading pools, `PoolReader.open(pool, runtime=runtime)` loads the pool's
persisted `PoolSchema` from `pool_catalog` and exposes read-side methods such as
`progress()` and `samples_list(...)`. For fair worker scheduling,
`RoundRobinClaimer` can interleave claims across an explicit key dimension while
still relying on `PoolStore.claim_lease(...)` for lease safety.

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
dr-llm query --provider codex --model gpt-5.1-codex-mini --reasoning-json '{"kind":"codex","thinking_level":"xhigh"}' --message TEXT
dr-llm query --provider google --model gemini-2.5-flash --reasoning-json '{"kind":"google","thinking_level":"budget","budget_tokens":512}' --message TEXT
dr-llm query --provider openrouter --model openai/gpt-oss-20b --reasoning-json '{"kind":"openrouter","effort":"high"}' --message TEXT

# Sampling / token controls
# Generic sampling API providers default omitted sampling controls to temperature=1.0 and top_p=0.95.
# OpenAI omits temperature/top_p unless you set them explicitly.
# OpenAI GPT-5 custom temperature/top_p controls are only supported on gpt-5.2/gpt-5.4 with reasoning off.
# --temperature, --top-p, and --max-tokens are rejected for headless providers (codex, claude-code)
# --temperature and --top-p are also rejected for kimi-code; its orchestrator supplies max-token defaults

# Projects (Docker-managed Postgres)
dr-llm project create NAME
dr-llm project list
dr-llm project use NAME
dr-llm project start|stop NAME
dr-llm pool destroy PROJECT_NAME POOL_NAME --yes-really-delete-everything
dr-llm pool destroy-testish PROJECT_NAME --yes-really-delete-everything
dr-llm pool destroy-testish PROJECT_NAME --dry-run
dr-llm project backup NAME
dr-llm project restore NAME BACKUP_PATH  # BACKUP_PATH must be .sql.gz
dr-llm project destroy NAME --yes-really-delete-everything
```

### Deleting pools and projects

Deletion now uses one standard primitive: pool deletion.

- `dr-llm pool destroy PROJECT_NAME POOL_NAME --yes-really-delete-everything`
  deletes the fixed pool table set for that pool name (`pool_<name>_samples`
  and `pool_<name>_leases`), any consumer claim tables
  (`pool_<name>_claims_<consumer_id>`), and the pool's row from `pool_catalog`.
- `dr-llm pool destroy-testish PROJECT_NAME --yes-really-delete-everything`
  discovers pools in that project and deletes only the ones whose
  underscore-delimited lowercase name tokens include `test`, `tst`, `smoke`, or `demo`
- `dr-llm pool destroy-testish PROJECT_NAME --dry-run` previews the matched
  pools and returns the same structured result shape without deleting anything
- direct pool deletion requires the project to be running, but leased rows
  do not block deletion
- legacy pools without persisted `pool_catalog` metadata can still be deleted,
  because deletion targets the derived table names directly rather than loading
  `PoolSchema` from `pool_catalog`

`dr-llm project destroy` is now an orchestrator over pool deletion rather than a
blind Docker destroy.

- if the project is stopped, it is started temporarily for pool discovery and deletion
- discovered pools are deleted with bounded parallelism, but result ordering is
  deterministic and follows pool discovery order rather than completion order
- if any pool deletion fails, project container and volume deletion are skipped
- if the project had to be started temporarily and deletion fails, it is stopped
  again to restore the original state

Both destroy commands now emit structured JSON results. For project deletion,
the payload includes `discovered_pool_names`, ordered `pool_results`,
`temporarily_started`, and `destroyed_project_resources`.

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
