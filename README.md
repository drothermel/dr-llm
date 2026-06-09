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

## Backends API

Programmatic integration surface for callers such as DSPy adapters. v1 supports
text-only requests; `extensions` must not include tools, structured output, or
multimodal payloads.

### DirectBackend (no database)

```python
import asyncio

from dr_llm.backends import BackendRequest, DirectBackend
from dr_llm.llm import CallMode, Message, ProviderName

backend = DirectBackend()
request = BackendRequest(
    provider=ProviderName.OPENAI,
    model="gpt-4.1-mini",
    mode=CallMode.api,
    messages=[Message(role="user", content="Hello")],
)

response = backend.complete(request)
async_response = asyncio.run(backend.acomplete(request))
```

### PoolBackend (cache, sessions, batch fill)

```python
from dr_llm.backends import BackendRequest, PoolBackend, PoolBackendConfig
from dr_llm.llm import CallMode, Message, ProviderName

pool = PoolBackend(
    PoolBackendConfig(
        pool_name="my_experiment",
        database_url="postgresql://localhost/dr_llm",
    )
)
request = BackendRequest(
    provider=ProviderName.OPENAI,
    model="gpt-4.1-mini",
    mode=CallMode.api,
    messages=[Message(role="user", content="Hello")],
)

cached = pool.complete(request)
session = pool.acquire(request, session_id="s1", n=10)
pool.submit_batch([request])
drain = pool.await_drain(timeout=60)
pool.close()
```

Async wrappers: `acomplete`, `aacquire`, and `adrain` delegate to the sync
implementations via `asyncio.to_thread`.

`PoolBackend.acquire()` and `PoolBackend.aacquire()` require a stable,
non-empty `session_id`. The session ID controls no-replacement acquisition:
repeat calls with the same `session_id` continue claiming from the same
session, while different session IDs start independent claim groups. Use an
experiment-stable value such as `experiment-name:split:seed` or a caller-owned
run ID. Do not derive acquisition session IDs from low-resolution timestamps.

### Fingerprinting

`fingerprint_request()` hashes a canonical JSON payload built from:
`provider`, `model`, `mode`, `messages`, `max_tokens`, `effort`, `reasoning`,
and `sampling`. `metadata` and `extensions` are excluded, so requests that
differ only in tracing metadata or unsupported extension payloads share pool
cache keys and session claims. Metadata is useful for audit context, but it is
not cache isolation and it is not acquisition session isolation.

Provider-output-affecting controls must be represented on the resolved
`BackendRequest` before fingerprinting. In particular, provider-native
reasoning belongs in `BackendRequest.reasoning`; top-level
`BackendRequest.effort` is not a substitute for OpenRouter's provider-specific
reasoning payload.

### submit_batch

`submit_batch()` seeds incomplete pool rows only for fingerprints that have no
complete samples and no pending incomplete samples for the same fingerprint.

### await_drain

`await_drain()` is single-flight per `PoolBackend` instance. Concurrent drain
calls on the same instance serialize on an internal lock rather than starting
overlapping worker fleets.

### dr-dspy experiment contract

The backend API is the integration surface for callers such as `dr-dspy`.
`dr-llm` owns provider/model routing, provider-native reasoning and sampling
controls, backend validation, fingerprinting, pool acquisition semantics, and
aggregate acquire provenance. `dr-dspy` owns TaskSpec/adapter prompt rendering,
DSPy LM request mapping, RunContext, transparency logs, and optimizer behavior.

For the reviewed compression experiments, resolved `BackendRequest` values
should use these provider-native fields:

| Experiment family | `provider` / `model` | Required fields |
|---|---|---|
| MiMo off | `openrouter` / `xiaomi/mimo-v2-flash` | `reasoning=OpenRouterReasoning(enabled=False)`, `sampling=SamplingControls(temperature=0.7, top_p=0.95)` |
| Nemotron off | `openrouter` / `nvidia/llama-3.3-nemotron-super-49b-v1.5` | `reasoning=OpenRouterReasoning(enabled=False)`, `sampling=SamplingControls(temperature=0.7, top_p=0.95)` |
| GPT-OSS low | `openrouter` / `openai/gpt-oss-20b` | `reasoning=OpenRouterReasoning(effort=OpenRouterEffortLevel.LOW)`, `sampling=SamplingControls(temperature=0.7, top_p=0.95)` |
| GPT-5 nano low | `openrouter` / `openai/gpt-5-nano` | `reasoning=OpenRouterReasoning(effort=OpenRouterEffortLevel.LOW)`, `sampling=SamplingControls(temperature=0.7, top_p=0.95)` |
| GPT-5 nano minimal | `openai` / `gpt-5-nano` | `reasoning=OpenAIReasoning(thinking_level=ThinkingLevel.MINIMAL)`, `sampling=None` |
| Gemini Flash Lite off | `google` / `gemini-2.5-flash-lite` | `reasoning=GoogleReasoning(thinking_level=ThinkingLevel.OFF)`, `sampling=SamplingControls(temperature=0.7, top_p=0.95)` |

`sampling=None` means no sampling override on the resolved request. Some
authoring configs accept `SamplingControls(temperature=None, top_p=None)` to
suppress provider defaults; that should resolve to `sampling=None` before the
request is sent or fingerprinted.

`EffortSpec.MAX` is intentionally a generic `BackendRequest.effort` value for
provider/model families whose capabilities include `"max"` in
`supported_effort_levels`, such as Anthropic or MiniMax effort-capable models.
Do not map it to OpenRouter provider-native reasoning: OpenRouter effort
controls use `OpenRouterReasoning(effort=OpenRouterEffortLevel.*)`, whose
supported values are `low`, `medium`, and `high`.

`AcquireResult(responses, claimed_from_cache, generated)` is stable public
provenance. Each returned `BackendResponse` also carries `source`, `sample_id`,
and `request_fingerprint` when available.

Single-completion backend calls do not carry an `n` field. For DSPy wrappers,
unset `n` and `n=1` are equivalent for direct or cache-first single completion.
Native multi-completion is not supported on `DirectBackend.complete()` or
`PoolBackend.complete()`; use `PoolBackend.acquire(..., n=...)` only when
explicit no-replacement pool acquisition is intended.

Built-in dr-dspy LM state loading should allow these class paths:
`dspy.clients.dr_llm.direct.DrLlmDirectLM` and
`dspy.clients.dr_llm.pool.DrLlmPoolLM`. Shared serialized fields are
`_dspy_lm_class`, `model`, `model_type`, `num_retries`,
`_dspy_provider_options`, optional `temperature`, optional `max_tokens`,
`dr_llm_mode`, and optional `dr_llm_provider_controls`. Pool state also stores
`dr_llm_pool_config` as `PoolBackendConfig.model_dump(mode="json")` and
optional `dr_llm_session_id`. Custom `registry` instances and pool
`session_id_resolver` callables are not serialized; restored LMs rebuild the
default provider registry and only restore an explicit session ID.

`PoolBackend` supports a native batch-fill workflow: build a request grid,
`submit_batch()`, `await_drain()`, then `acquire()` with a stable session ID.
Wrapper-level cache-first single completions are a different workflow. They do
not reproduce `nl_latents` grid axes, encoder-to-decoder lineage, prompt
bindings, compression baselines, or curve aggregation.

For exact `nl_latents` compression-curve replay, keep using the raw
single-user-message `nl_latents` plus `dr-llm` pool harness. DSPy
`Predict(TaskSpec)` rendering adds adapter prompt structure and should be
treated as a new prompt condition unless a raw request path is implemented and
verified separately.

The `nl_latents` parity check remains external to this repository to avoid a
cross-repo test dependency. From the sibling `../nl_latents` checkout, inspect:

```bash
uv run python - <<'PY'
from nl_latents.sampling.llm.catalog import get_llm_configs

ids = [
    "openrouter/xiaomi/mimo-v2-flash/off/v1",
    "openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5/off/v1",
    "openrouter/openai/gpt-5-nano/low/v1",
    "openrouter/openai/gpt-oss-20b/low/v1",
    "openai/gpt-5-nano/minimal/v1",
]
configs = get_llm_configs()
for config_id in ids:
    print(config_id, configs[config_id].model_dump(mode="json"))
PY
```

The printed provider, model, reasoning, and sampling fields should match the
table above, with `sampling=None` only for direct OpenAI GPT-5 nano minimal.

The v1 backend surface is text-only. Supported usage is plain text direct and
pool requests. Unsupported features should be rejected before provider calls:
tools and tool-call history, multimodal parts, native structured response
formats, stop sequences, logprobs, prompt-cache controls, and unsupported
reasoning shapes.

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
| `nbs/hit_providers.py` | Manually send prompts through curated provider configs and save response history. | API keys for the selected providers; writes optional logs under `logs/`. |

```bash
uv run python scripts/demo-providers.py
uv run python scripts/demo-pool-providers.py --help
uv run python scripts/demo-pool-fill.py --help
uv run python scripts/demo_thinking_and_effort.py --provider openai
uv run marimo run nbs/hit_providers.py
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

Provider implementation details live under
`src/dr_llm/llm/providers/impls/<provider>/`. Provider API, catalog, and docs
URLs are collected in provider-specific `<Provider>Urls` enums. Shared API key
environment variable names live in
`src/dr_llm/llm/providers/names.py` as `ApiKeyNames`; the default registry only
registers orchestrators.

Headless providers shell out to CLI tools. `minimax` and `kimi-code` are direct Anthropic-compatible `/messages` API providers. Headless input shapes do not expose `temperature`, `top_p`, or `max_tokens`. `kimi-code` rejects `temperature` and `top_p`; its orchestrator supplies the provider max-token default when callers omit it.

Some provider orchestrators use static fallback catalogs when a provider has no
`/models` endpoint or live discovery is unavailable. The CLI notes when a list
may be out of date and links to docs.

Catalog entries expose a `control_mode` field plus provider-owned
`metadata["dr_llm_controls"]` details instead of a flattened
`supports_reasoning` flag.

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
Provider control support is backed by provider-specific family capability
models, so model matching, thinking levels, effort support, budget bounds, and
provider defaults are defined in one inspectable data object per provider.

Provider orchestrators construct requests from stored configs or caller inputs.
Both paths run the same provider validation before generation, so persisted
`LlmConfig` values cannot bypass mode, max-token, sampling, effort, or reasoning
constraints. Orchestrators apply provider defaults for effort, reasoning, max
tokens, and sampling controls before generation. For generic sampling-capable
API providers, omitted sampling controls default to `temperature=1.0` and
`top_p=0.95`. OpenAI omits those fields unless you set them explicitly.
`kimi-code` and headless providers reject those fields entirely.

Use `build_default_registry().get(provider).request_defaults(model)` when
inspecting generic provider requests. It returns the orchestrator-owned defaults
for effort, reasoning, token limits, and supported sampling controls.

Use `build_default_registry().get(provider).controls(model)` when you need the
full provider control object, including `control_mode`,
`supported_thinking_levels`, `supported_effort_levels`, budget bounds where
available, and provider-specific request validation.

### Calling a provider

```python
from dr_llm.llm import (
    Message,
    OpenAIGpt52Config,
    SamplingControls,
    ThinkingLevel,
    build_default_registry,
)

registry = build_default_registry()
orchestrator = registry.get("openai")
config = OpenAIGpt52Config(
    model="gpt-5.2-mini",
    thinking_level=ThinkingLevel.OFF,
    sampling=SamplingControls(temperature=0.7, top_p=0.95),
).to_llm_config(registry)

response = orchestrator.generate(
    orchestrator.build_request_from_config(
        config=config,
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
dr-llm models list [--provider NAME] [--control-mode MODE] [--model-contains TEXT] [--json]
dr-llm models sync-list [--provider NAME] [--control-mode MODE] [--model-contains TEXT] [--json]
dr-llm models show --provider NAME --model NAME

# Query
dr-llm query --provider NAME --model NAME --message TEXT
dr-llm query --provider openai --model gpt-5-mini --reasoning-json '{"kind":"openai","thinking_level":"high"}' --message TEXT
dr-llm query --provider codex --model gpt-5.1-codex-mini --reasoning-json '{"kind":"codex","thinking_level":"xhigh"}' --message TEXT
dr-llm query --provider google --model gemini-2.5-flash --reasoning-json '{"kind":"google","thinking_level":"budget","budget_tokens":512}' --message TEXT
dr-llm query --provider openrouter --model openai/gpt-oss-20b --reasoning-json '{"kind":"openrouter","effort":"high"}' --message TEXT
OPENROUTER_API_KEY=... uv run pytest tests/integration/test_live_openrouter.py -q
# Full acceptance requires this live test to pass, not skip.

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
