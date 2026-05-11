# Changelog

## 4.1.0 - 2026-05-11

### Added

- Added provider orchestrators for OpenAI, Anthropic, Google, OpenRouter, GLM,
  Codex, Claude Code, Kimi Code, and MiniMax to own provider-specific request
  defaults, validation, reasoning controls, catalog policy, and generation.
- Added provider-specific authoring config models such as `OpenAIGpt52Config`,
  `AnthropicBudgetConfig`, `GoogleBudgetConfig`, `CodexGpt54Config`, and related
  model-family config helpers that serialize to the shared `LlmConfig` shape.
- Added `ProviderRequestDefaults` and orchestrator `request_defaults(...)`
  helpers for inspecting default effort, reasoning, sampling, token, and mode
  behavior by provider/model.

### Changed

- Replaced the provider-specific runtime config/request union with shared
  `LlmConfig` and `LlmRequest` models plus optional nested `SamplingControls`.
- Moved provider implementation modules under `dr_llm.llm.providers.impls`,
  shared orchestration and registry code under `dr_llm.llm.providers.core`, and
  reusable reasoning/capability concepts under `dr_llm.llm.providers.concepts`.
- Centralized provider defaults and validation in orchestrators so stored
  configs and direct request construction follow the same provider constraints.
- Updated model catalog sync, CLI flows, demo scripts, notebooks, README
  examples, and tests to use orchestrator-backed provider access.
- Kimi Code max-token defaults are now supplied by its orchestrator when callers
  omit them.

### Breaking

- `ProviderRegistry` now registers and returns provider orchestrators instead of
  raw provider transports.
- Removed old concrete runtime config/request classes such as
  `OpenAILlmConfig`, `ApiLlmConfig`, `KimiCodeLlmConfig`, `HeadlessLlmConfig`,
  `OpenAILlmRequest`, `ApiLlmRequest`, `KimiCodeLlmRequest`, and
  `HeadlessLlmRequest`; use `LlmConfig`, `LlmRequest`, orchestrator
  `build_request(...)`, or provider-specific authoring configs instead.
- Direct imports from the previous provider package layout, including central
  reasoning/default helper modules, must move to the new `core`, `concepts`, or
  `impls` modules.

## 4.0.3 - 2026-05-10

### Added

- Added public `drain_until` worker and `drain_pool` pool helpers for reusable
  worker-drain flows.

### Changed

- Updated the pool-fill demo to use `drain_pool` for progress-aware worker
  draining.

## 4.0.2 - 2026-05-10

### Changed

- Removed `ReasoningWarning.mode`. Warning mode is now represented by the
  containing `LlmResponse.mode` instead.
- Moved `Message` to `dr_llm.llm.request` and `CallMode` to
  `dr_llm.llm.response`; both remain exported from `dr_llm.llm`.

### Breaking

- Code that reads or passes `ReasoningWarning.mode` must use
  `LlmResponse.mode` instead.
- Direct imports from the removed `dr_llm.llm.messages` module no longer work.

## 4.0.1 - 2026-05-10

### Changed

- Added public `dr_llm.llm`, `dr_llm.pool`, and `dr_llm.project` imports for
  common provider, pool, project, reasoning, and worker helpers.
- Updated demo scripts and pool-inspection notebooks to use the public import
  surfaces, including `PoolReader.open(...)` for catalog-backed pool reads.
- Refocused the README around high-level usage and one-line demo commands, with
  runnable end-to-end workflows kept in `scripts/demo-*.py`.
- Clarified demo script prerequisites and `--help` output for pool filling and
  reasoning/effort checks.

## 4.0.0 - 2026-05-10

### Changed

- Replaced the `pool/pending/` subpackage with a unified two-table design
  (`pool_<name>_samples` plus `pool_<name>_leases`); `PoolStore` now drives
  seeding, lease acquisition, and completion through one set of APIs.
- Extracted no-replacement claim acquisition out of `pool/` into a new
  `dr_llm.sampling` package with its own claims table, store, and service.
- Rebuilt `PoolReader` and `pool.admin` (creation, deletion, discovery,
  inspection) on top of the unified store; admin entry points are now imported
  from `dr_llm.pool.admin.*` rather than the removed `pool.admin_service`.
- Renamed `pool/results.py` to `pool/insert_result.py` and `pool/key_filter.py`
  to `pool/db/key_filter.py`; the `LlmPoolAdapter` shim was removed in favor of
  using `LlmPoolBackend` directly.
- Added `seed_llm_grid` for declarative `(axis × axis × …)` seeding from
  `Axis` / `AxisMember` lists.
- Pool deletion now drops consumer-scoped sampling claim tables for the target
  pool, and sampling acquisition only claims completed samples.
- Documented `RoundRobinClaimer` for worker claiming across explicit key values.

### Fixed

- Several provider request validators now require explicit `reasoning` and/or
  `effort` (google, glm, codex, openrouter, minimax, kimi-code, claude-code).
  Updated `scripts/demo-pool-providers.py` to send the minimum valid config
  per provider; the script was previously also passing a removed top-level
  `--project` option to the `dr-llm` CLI.

## 3.0.0 - 2026-05-09

### Changed

- Added a pool inspection marimo notebook with pool status views, and updated
  notebook dependencies/tooling.
- Added `PoolTableType` and enum-backed table-name helpers for pool database
  table naming.
- Added enum-backed pool database column names, server defaults, and index names,
  and extracted per-table definitions for samples, claims, pending rows,
  metadata, and call stats.
- Removed `PoolSchema.samples_table`, `claims_table`, `pending_table`,
  `metadata_table`, and `call_stats_table`; use
  `PoolSchema.table_name(...)` instead.
- Refactored `PoolTables` into an enum-keyed table collection accessed through
  `tables` or `PoolTables[PoolTableType]`, removing named table attributes and
  table-specific helper members.
- Split pool request/result/admin models out of `dr_llm.pool.models` into
  functional modules such as `dr_llm.pool.acquisition`,
  `dr_llm.pool.results`, and `dr_llm.pool.admin.*`.
- Split pool administration out of `dr_llm.pool.admin_service` into discovery,
  inspection, creation, and deletion modules under `dr_llm.pool.admin`.
- Removed package-level shortcuts for pool admin/model APIs; import pool
  acquisition, deletion, inspection, and result types from their direct modules.
- Preserved existing pool database table names, so this API break does not
  require a database migration.
- Removed the one-off call-stats migration script; existing pools can create
  the missing `call_stats` table by running `PoolStore.ensure_schema()` with
  their schema.
- Documented pool `call_stats` backfill expectations and updated examples for
  the new pool import paths.
- Removed noisy re-export, CLI surface, one-off regression, runtime typing, and
  demo-script tests from the suite.
- Narrowed catalog and OpenRouter policy data tests to validation plus
  representative behavior checks instead of exact inventory snapshots.

## 2.3.0 - 2026-04-23

### Added

- Added structured pool deletion APIs and models:
  `assess_pool_deletion(...)`, `delete_pool(...)`, `DeletePoolRequest`,
  readiness models, violations, statuses, and results.
- Added a `dr-llm pool destroy-testish PROJECT_NAME --yes-really-delete-everything`
  convenience command that discovers pools in a project and deletes only the
  ones whose underscore-delimited lowercase name tokens include `test`, `tst`,
  `smoke`, or `demo`.
- Added `--dry-run` support for `dr-llm pool destroy-testish` so matching pools
  can be previewed without deletion.
- Added structured project deletion APIs and models:
  `assess_project_deletion(...)`, `delete_project(...)`, `DeleteProjectRequest`,
  readiness models, violations, statuses, and results.
- Added `dr-llm pool destroy PROJECT_NAME POOL_NAME --yes-really-delete-everything`.
- Added integration coverage for normal pool deletion, legacy pool deletion
  without `_schema` metadata, and deletion with missing ancillary tables.

### Changed

- Redefined `dr-llm project destroy` to orchestrate pool deletion before
  destroying project resources.
- Pool deletion no longer blocks on pending or leased rows in a pool's pending
  table; pending-row counts are still reported in deletion readiness.
- Project deletion now auto-starts stopped projects when needed for discovery,
  deletes pools with bounded parallelism, preserves discovered pool order in
  results, and skips Docker destroy if any pool deletion fails.
- Destroy commands now emit structured JSON results instead of ad hoc success
  messages.
- Package exports now include the new deletion APIs from `dr_llm.pool`,
  `dr_llm.project`, and top-level package entrypoints.
