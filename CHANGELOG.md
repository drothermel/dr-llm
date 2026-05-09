# Changelog

## 3.0.0 - 2026-05-09

### Changed

- Added `PoolTableType` and enum-backed table-name helpers for pool database
  table naming.
- Removed `PoolSchema.samples_table`, `claims_table`, `pending_table`,
  `metadata_table`, and `call_stats_table`; use
  `PoolSchema.table_name(...)` instead.
- Refactored `PoolTables` into an enum-keyed table collection accessed through
  `tables` or `PoolTables[PoolTableType]`, removing named table attributes and
  table-specific helper members.
- Preserved existing pool database table names, so this API break does not
  require a database migration.
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
