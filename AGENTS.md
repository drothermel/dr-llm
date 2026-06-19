# AGENTS.md

## Required Quality Gate

Before completing any library code change in this repository, always run:

1. `uv run ruff format`
2. `uv run ruff check --fix .`
3. Manually fix any remaining lint issues in the repository.
4. `uv run ty check`
5. `uv run pytest tests/ -v -m "not integration"`
6. `./scripts/run-tests-local.sh` (integration tests — requires Docker)

Fix all issues reported by these commands before considering the task complete.

For notebook-only changes under `nbs/`, do not run the full test suite unless
the notebook change also modifies library code or testable shared behavior.
Instead, run `uvx marimo check <notebook.py>` for each changed notebook.

## Integration Tests

`pytest` defaults to `pytest-xdist`, so `uv run pytest tests/ -v -m "not integration"` runs the safe non-integration suite in parallel.

`./scripts/run-tests-local.sh` auto-creates a temporary Docker Postgres project, runs `pytest -m integration -n 0`, and destroys it on exit. No manual setup needed — just Docker.

For targeted runs, pass extra pytest args: `./scripts/run-tests-local.sh -k test_pool_store`

## CI Parity

CI mirrors the local quality gate by splitting test scope across workflows:

1. Local quality gate runs `uv run pytest tests/ -v -m "not integration"` plus `./scripts/run-tests-local.sh`.
1. `.github/workflows/ci.yml`:
    - `quality-unit`: `ruff format --check`, `ruff check .`, `ty check`, `pytest -m "not integration"` for fast PR feedback.
    - `security`: `uv lock --check`, `uvx pip-audit`.
1. `.github/workflows/integration.yml`:
    - Postgres-backed `pytest -m integration` on `main`, manual dispatch, and PRs labeled `run-integration`.

Together, CI runs the same overall test categories as the local quality gate, but in separate jobs.

## Test Quality Guidance

Prefer tests that verify durable project behavior over tests that freeze
incidental surface form.

Good tests usually exercise one of these:

- Core domain behavior, data transformations, validation rules, and error
  handling.
- Public API or CLI behavior that users rely on, without asserting unnecessary
  formatting or implementation details.
- Persistence contracts such as database schema, indexes, serialization, and
  round trips.
- Integration boundaries where components can break when wired together.
- Regression coverage for bugs that represent an ongoing behavioral risk.

Avoid adding tests that mainly verify:

- Re-export availability from package `__init__` modules.
- Exact command/module surface shape when behavior is already covered elsewhere.
- Runtime typing smoke tests that duplicate static type checker coverage.
- One-off migration or compatibility behavior after the old behavior has been
  intentionally removed.
- Exact inventory snapshots of fast-changing curated data, model lists, policy
  files, or demo-script constants unless that inventory is itself a product
  contract.
- Demo or exploratory script internals unless the script is treated as supported
  user-facing behavior.

For curated data tests, prefer:

- Loading and schema validation.
- Representative spot checks for important policy decisions.
- Fresh-copy or immutability checks when callers rely on that behavior.

For CLI tests, prefer:

- Command dispatch to the correct service.
- User-visible error handling.
- Confirmation and destructive-action safeguards.
- JSON/output contracts only when consumers depend on them.

Before adding a test, ask: would this fail for a meaningful product regression,
or mostly because someone reorganized code, renamed an internal field, changed a
demo list, or removed an already-deprecated path? Prefer the former.

## Visual Preferences

When generating HTML pages, web UIs, dashboards, review tools, or any visual
artifact that has a color scheme, default to light mode. Use dark mode only
when explicitly requested.

## Modeling Standard

Always use Pydantic models instead of Python `dataclass` definitions.

- Do not introduce new `@dataclass` usage in this repository.
- When touching existing `@dataclass` models, migrate them to `pydantic.BaseModel`.
- Prefer constructor-style validation for mapping inputs: `Model(**payload)`.
- Use `model_validate()` only when the input is not a plain mapping or when constructor style is not viable.
- When using `StrEnum`, pass enum members directly where strings are accepted;
  do not use `.value` only to recover the string value.
