# AGENTS.md

## Required Quality Gate

Before completing any coding task in this repository, always run:

1. `uv run ruff format`
2. `uv run ruff check --fix .`
3. Manually fix any remaining lint issues in the repository.
4. `uv run ty check`
5. `uv run pytest tests/ -v -m "not integration"`
6. `./scripts/run-tests-local.sh` (integration tests — requires Docker)

Fix all issues reported by these commands before considering the task complete.

## Integration Tests

`pytest` defaults to `pytest-xdist`, so `uv run pytest tests/ -v -m "not integration"` runs the safe non-integration suite in parallel.

`./scripts/run-tests-local.sh` auto-creates a temporary Docker Postgres project, runs `pytest -m integration -n 0`, and destroys it on exit. No manual setup needed — just Docker.

For targeted runs, pass extra pytest args: `./scripts/run-tests-local.sh -k test_pool_fill`

## CI Parity

CI mirrors the local quality gate by splitting test scope across workflows:

1. Local quality gate runs `uv run pytest tests/ -v -m "not integration"` plus `./scripts/run-tests-local.sh`.
1. `.github/workflows/ci.yml`:
    - `quality-unit`: `ruff format --check`, `ruff check .`, `ty check`, `pytest -m "not integration"` for fast PR feedback.
    - `security`: `uv lock --check`, `uvx pip-audit`.
1. `.github/workflows/integration.yml`:
    - Postgres-backed `pytest -m integration` on `main`, manual dispatch, and PRs labeled `run-integration`.

Together, CI runs the same overall test categories as the local quality gate, but in separate jobs.

## Modeling Standard

Always use Pydantic models instead of Python `dataclass` definitions.

- Do not introduce new `@dataclass` usage in this repository.
- When touching existing `@dataclass` models, migrate them to `pydantic.BaseModel`.
- Prefer constructor-style validation for mapping inputs: `Model(**payload)`.
- Use `model_validate()` only when the input is not a plain mapping or when constructor style is not viable.
