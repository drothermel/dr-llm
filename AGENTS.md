# AGENTS.md

## Required Quality Gate

Before completing any coding task in this repository, always run:

1. `uv run ruff format`
2. `uv run ruff check --fix .`
3. Manually fix any remaining lint issues in the repository.
4. `uv run ty check`
5. `uv run pytest tests/ -v`

Fix all issues reported by these commands before considering the task complete.

## Postgres Test Flow

Some tests and demo scripts require a live local Postgres instance.

1. Start the disposable test database with `./scripts/start-test-postgres.sh`.
2. The script starts a Docker container on `localhost:5433`, exports `DR_LLM_DATABASE_URL` and `DR_LLM_TEST_DATABASE_URL`, and applies the pool schema plus migrations.
3. In the same shell, run any Postgres-backed checks, for example:
- `uv run pytest tests/integration/test_pool_fill.py -v`
- `uv run pytest tests/integration/test_pool_store.py -v`
- `uv run python scripts/demo-pool-fill.py --dsn postgresql://postgres:postgres@localhost:5433/dr_llm_test`
4. When finished, remove the container with `./scripts/stop-test-postgres.sh`.

If the Postgres container is not running, the integration fixtures should skip rather than fail, but do not rely on skip-only runs when changing Postgres-backed behavior. For pool changes or Postgres-backed demo changes, explicitly bring the container up and run the relevant integration tests or scripts against it.

## CI Parity

CI mirrors the local quality gate by splitting test scope across workflows:

1. Local quality gate runs `pytest tests/ -v`.
2. `.github/workflows/ci.yml`:
- `quality-unit`: `ruff format --check`, `ruff check .`, `ty check`, `pytest -m "not integration"` for fast PR feedback.
- `security`: `uv lock --check`, `uvx pip-audit`.
3. `.github/workflows/integration.yml`:
- Postgres-backed `pytest -m integration` on `main`, manual dispatch, and PRs labeled `run-integration`.

Together, CI runs the same overall test categories as local `pytest tests/ -v`, but in separate jobs.

## Modeling Standard

Always use Pydantic models instead of Python `dataclass` definitions.

- Do not introduce new `@dataclass` usage in this repository.
- When touching existing `@dataclass` models, migrate them to `pydantic.BaseModel`.
- Prefer constructor-style validation for mapping inputs: `Model(**payload)`.
- Use `model_validate()` only when the input is not a plain mapping or when constructor style is not viable.
