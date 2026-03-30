# CLAUDE.md

## Quality Gate

Before finishing coding work in this repository, run:

1. `uv run ruff format`
2. `uv run ruff check --fix .`
3. Manually fix any remaining lint issues
4. `uv run ty check`
5. `uv run pytest tests/ -v`

## Local Postgres Flow

Postgres-backed integration tests and demo scripts should be run against the disposable Docker database provided by the repo scripts.

1. Start it with `./scripts/start-test-postgres.sh`.
2. This creates `dr-llm-pg-test` on `localhost:5433`, exports `DR_LLM_DATABASE_URL` and `DR_LLM_TEST_DATABASE_URL`, and applies schema bootstrap plus migrations.
3. In that shell, run the relevant Postgres-backed verification:
- `uv run pytest tests/integration/test_pool_fill.py -v`
- `uv run pytest tests/integration/test_pool_store.py -v`
- `uv run pytest tests/integration/test_postgres_repository.py -v`
- `uv run python scripts/demo-pool-fill.py --dsn postgresql://postgres:postgres@localhost:5433/dr_llm_test`
4. Tear it down with `./scripts/stop-test-postgres.sh`.

Integration fixtures may skip when Postgres is unavailable, but for any change that touches Postgres-backed behavior, explicitly run the relevant integration tests or demo against the live container.
