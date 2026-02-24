#!/usr/bin/env bash
set -euo pipefail

DEFAULT_TEST_DSN="postgresql://postgres:postgres@localhost:5433/llm_pool_test"
TEST_DSN="${LLM_POOL_TEST_DATABASE_URL:-${LLM_POOL_DATABASE_URL:-$DEFAULT_TEST_DSN}}"

if ! command -v psql >/dev/null 2>&1; then
  echo "psql is required for preflight checks. Install PostgreSQL client tools and retry."
  exit 1
fi

echo "Using integration DB: ${TEST_DSN}"
echo "Running DB preflight query..."
psql "${TEST_DSN}" -c "select current_user, current_database();" >/dev/null

echo "Running integration tests..."
LLM_POOL_TEST_DATABASE_URL="${TEST_DSN}" uv run pytest tests/ -v -m integration
