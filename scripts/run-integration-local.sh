#!/usr/bin/env bash
set -euo pipefail

DEFAULT_TEST_DSN="postgresql://postgres:postgres@localhost:5433/llm_pool_test"
TEST_DSN="${LLM_POOL_TEST_DATABASE_URL:-${LLM_POOL_DATABASE_URL:-$DEFAULT_TEST_DSN}}"

if ! command -v psql >/dev/null 2>&1; then
  echo "psql is required for preflight checks. Install PostgreSQL client tools and retry."
  exit 1
fi

SAFE_DSN="$(printf '%s' "${TEST_DSN}" | sed -E 's#(postgres(ql)?://)[^@/]+@#\1****:****@#')"
echo "Using integration DB: ${SAFE_DSN}"
echo "Running DB preflight query..."
MAX_RETRIES="${DB_READY_RETRIES:-15}"
SLEEP_SECONDS="${DB_READY_SLEEP_SECONDS:-1}"
for attempt in $(seq 1 "${MAX_RETRIES}"); do
  if psql "${TEST_DSN}" -c "select current_user, current_database();" >/dev/null 2>&1; then
    break
  fi
  if [ "${attempt}" -eq "${MAX_RETRIES}" ]; then
    echo "DB preflight failed after ${MAX_RETRIES} attempts."
    exit 1
  fi
  sleep "${SLEEP_SECONDS}"
done

echo "Running integration tests..."
LLM_POOL_TEST_DATABASE_URL="${TEST_DSN}" uv run pytest tests/ -v -m integration
