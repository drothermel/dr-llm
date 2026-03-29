#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set -euo pipefail
fi

CONTAINER_NAME="${CONTAINER_NAME:-dr-llm-pg-test}"
PORT="${PORT:-5433}"
DB_NAME="${DB_NAME:-dr_llm_test}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-postgres}"
IMAGE="${IMAGE:-postgres:16}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to start local integration postgres."
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  docker start "${CONTAINER_NAME}" >/dev/null
else
  docker run -d \
    --name "${CONTAINER_NAME}" \
    -e "POSTGRES_DB=${DB_NAME}" \
    -e "POSTGRES_USER=${DB_USER}" \
    -e "POSTGRES_PASSWORD=${DB_PASSWORD}" \
    -p "${PORT}:5432" \
    "${IMAGE}" >/dev/null
fi

MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-30}"
echo "Waiting for Postgres readiness (up to ${MAX_WAIT_SECONDS}s)..."
for _ in $(seq 1 "${MAX_WAIT_SECONDS}"); do
  if docker exec "${CONTAINER_NAME}" pg_isready -U "${DB_USER}" -d "${DB_NAME}" >/dev/null 2>&1; then
    echo "Postgres is ready."
    sleep 2
    break
  fi
  sleep 1
done

if ! docker exec "${CONTAINER_NAME}" pg_isready -U "${DB_USER}" -d "${DB_NAME}" >/dev/null 2>&1; then
  echo "Postgres did not become ready in time."
  exit 1
fi

echo "Started container '${CONTAINER_NAME}' on localhost:${PORT}"

DATABASE_URL="postgresql://${DB_USER}:${DB_PASSWORD}@localhost:${PORT}/${DB_NAME}"
export DR_LLM_DATABASE_URL="${DATABASE_URL}"
export DR_LLM_TEST_DATABASE_URL="${DATABASE_URL}"
echo "Exported DR_LLM_DATABASE_URL and DR_LLM_TEST_DATABASE_URL"

REPO_ROOT="$(git rev-parse --show-toplevel)"
POOL_DIR="${REPO_ROOT}/src/dr_llm/pool"

echo "Applying schema migrations..."
psql -v ON_ERROR_STOP=1 "${DATABASE_URL}" -f "${POOL_DIR}/schema_bootstrap.sql" || { echo "Migration failed: schema_bootstrap.sql"; return 1 2>/dev/null || exit 1; }
for migration in "${POOL_DIR}"/migrations/*.sql; do
  [ -f "$migration" ] || continue
  psql -v ON_ERROR_STOP=1 "${DATABASE_URL}" -f "$migration" || { echo "Migration failed: $(basename "$migration")"; return 1 2>/dev/null || exit 1; }
done
echo "Schema migrations applied."
