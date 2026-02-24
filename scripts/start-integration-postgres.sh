#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-llm-pool-pg-test}"
PORT="${PORT:-5433}"
DB_NAME="${DB_NAME:-llm_pool_test}"
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

echo "Started container '${CONTAINER_NAME}' on localhost:${PORT}"
echo "Export this for integration tests:"
echo "export LLM_POOL_TEST_DATABASE_URL='postgresql://${DB_USER}:${DB_PASSWORD}@localhost:${PORT}/${DB_NAME}'"
