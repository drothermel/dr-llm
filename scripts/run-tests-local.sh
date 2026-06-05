#!/usr/bin/env bash
set -euo pipefail

# Run integration tests using a temporary Docker-managed Postgres project.
# Usage: ./scripts/run-tests-local.sh
#
# The script creates a project via `dr-llm project create`, runs the
# integration test suite, and destroys the project on exit.

PROJECT_NAME="dr_llm_test_runner"
NATS_CONTAINER_NAME="dr_llm_nats_test_runner"

cleanup() {
  echo "Destroying temporary NATS container '${NATS_CONTAINER_NAME}'..."
  docker rm -f "${NATS_CONTAINER_NAME}" >/dev/null 2>&1 || true
  echo "Destroying temporary project '${PROJECT_NAME}'..."
  uv run dr-llm project destroy "${PROJECT_NAME}" --yes-really-delete-everything 2>/dev/null || true
}

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to run integration tests."
  exit 1
fi

trap cleanup EXIT

# Clean up any leftover project from a prior run
uv run dr-llm project destroy "${PROJECT_NAME}" --yes-really-delete-everything 2>/dev/null || true
docker rm -f "${NATS_CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "Creating temporary NATS container '${NATS_CONTAINER_NAME}'..."
docker run -d \
  --name "${NATS_CONTAINER_NAME}" \
  -p 127.0.0.1::4222 \
  nats -js >/dev/null
NATS_PORT_LINE=$(docker port "${NATS_CONTAINER_NAME}" 4222/tcp | head -n 1)
NATS_PORT="${NATS_PORT_LINE##*:}"
NATS_URL="nats://127.0.0.1:${NATS_PORT}"
NATS_READY_TIMEOUT_SECONDS="${NATS_READY_TIMEOUT_SECONDS:-30}"
NATS_READY_DEADLINE=$((SECONDS + NATS_READY_TIMEOUT_SECONDS))
while ! (exec 3<>"/dev/tcp/127.0.0.1/${NATS_PORT}") 2>/dev/null; do
  if [ "${SECONDS}" -ge "${NATS_READY_DEADLINE}" ]; then
    echo "NATS did not become ready at ${NATS_URL} within ${NATS_READY_TIMEOUT_SECONDS}s."
    exit 1
  fi
  sleep 0.2
done
exec 3>&- 3<&-
echo "NATS ready at ${NATS_URL}"

echo "Creating temporary project '${PROJECT_NAME}'..."
PROJECT_JSON=$(uv run dr-llm project create "${PROJECT_NAME}") || {
  echo "Failed to create project '${PROJECT_NAME}'."
  exit 1
}
if [ -z "${PROJECT_JSON}" ]; then
  echo "Project create returned empty output."
  exit 1
fi

DSN=$(echo "${PROJECT_JSON}" | uv run python -c "import sys,json; print(json.load(sys.stdin)['dsn'])" 2>/dev/null) || {
  echo "Failed to parse DSN from project create output:"
  echo "${PROJECT_JSON}"
  exit 1
}
echo "Postgres ready at ${DSN}"

echo "Running integration tests..."
DR_LLM_TEST_DATABASE_URL="${DSN}" \
DR_LLM_TEST_NATS_URL="${NATS_URL}" \
uv run pytest tests/ -v -m integration -n 0 "$@"
