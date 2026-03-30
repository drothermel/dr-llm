#!/usr/bin/env bash
set -euo pipefail

# Run integration tests using a temporary Docker-managed Postgres project.
# Usage: ./scripts/run-tests-local.sh
#
# The script creates a project via `dr-llm project create`, runs the
# integration test suite, and destroys the project on exit.

PROJECT_NAME="dr-llm-test-runner"

cleanup() {
  echo "Destroying temporary project '${PROJECT_NAME}'..."
  uv run dr-llm project destroy "${PROJECT_NAME}" --yes-really-delete-everything 2>/dev/null || true
}

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to run integration tests."
  exit 1
fi

# Clean up any leftover project from a prior run
uv run dr-llm project destroy "${PROJECT_NAME}" --yes-really-delete-everything 2>/dev/null || true

echo "Creating temporary project '${PROJECT_NAME}'..."
PROJECT_JSON=$(uv run dr-llm project create "${PROJECT_NAME}")
DSN=$(echo "${PROJECT_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin)['dsn'])")
echo "Postgres ready at ${DSN}"

trap cleanup EXIT

echo "Running integration tests..."
DR_LLM_TEST_DATABASE_URL="${DSN}" uv run pytest tests/ -v -m integration "$@"
