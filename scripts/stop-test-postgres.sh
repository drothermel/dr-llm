#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-dr-llm-pg-test}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to stop local integration postgres."
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null
  echo "Removed container '${CONTAINER_NAME}'."
else
  echo "Container '${CONTAINER_NAME}' not found."
fi
