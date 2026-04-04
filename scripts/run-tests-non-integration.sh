#!/usr/bin/env bash
set -euo pipefail

# Run the non-integration test suite with xdist.
# Usage: ./scripts/run-tests-non-integration.sh

uv run pytest tests/ -v -m "not integration" -n auto "$@"
