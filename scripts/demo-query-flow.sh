#!/usr/bin/env bash
set -euo pipefail

# Demo: end-to-end query recording and retrieval flow.
#
# Prerequisites:
#   source ./scripts/start-integration-postgres.sh
#   At least one of OPENAI_API_KEY or ANTHROPIC_API_KEY set.

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

step() { echo -e "\n${BOLD}${CYAN}── $1${RESET}\n"; }
ok()   { echo -e "${GREEN}✓ $1${RESET}"; }
fail() { echo -e "${RED}✗ $1${RESET}"; exit 1; }

if [[ -z "${DR_LLM_DATABASE_URL:-}" ]]; then
  fail "DR_LLM_DATABASE_URL is not set. Run: source ./scripts/start-integration-postgres.sh"
fi

# Pick a provider based on available API keys.
PROVIDER=""
MODEL=""
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  PROVIDER="openai"
  MODEL="gpt-4o-mini"
elif [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  PROVIDER="anthropic"
  MODEL="claude-sonnet-4-20250514"
else
  fail "Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run this demo."
fi

echo -e "${BOLD}Using provider=${PROVIDER} model=${MODEL}${RESET}"

# ── 1. Start a run ──────────────────────────────────────────────────────────

step "1. Starting a new run"

RUN_OUTPUT=$(uv run dr-llm run start \
  --run-type "demo" \
  --metadata-json '{"description": "demo query flow"}')

RUN_ID=$(echo "$RUN_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['run_id'])")
ok "Created run: ${RUN_ID}"

# ── 2. Make a couple of queries ─────────────────────────────────────────────

step "2. Sending first query"

RESP1=$(uv run dr-llm query \
  --provider "$PROVIDER" --model "$MODEL" \
  --message "What is the capital of France? Reply in one sentence." \
  --run-id "$RUN_ID")

TEXT1=$(echo "$RESP1" | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])")
ok "Response: ${TEXT1}"

step "3. Sending second query"

RESP2=$(uv run dr-llm query \
  --provider "$PROVIDER" --model "$MODEL" \
  --message "Name three primary colors. Reply in one sentence." \
  --run-id "$RUN_ID")

TEXT2=$(echo "$RESP2" | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])")
ok "Response: ${TEXT2}"

# ── 3. Finish the run ───────────────────────────────────────────────────────

step "4. Finishing the run"

uv run dr-llm run finish --run-id "$RUN_ID" --status success > /dev/null
ok "Run ${RUN_ID} marked as success"

# ── 4. List recorded calls ──────────────────────────────────────────────────

step "5. Listing recorded calls for this run"

uv run dr-llm run list-calls --run-id "$RUN_ID" | python3 -m json.tool

# ── Done ────────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}${GREEN}Demo complete!${RESET}"
echo -e "Run ${CYAN}source ./scripts/stop-integration-postgres.sh${RESET} to clean up the database container."
