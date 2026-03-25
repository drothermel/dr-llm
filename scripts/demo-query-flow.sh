#!/usr/bin/env bash
set -euo pipefail

# Demo: end-to-end project creation, query recording, and retrieval flow.
#
# Prerequisites:
#   Docker running.
#   At least one of OPENAI_API_KEY or ANTHROPIC_API_KEY set.

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

PROJECT_NAME="demo-flow"

step() { echo -e "\n${BOLD}${CYAN}── $1${RESET}\n"; }
ok()   { echo -e "${GREEN}✓ $1${RESET}"; }
fail() { echo -e "${RED}✗ $1${RESET}"; exit 1; }

DEMO_SUCCEEDED=false
cleanup() {
  if [[ "$DEMO_SUCCEEDED" == "true" ]]; then
    return
  fi
  echo -e "\n${BOLD}Cleaning up demo project after failure...${RESET}"
  uv run dr-llm project destroy "$PROJECT_NAME" --yes-really-delete-everything 2>/dev/null || true
}
trap cleanup EXIT INT TERM

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

# ── 1. Create a project ────────────────────────────────────────────────────

step "1. Creating project '${PROJECT_NAME}'"

# Clean up any leftover from a previous run.
uv run dr-llm project destroy "$PROJECT_NAME" --yes-really-delete-everything 2>/dev/null || true

uv run dr-llm project create "$PROJECT_NAME"
ok "Project created"

# ── 2. Start a run ──────────────────────────────────────────────────────────

step "2. Starting a new run"

RUN_OUTPUT=$(uv run dr-llm --project "$PROJECT_NAME" run start \
  --run-type "demo" \
  --metadata-json '{"description": "demo query flow"}')

RUN_ID=$(echo "$RUN_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['run_id'])")
ok "Created run: ${RUN_ID}"

# ── 3. Make a couple of queries ─────────────────────────────────────────────

step "3. Sending first query"

RESP1=$(uv run dr-llm --project "$PROJECT_NAME" query \
  --provider "$PROVIDER" --model "$MODEL" \
  --message "What is the capital of France? Reply in one sentence." \
  --run-id "$RUN_ID")

TEXT1=$(echo "$RESP1" | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])")
ok "Response: ${TEXT1}"

step "4. Sending second query"

RESP2=$(uv run dr-llm --project "$PROJECT_NAME" query \
  --provider "$PROVIDER" --model "$MODEL" \
  --message "Name three primary colors. Reply in one sentence." \
  --run-id "$RUN_ID")

TEXT2=$(echo "$RESP2" | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])")
ok "Response: ${TEXT2}"

# ── 4. Finish the run ───────────────────────────────────────────────────────

step "5. Finishing the run"

uv run dr-llm --project "$PROJECT_NAME" run finish --run-id "$RUN_ID" --status success > /dev/null
ok "Run ${RUN_ID} marked as success"

# ── 5. List recorded calls ──────────────────────────────────────────────────

step "6. Listing recorded calls for this run"

uv run dr-llm --project "$PROJECT_NAME" run list-calls --run-id "$RUN_ID" | python3 -m json.tool

# ── 6. Backup the project ────────────────────────────────────────────────────

step "7. Backing up project"

BACKUP_OUTPUT=$(uv run dr-llm project backup "$PROJECT_NAME")
BACKUP_PATH=${BACKUP_OUTPUT#Backup saved to }
ok "$BACKUP_OUTPUT"

# ── 7. Destroy and restore ──────────────────────────────────────────────────

step "8. Destroying project to test restore"

uv run dr-llm project destroy "$PROJECT_NAME" --yes-really-delete-everything
ok "Project destroyed"

step "9. Recreating project and restoring from backup"

uv run dr-llm project create "$PROJECT_NAME" > /dev/null
uv run dr-llm project restore "$PROJECT_NAME" "$BACKUP_PATH"
ok "Restored from backup"

step "10. Verifying restored data"

RESTORED_CALLS=$(uv run dr-llm --project "$PROJECT_NAME" run list-calls --run-id "$RUN_ID")
RESTORED_COUNT=$(echo "$RESTORED_CALLS" | python3 -c "import sys,json; print(json.load(sys.stdin)['count'])")
if [[ "$RESTORED_COUNT" -eq 2 ]]; then
  ok "Found ${RESTORED_COUNT} calls after restore"
else
  fail "Expected 2 calls after restore, got ${RESTORED_COUNT}"
fi

# ── 8. Show all projects ────────────────────────────────────────────────────

step "11. Listing all projects"

uv run dr-llm project list

# ── Done ────────────────────────────────────────────────────────────────────

DEMO_SUCCEEDED=true

echo ""
echo -e "${BOLD}${GREEN}Demo complete!${RESET}"
echo -e "The '${PROJECT_NAME}' project is still running with your data preserved."
echo -e ""
echo -e "To stop it (data preserved):  ${CYAN}uv run dr-llm project stop ${PROJECT_NAME}${RESET}"
echo -e "To destroy it permanently:    ${CYAN}uv run dr-llm project destroy ${PROJECT_NAME} --yes-really-delete-everything${RESET}"
