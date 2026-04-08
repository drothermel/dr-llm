# CLAUDE.md

## Quality Gate

After any major change, run:

1. `uv run ruff format`
2. `uv run ruff check --fix .`
3. Manually fix any remaining lint issues
4. `uv run ty check`
5. `uv run pytest tests/ -v -m "not integration"`

Then, immediately before committing, also run:

6. `./scripts/run-tests-local.sh` (integration tests — requires Docker)
