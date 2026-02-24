# AGENTS.md

## Required Quality Gate

Before completing any coding task in this repository, always run:

1. `uv run ruff format`
2. `uv run ruff check --fix src/ tests/ scripts/`
3. Manually fix any remaining lint issues in `src/` only.
4. `uv run ty check src`
5. `uv run pytest tests/ -v`

Fix all issues reported by these commands before considering the task complete.

## Modeling Standard

Always use Pydantic models instead of Python `dataclass` definitions.

- Do not introduce new `@dataclass` usage in this repository.
- When touching existing `@dataclass` models, migrate them to `pydantic.BaseModel`.
- Prefer constructor-style validation for mapping inputs: `Model(**payload)`.
- Use `model_validate()` only when the input is not a plain mapping or when constructor style is not viable.
