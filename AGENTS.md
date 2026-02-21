# AGENTS.md

## Required Quality Gate

Before completing any coding task in this repository, always run:

1. `uv run ruff format`
2. `uv run ruff check .`
3. `uv run ty check`

Fix all issues reported by these commands before considering the task complete.

## Modeling Standard

Always use Pydantic models instead of Python `dataclass` definitions.

- Do not introduce new `@dataclass` usage in this repository.
- When touching existing `@dataclass` models, migrate them to `pydantic.BaseModel`.
