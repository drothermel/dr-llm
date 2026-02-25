# Releasing `dr-llm` to PyPI

This runbook documents the manual local release process for publishing `dr-llm` to public PyPI.

## Prerequisites

- You have maintainer access to the `dr-llm` project on PyPI.
- `UV_PUBLISH_TOKEN` is set with a valid PyPI API token.
- You are on the intended release commit.

## Release Flow

1. Bump the package version in `pyproject.toml` under `[project].version`.
2. Run the required quality gate commands from `AGENTS.md`:

```bash
uv run ruff format
uv run ruff check --fix .
uv run ty check
uv run pytest tests/ -v
```

3. Build distributions:

```bash
uv build
```

4. Validate package metadata/rendering:

```bash
uvx twine check dist/*
```

5. Preflight package name and version against PyPI:

```bash
curl -s -o /tmp/pypi_dr_llm.json -w '%{http_code}\n' https://pypi.org/pypi/dr-llm/json
```

- `404` means the distribution name is currently available.
- `200` means the name already exists. Inspect `/tmp/pypi_dr_llm.json` for current versions.

6. Publish to PyPI:

```bash
uv publish --token "$UV_PUBLISH_TOKEN"
```

7. Post-publish smoke test from a clean temp project:

```bash
mkdir -p /tmp/dr-llm-smoke && cd /tmp/dr-llm-smoke
uv init
uv add dr-llm==<version>
uv run python -c "import dr_llm; print(dr_llm.__all__[:3])"
uv run dr-llm --help
```

## Name Collision Contingency

If the `dr-llm` distribution name is taken at publish time:

1. Update `[project].name` in `pyproject.toml` to another available name (for example, `dr-llm-core`).
2. Update install docs and examples that reference `dr-llm` as a distribution name.
3. Rebuild and re-validate:

```bash
uv build
uvx twine check dist/*
```

4. Publish with the new name:

```bash
uv publish --token "$UV_PUBLISH_TOKEN"
```

Note: the Python import path remains `dr_llm`.
