# dr-llm UI Frontend

React + Vite frontend for the `dr-llm` provider explorer.

## Commands

```bash
pnpm install
pnpm dev
pnpm lint
pnpm build
```

The dev server runs on `http://localhost:5173` and proxies `/api` requests to the backend at `http://localhost:8000`.

## Backend

Start the FastAPI backend from the repo root:

```bash
uv run uvicorn ui.api.main:app --reload --port 8000
```

## Current Surface

The frontend currently exposes a provider and model explorer:

- lists supported providers and local availability
- lazily loads models for each provider
- triggers live sync for available providers
- falls back to static model lists when live catalog fetches are unavailable
