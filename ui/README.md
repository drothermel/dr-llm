# dr-llm UI Tools

Standalone web UI for inspecting and interacting with dr-llm subsystems.

## Architecture

- **Backend**: FastAPI app (`ui/api/`) that wraps the `dr_llm` library
- **Frontend**: Next.js app (`ui/frontend/`) with App Router navigation

## Quick Start

### Backend

```bash
# From the repo root
uv run uvicorn ui.api.main:app --reload --port 8000
```

### Frontend

```bash
cd ui/frontend
pnpm install
pnpm dev
```

The frontend dev server runs on `http://localhost:3000` and proxies API calls to `http://localhost:8000`.

## Available Tools

The app opens to a landing hub (`/`) that links to each tool.

### Providers & Models

(`/providers`) Browse all supported LLM providers, see their availability status (API keys, executables), and explore the model catalog for each provider. Supports live sync for providers with valid credentials and falls back to static model lists otherwise.

### nl-latents

(`/nl-latents`) Browse published NL-latent experiment samples: filter by family, difficulty, split, model, and budget, then open a sample to inspect its encoder prompt, NL latent, decoder prompts, validation, and the input → decoded code side by side.
