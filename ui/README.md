# dr-llm UI Tools

Standalone web UI for inspecting and interacting with dr-llm subsystems.

## Architecture

- **Backend**: FastAPI app (`ui/api/`) that wraps the `dr_llm` library
- **Frontend**: React + Vite app (`ui/frontend/`) with React Router for multi-page navigation

## Quick Start

### Backend

```bash
# From the repo root (with dr-llm installed)
uvicorn ui.api.main:app --reload --port 8000
```

### Frontend

```bash
cd ui/frontend
npm install
npm run dev
```

The frontend dev server runs on `http://localhost:5173` and proxies API calls to `http://localhost:8000`.

## Available Tools

### Providers & Models

Browse all supported LLM providers, see their availability status (API keys, executables), and explore the model catalog for each provider. Supports live sync for providers with valid credentials and falls back to static model lists otherwise.
