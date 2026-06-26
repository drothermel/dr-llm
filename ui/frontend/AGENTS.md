# AGENTS.md — `ui/frontend`

Next.js 16 (app-router) + React 19 + Tailwind v4 frontend. This file governs work
under `ui/frontend/`; the repo-root `AGENTS.md` covers the Python library.

## Package manager: pnpm (NOT bun)

This app uses **pnpm** (`pnpm-lock.yaml`, `packageManager: pnpm@9.15.2`). Ignore any
global "use bun for frontend" preference here.

```bash
pnpm install
pnpm add <pkg>      # add a dependency
pnpm dev            # dev server on :3000, proxies /api -> :8000
pnpm lint           # eslint
pnpm build          # next build — ALSO runs the TypeScript check
```

There is **no separate typecheck script** — `pnpm build` runs `tsc`. The frontend
quality gate before considering a change done is: `pnpm lint` clean + `pnpm build`
green. (The root Python quality gate does not apply to frontend-only changes.)

Backend for local data: `uv run uvicorn ui.api.main:app --reload --port 8000` (repo root).

## Gotcha: root `.gitignore` traps `src/lib/`

The repo-root (Python) `.gitignore` has a bare `lib/` rule that matches
`ui/frontend/src/lib/`. New `.ts` files there are silently un-committed unless the
negation `!ui/frontend/src/lib/` is kept in that root `.gitignore`. Don't remove it;
after adding files under `src/lib/`, confirm with `git status` that they appear.

## Component architecture

Presentational components are **portable and themeable** — keep them that way:

- Live in grouped dirs: `src/components/{code,stats,panels,chips}/` and
  `src/components/primitives.tsx`. They hold **no domain types** (e.g. `CodePane`
  takes `{label, value, language, badge}`, never `NlLatentsSampleDetail`).
- Page-specific composition lives in `src/views/*`; pages in `src/app/*` are thin
  server wrappers that fetch and pass props.
- Styling is **entirely via CSS-variable tokens** (no raw colors). The full token
  contract is documented in `src/components/TOKENS.md` and defined in
  `src/app/globals.css`. A consuming project supplies those tokens to reuse a component.
- Every component accepts a trailing `className?` prop, **merged last** via `cn()` so
  callers can override defaults.

## Styling conventions

- **Conditional / merged class names go through `cn()`** (`src/lib/cn.ts`, a clsx
  wrapper) — not template-literal interpolation or `array.join(' ')`. Put the
  incoming `className` last so it wins.
- Tailwind-first; reference tokens as `bg-[var(--token)]`. Static class strings can
  stay as plain string constants — `cn()` is for the conditional/override cases.
- Fonts: `--font-mono` is **Fira Code** with ligatures (`'liga'`/`'calt'`) for all
  code/`<pre>` surfaces; `--font-display` (Space Grotesk) for uppercase micro-labels.
- Syntax highlighting (`CodePane`) uses highlight.js with a curated lazy-registered
  language set + `highlightAuto` fallback; `.hljs-*` colors map to `--syntax-*` tokens.

## Conventions

- Path alias `@/*` → `src/*` (tsconfig). Prefer `@/...` imports over relative `./...`.
- All imports at top of file; comprehensive type hints on props and helpers.
- Data fetching uses TanStack Query (`@tanstack/react-query`); don't hand-roll module
  -level caches.
