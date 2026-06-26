# dr-llm UI Redesign — Design

Date: 2026-06-26
Status: implemented (all three surfaces); shared primitives in `components/primitives.tsx`

## Context

The UI was migrated from Vite + Tailwind v3 to Next.js + Tailwind v4. The migration
left the UI badly broken; root cause was an unlayered `* { margin:0; padding:0 }`
reset in `globals.css` overriding all Tailwind v4 spacing utilities (unlayered CSS
beats `@layer`). That is fixed (reset moved into `@layer base`).

Building on the restored parity, this is a deliberate redesign of the three surfaces
following the impeccable **product** register: design serves the task, earned
familiarity, density where it helps.

## Direction & identity

- **Register:** Editorial-analytical, light, Linear/Stripe-grade. Evolve the existing
  light token system rather than reinvent (committed indigo `--accent` is preserved).
- **Type system:**
  - Display / headings / `dr-llm` mark: **Space Grotesk**
  - Body / labels / table data: **Hanken Grotesk**
  - Code panes / IDs / model names / numeric metrics: **Fira Code** (ligatures on)
  - Self-hosted via `next/font` (no layout shift). Vars: `--font-display`,
    `--font-sans`, `--font-mono`.
- **Type scale:** fixed rem, ratio ~1.2 (`12 / 13 / 14 / 16 / 20 / 26px`). Display
  tracking `-0.02em` (never below −0.04em). Body 14px / 1.55. Uppercase micro-labels
  tracked `0.06em`.
- **Color:** keep indigo accent. Refine neutral ramp (slightly cooler, deliberate);
  verify body text ≥4.5:1 against tinted surfaces. One standardized semantic badge
  vocabulary (pass / fail / pending). Add state vocabulary:
  hover/focus/active/disabled/selected/loading.
- **Surfaces:** one content surface + one cooler chrome layer (sidebar, pane headers).
  Radius capped 10–12px. Hairline borders; no border+wide-shadow ghost-card pairing.
- **Motion:** state-only, 150–200ms ease-out (hover/focus, copy confirm, expand). No
  page-load choreography. Full `prefers-reduced-motion` fallback.

## Sequencing

1. **Sample detail page** — reference surface; establishes tokens + shared components.
2. **nl-latents table** — reuse tokens/components.
3. **Providers & Models list** — reuse tokens/components.

## Sample detail page layout

Keeps every field; removes box-in-box density (no nested cards).

1. **Masthead** — quiet "← Back"; context line (`family` · difficulty · split ·
   language, family bold); `task_id` as H1 in Fira Code; `ResultBadge` top-right with
   failure category. Space Grotesk for section labels.
2. **Stat bar** — budget / encoder / decoder / validation as ONE flat hairline-bounded
   row with vertical-rule dividers (no nested boxes). Budget keeps inline meter; numbers
   in Fira Code.
3. **Provenance line** — prompt-config chips + copyable IDs (sample/config/call/run/
   created/finish/version) in one quiet monospace row under a `Provenance` label;
   copy-on-click preserved, visually recessive.
4. **Content, priority order:**
   - **Encoded output** (NL latent) — promoted centerpiece, full-width, accent-framed.
   - **Error detail** — red, directly under, when failed.
   - **Prompts** (encoder / decoder-system / decoder-task) — quieter, collapsible
     reference sections.
   - **Input code → Decoded code** — climax: side-by-side editor containers (border +
     header with line/char/KB stats), framed input vs decoded. Fira Code + ligatures.
   - Responsive: stat bar wraps; code panes stack below `lg`.

## Out of scope (for now)

- Backend/API changes.
- New data or endpoints.
- The 404 on direct truncated-ID navigation is expected (display truncates to 12 chars;
  the link uses the full `sample_id`).
