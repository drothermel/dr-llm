# Component theme contract

The components in this directory (`code/`, `stats/`, `panels/`, `chips/`,
`primitives.tsx`) are presentational and portable: they hold no domain types and
style themselves entirely through CSS custom properties. To reuse them in another
project, define the tokens below on `:root` (see `src/app/globals.css` for the
reference values) and import the components. Every component also accepts a
trailing `className` prop, merged last via `cn()`, so callers can override or
extend any style.

## Required tokens

### Surfaces
- `--bg-primary` — main content background
- `--bg-secondary` — panel/header backgrounds
- `--bg-tertiary` — chips, stat badges
- `--bg-hover` — interactive hover background
- `--bg-code` — `<pre>` code surface (CodePane)

### Borders
- `--border` — standard hairline
- `--border-subtle` — low-contrast dividers
- `--border-strong` — emphasis borders

### Text
- `--text-primary` — body text / headings
- `--text-secondary` — supporting copy
- `--text-muted` — labels, placeholders

### Accent
- `--accent`, `--accent-hover`, `--accent-strong` — interactive color ramp
- `--accent-bg` — low-saturation accent surface

### Semantic
- `--green`, `--green-bg` — success / passed
- `--red`, `--red-bg`, `--red-border` — error / failed
- `--yellow`, `--yellow-bg` — warning / pending
- `--blue`, `--blue-bg` — info

### Syntax (CodePane, via highlight.js `.hljs-*` classes)
- `--syntax-keyword`, `--syntax-string`, `--syntax-number`, `--syntax-function`,
  `--syntax-builtin`, `--syntax-comment`, `--syntax-punctuation`

### Typography
- `--font-display` — uppercase micro-labels / headings
- `--font-mono` — code, IDs, numeric stats

The `.hljs` / `.hljs-*` color rules and font-feature (ligature) settings live in
`globals.css`; copy that block alongside the tokens when porting CodePane.
