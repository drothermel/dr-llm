import type { ReactNode } from 'react'

/** Space Grotesk uppercase micro-label used across every surface. */
export const SECTION_LABEL =
  'font-display text-[11px] font-semibold tracking-[0.08em] text-[var(--text-muted)] uppercase'

const STATE_BADGE: Record<string, string> = {
  passed: 'bg-[var(--green-bg)] text-[var(--green)]',
  failed: 'bg-[var(--red-bg)] text-[var(--red)]',
  pending: 'bg-[var(--yellow-bg)] text-[var(--yellow)]',
}

/** Small separator dot. */
export function Dot() {
  return (
    <span
      aria-hidden="true"
      className="h-[3px] w-[3px] shrink-0 rounded-full bg-[var(--border-strong)]"
    />
  )
}

type ResultBadgeProps = {
  state: string
  failure?: string | null
  size?: 'sm' | 'md'
}

/** Canonical pass/fail/pending badge — identical on every surface. */
export function ResultBadge({ state, failure, size = 'md' }: ResultBadgeProps) {
  const sized =
    size === 'sm'
      ? 'gap-1.5 px-2 py-1 text-[11px]'
      : 'gap-2 px-3 py-1.5 text-[13px]'
  return (
    <span
      className={`inline-flex shrink-0 items-center rounded-md leading-none font-semibold ${sized} ${
        STATE_BADGE[state] ?? 'bg-[var(--bg-tertiary)] text-[var(--text-secondary)]'
      }`}
    >
      <span
        className="h-[6px] w-[6px] shrink-0 rounded-full bg-current"
        aria-hidden="true"
      />
      <span className="capitalize">{state}</span>
      {failure && state === 'failed' && (
        <span className="border-l border-current/30 pl-1.5 font-mono text-[11px] font-medium">
          {failure}
        </span>
      )}
    </span>
  )
}

type TagTone = 'neutral' | 'accent' | 'blue' | 'green' | 'yellow' | 'red'

const TAG_TONE: Record<TagTone, string> = {
  neutral:
    'border border-[var(--border-subtle)] bg-[var(--bg-tertiary)] text-[var(--text-secondary)]',
  accent: 'bg-[var(--accent-bg)] text-[var(--accent)]',
  blue: 'bg-[var(--blue-bg)] text-[var(--blue)]',
  green: 'bg-[var(--green-bg)] text-[var(--green)]',
  yellow: 'bg-[var(--yellow-bg)] text-[var(--yellow)]',
  red: 'bg-[var(--red-bg)] text-[var(--red)]',
}

type TagProps = {
  children: ReactNode
  tone?: TagTone
  mono?: boolean
  className?: string
}

/** Small inline label chip. */
export function Tag({
  children,
  tone = 'neutral',
  mono = false,
  className = '',
}: TagProps) {
  return (
    <span
      className={`inline-flex items-center rounded px-2 py-0.5 text-[11px] font-medium whitespace-nowrap ${
        mono ? 'font-mono' : ''
      } ${TAG_TONE[tone]} ${className}`}
    >
      {children}
    </span>
  )
}
