import { cn } from '@/lib/cn'

type TextPanelProps = {
  label: string
  value: string | null | undefined
  className?: string
}

/** Plain reference text panel (prompts) — always visible, no char count. */
export function TextPanel({ label, value, className }: TextPanelProps) {
  if (!value) return null
  return (
    <section
      className={cn(
        'overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--bg-primary)]',
        className,
      )}
    >
      <header className="border-b border-[var(--border-subtle)] bg-[var(--bg-secondary)] px-4 py-2.5">
        <h3 className="font-display text-[12px] font-semibold tracking-[0.04em] text-[var(--text-secondary)] uppercase">
          {label}
        </h3>
      </header>
      <pre className="m-0 overflow-auto px-4 py-3 font-mono text-[12.5px] leading-relaxed whitespace-pre-wrap break-words text-[var(--text-secondary)]">
        {value}
      </pre>
    </section>
  )
}
