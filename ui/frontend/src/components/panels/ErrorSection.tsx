import { cn } from '@/lib/cn'

type ErrorSectionProps = {
  label: string
  value: string | null | undefined
  className?: string
}

/** Red error section, full width on failure. */
export function ErrorSection({ label, value, className }: ErrorSectionProps) {
  if (!value) return null
  return (
    <section
      className={cn(
        'overflow-hidden rounded-xl border border-[var(--red-border)] bg-[var(--red-bg)]',
        className,
      )}
    >
      <header className="border-b border-[color-mix(in_oklch,var(--red)_18%,var(--border-subtle))] px-4 py-2.5">
        <h3 className="font-display text-[12px] font-semibold tracking-[0.04em] text-[var(--red)] uppercase">
          {label}
        </h3>
      </header>
      <pre className="m-0 max-h-[320px] overflow-auto px-4 py-3 font-mono text-[12.5px] leading-relaxed whitespace-pre-wrap break-words text-[var(--red)]">
        {value}
      </pre>
    </section>
  )
}
