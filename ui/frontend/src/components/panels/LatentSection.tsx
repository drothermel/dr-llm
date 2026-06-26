import { cn } from '@/lib/cn'
import { formatBytes } from '@/lib/format'

type LatentSectionProps = {
  label: string
  value: string | null | undefined
  className?: string
}

/** Prose-forward, accent-framed section with char + byte counts. */
export function LatentSection({ label, value, className }: LatentSectionProps) {
  if (!value) return null
  const chars = [...value].length
  const bytes = new TextEncoder().encode(value).length
  return (
    <section
      className={cn(
        'overflow-hidden rounded-xl border border-[color-mix(in_oklch,var(--accent)_28%,var(--border))] bg-[var(--accent-bg)]',
        className,
      )}
    >
      <header className="flex items-center gap-2.5 border-b border-[color-mix(in_oklch,var(--accent)_18%,var(--border-subtle))] px-4 py-2.5">
        <h3 className="font-display text-[12px] font-semibold tracking-[0.04em] text-[var(--accent-strong)] uppercase">
          {label}
        </h3>
        <div className="ml-auto flex items-center gap-1.5">
          {[`${chars.toLocaleString()} chars`, formatBytes(bytes)].map(text => (
            <span
              key={text}
              className="rounded bg-[color-mix(in_oklch,var(--accent)_12%,transparent)] px-[7px] py-0.5 font-mono text-[11px] whitespace-nowrap text-[var(--accent-strong)]"
            >
              {text}
            </span>
          ))}
        </div>
      </header>
      <p className="px-5 py-4 text-[15px] leading-[1.65] text-[var(--text-primary)]">
        {value}
      </p>
    </section>
  )
}
