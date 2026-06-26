import { cn } from '@/lib/cn'
import { SECTION_LABEL } from '@/components/primitives'

type StatCellProps = {
  label: string
  value: string | null
  sub?: string | null
  mono?: boolean
  className?: string
}

/** Single labeled stat in a flat, hairline-bounded stat bar. */
export function StatCell({ label, value, sub, mono, className }: StatCellProps) {
  return (
    <div
      className={cn(
        'flex min-w-0 flex-col gap-1.5 bg-[var(--bg-primary)] px-5 py-4',
        className,
      )}
    >
      <span className={SECTION_LABEL}>{label}</span>
      <span
        className={cn(
          '[overflow-wrap:anywhere] text-sm leading-snug font-semibold text-[var(--text-primary)]',
          mono && 'font-mono',
        )}
      >
        {value ?? '—'}
      </span>
      {sub && (
        <span className="[overflow-wrap:anywhere] text-xs text-[var(--text-secondary)]">
          {sub}
        </span>
      )}
    </div>
  )
}
