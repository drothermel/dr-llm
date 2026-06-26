import { cn } from '@/lib/cn'
import { truncate } from '@/lib/format'

type IdChipProps = {
  label: string
  value: string | number | null | undefined
  display?: string | null
  copied: string | null
  onCopy: (value: string) => void
  className?: string
}

/** Compact label + value chip that copies its value on click. */
export function IdChip({
  label,
  value,
  display,
  copied,
  onCopy,
  className,
}: IdChipProps) {
  if (value === null || value === undefined || value === '') return null
  const stringValue = String(value)
  const isCopied = copied === stringValue
  return (
    <button
      type="button"
      className={cn(
        'group inline-flex cursor-pointer items-baseline gap-1.5 rounded px-1.5 py-0.5 text-left transition-colors hover:bg-[var(--bg-hover)]',
        className,
      )}
      title={`${label}: ${stringValue} (click to copy)`}
      onClick={() => onCopy(stringValue)}
    >
      <span className="text-[10px] font-semibold tracking-[0.06em] text-[var(--text-muted)] uppercase">
        {label}
      </span>
      <span className="font-mono text-[12px] text-[var(--text-secondary)] group-hover:text-[var(--text-primary)]">
        {isCopied ? 'copied ✓' : (display ?? truncate(stringValue, 14))}
      </span>
    </button>
  )
}
