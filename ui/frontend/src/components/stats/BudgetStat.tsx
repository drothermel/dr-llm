import { cn } from '@/lib/cn'
import { SECTION_LABEL } from '@/components/primitives'

type BudgetStatProps = {
  budget: string | null
  actual: number | null
  budgetOk: boolean | null
  className?: string
}

/** Budget stat with a used/limit meter that turns red when over budget. */
export function BudgetStat({
  budget,
  actual,
  budgetOk,
  className,
}: BudgetStatProps) {
  const limit = Number(budget)
  const used = actual === null || actual === undefined ? null : Number(actual)
  const hasMeter = Number.isFinite(limit) && limit > 0 && used !== null
  const ratio = hasMeter ? used / limit : 0
  const over = budgetOk === false || (hasMeter && used > limit)

  return (
    <div
      className={cn(
        'flex min-w-0 flex-col gap-1.5 bg-[var(--accent-bg)] px-5 py-4',
        className,
      )}
    >
      <span className={SECTION_LABEL}>Budget</span>
      <span className="flex items-baseline gap-1.5">
        <strong className="font-mono text-xl leading-none font-semibold text-[var(--text-primary)]">
          {used ?? '—'}
        </strong>
        <span className="font-mono text-[13px] text-[var(--text-secondary)]">
          / {Number.isFinite(limit) ? limit : (budget ?? '—')} chars
        </span>
      </span>
      {hasMeter && (
        <span
          className="mt-0.5 block h-1.5 overflow-hidden rounded-full bg-[color-mix(in_oklch,var(--accent)_18%,transparent)]"
          aria-hidden="true"
        >
          <span
            className={cn(
              'block h-full rounded-full transition-[width] duration-200',
              over ? 'bg-[var(--red)]' : 'bg-[var(--accent)]',
            )}
            style={{ width: `${Math.min(100, Math.max(2, ratio * 100))}%` }}
          />
        </span>
      )}
      <span
        className={cn(
          'text-xs font-medium',
          over ? 'text-[var(--red)]' : 'text-[var(--accent-strong)]',
        )}
      >
        {over
          ? 'over budget'
          : hasMeter
            ? `${Math.round(ratio * 100)}% of budget`
            : 'within budget'}
      </span>
    </div>
  )
}
