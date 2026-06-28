const COLUMN_HEADERS = [
  'Sample',
  'Project',
  'Pool',
  'Role',
  'Task',
  'Model',
  'Result',
  'Created',
]

function Bar({ className = '' }: { className?: string }) {
  return (
    <span
      className={`block animate-pulse rounded bg-[var(--bg-tertiary)] ${className}`}
    />
  )
}

export default function Loading() {
  return (
    <div className="w-full" aria-busy="true">
      <div className="mb-7">
        <h1 className="font-display text-[26px] leading-tight font-bold text-[var(--text-primary)]">
          Published samples
        </h1>
        <p className="mt-1 text-sm text-[var(--text-secondary)]">
          Unified summaries from{' '}
          <span className="font-mono text-[var(--text-primary)]">
            published_pool_samples
          </span>
        </p>
      </div>

      <div className="mb-6 grid grid-cols-[repeat(auto-fit,minmax(150px,1fr))] gap-3">
        {Array.from({ length: 7 }).map((_, index) => (
          <div key={index} className="flex flex-col gap-1.5">
            <Bar className="h-3 w-16" />
            <Bar className="h-[34px] w-full" />
          </div>
        ))}
      </div>

      <div className="overflow-x-auto rounded-xl border border-[var(--border)]">
        <div className="flex items-center justify-between gap-3 border-b border-[var(--border-subtle)] bg-[var(--bg-secondary)] px-4 py-2.5 text-[13px] text-[var(--text-secondary)]">
          <span>Loading</span>
        </div>
        <table className="w-full min-w-[1120px] border-collapse">
          <thead>
            <tr>
              {COLUMN_HEADERS.map(header => (
                <th
                  key={header}
                  className="border-b border-[var(--border)] bg-[var(--bg-secondary)] px-4 py-2.5 text-left align-middle font-display text-[11px] font-semibold tracking-[0.06em] text-[var(--text-muted)] uppercase"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: 12 }).map((_, rowIndex) => (
              <tr key={rowIndex} className="last:[&>td]:border-b-0">
                {COLUMN_HEADERS.map((_, cellIndex) => (
                  <td
                    key={cellIndex}
                    className="border-b border-[var(--border-subtle)] px-4 py-2.5 align-middle"
                  >
                    <Bar className="h-4 w-full max-w-[120px]" />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
