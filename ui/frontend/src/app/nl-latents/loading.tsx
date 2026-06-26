const COLUMN_HEADERS = [
  'Sample',
  'Family',
  'Diff',
  'Split',
  'Model',
  'Budget',
  'Prompt config',
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
    <div className="w-full max-w-none" aria-busy="true">
      <div className="mb-8 flex items-start justify-between gap-6 max-md:block">
        <div>
          <h2 className="mb-1 text-2xl font-bold text-[var(--text-primary)]">
            nl-latents
          </h2>
          <p className="text-sm text-[var(--text-secondary)]">
            Published sample summaries from published_nl_latents_samples
          </p>
        </div>
        <span className="shrink-0 rounded-md border border-[var(--border)] px-2 py-1 font-mono text-xs text-[var(--text-secondary)] max-md:mt-3 max-md:inline-flex">
          published_nl_latents_samples
        </span>
      </div>

      <div className="mb-6 grid grid-cols-[repeat(auto-fit,minmax(150px,1fr))] gap-3">
        {Array.from({ length: 6 }).map((_, index) => (
          <div key={index} className="flex flex-col gap-1">
            <Bar className="h-3 w-16" />
            <Bar className="h-[34px] w-full" />
          </div>
        ))}
      </div>

      <div className="overflow-x-auto rounded-lg border border-[var(--border)]">
        <div className="flex items-center justify-between gap-3 border-b border-[var(--border-subtle)] px-3 py-2.5 text-[13px] text-[var(--text-secondary)]">
          <span>Loading…</span>
        </div>
        <table className="w-full min-w-[1120px] border-collapse">
          <thead>
            <tr>
              {COLUMN_HEADERS.map(header => (
                <th
                  key={header}
                  className="border-b border-[var(--border-subtle)] bg-[var(--bg-secondary)] px-3 py-2.5 text-left align-top text-[11px] font-bold tracking-[0.5px] text-[var(--text-secondary)] uppercase"
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
                    className="border-b border-[var(--border-subtle)] px-3 py-2.5 align-top"
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
