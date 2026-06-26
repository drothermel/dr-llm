function Bar({ className = '' }: { className?: string }) {
  return (
    <span
      className={`block animate-pulse rounded bg-[var(--bg-tertiary)] ${className}`}
    />
  )
}

function PaneSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <section className="min-w-0 overflow-hidden rounded-[10px] border border-[var(--border)] bg-[var(--bg-primary)]">
      <header className="flex items-center gap-2.5 border-b border-[var(--border-subtle)] bg-[var(--bg-secondary)] px-3.5 py-2.5">
        <Bar className="h-3 w-28" />
        <span className="ml-auto">
          <Bar className="h-3 w-16" />
        </span>
      </header>
      <div className="flex flex-col gap-2 p-4">
        {Array.from({ length: rows }).map((_, index) => (
          <Bar key={index} className="h-3.5 w-full" />
        ))}
      </div>
    </section>
  )
}

export default function Loading() {
  return (
    <div className="w-full max-w-none" aria-busy="true">
      <div className="mb-[18px] text-[13px]">
        <span className="font-medium text-[var(--text-secondary)]">
          ← Back to samples
        </span>
      </div>

      <header className="mb-6 flex items-start justify-between gap-5 max-md:flex-col">
        <div className="flex flex-col gap-2">
          <Bar className="h-6 w-72 max-w-full" />
          <Bar className="h-3.5 w-52 max-w-full" />
        </div>
        <Bar className="h-8 w-28" />
      </header>

      <section className="mb-7 overflow-hidden rounded-[10px] border border-[var(--border)] bg-[var(--bg-primary)]">
        <div className="grid grid-cols-[repeat(auto-fit,minmax(190px,1fr))] gap-px border-b border-[var(--border-subtle)] bg-[var(--border-subtle)]">
          {Array.from({ length: 4 }).map((_, index) => (
            <div
              key={index}
              className="flex flex-col gap-2 bg-[var(--bg-primary)] px-[18px] py-4"
            >
              <Bar className="h-3 w-16" />
              <Bar className="h-5 w-24" />
              <Bar className="h-3 w-20" />
            </div>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-y-2 gap-x-[18px] bg-[var(--bg-secondary)] px-[18px] py-3">
          {Array.from({ length: 6 }).map((_, index) => (
            <Bar key={index} className="h-6 w-24" />
          ))}
        </div>
      </section>

      <div className="grid grid-cols-2 gap-y-4 gap-x-5 max-lg:grid-cols-1">
        <PaneSkeleton rows={4} />
        <PaneSkeleton rows={4} />
        <PaneSkeleton rows={6} />
        <PaneSkeleton rows={6} />
        <PaneSkeleton rows={8} />
        <PaneSkeleton rows={8} />
      </div>
    </div>
  )
}
