function Bar({ className = '' }: { className?: string }) {
  return (
    <span
      className={`block animate-pulse rounded bg-[var(--bg-tertiary)] ${className}`}
    />
  )
}

function TextPaneSkeleton({ rows = 4 }: { rows?: number }) {
  return (
    <section className="min-w-0 overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--bg-primary)]">
      <header className="border-b border-[var(--border-subtle)] bg-[var(--bg-secondary)] px-4 py-2.5">
        <Bar className="h-3 w-32" />
      </header>
      <div className="flex flex-col gap-2 px-4 py-3">
        {Array.from({ length: rows }).map((_, index) => (
          <Bar key={index} className="h-3.5 w-full" />
        ))}
      </div>
    </section>
  )
}

function CodePaneSkeleton() {
  return (
    <section className="min-w-0 overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--bg-primary)]">
      <header className="flex items-center gap-2.5 border-b border-[var(--border-subtle)] bg-[var(--bg-secondary)] px-4 py-2.5">
        <Bar className="h-3 w-28" />
        <span className="ml-auto flex gap-1.5">
          <Bar className="h-3 w-12" />
          <Bar className="h-3 w-12" />
          <Bar className="h-3 w-10" />
        </span>
      </header>
      <div className="flex flex-col gap-2 p-4">
        {Array.from({ length: 10 }).map((_, index) => (
          <Bar key={index} className="h-3.5 w-full" />
        ))}
      </div>
    </section>
  )
}

export default function Loading() {
  return (
    <div className="w-full" aria-busy="true">
      <div className="mb-6 max-w-[1280px]">
        <Bar className="h-3.5 w-28" />
      </div>

      {/* Masthead */}
      <header className="mb-8 flex max-w-[1280px] items-start justify-between gap-6 max-md:flex-col">
        <div className="flex flex-col gap-2.5">
          <Bar className="h-3.5 w-56 max-w-full" />
          <Bar className="h-7 w-80 max-w-full" />
        </div>
        <Bar className="h-8 w-32" />
      </header>

      {/* Stat bar */}
      <section className="mb-7 grid max-w-[1280px] grid-cols-4 gap-px border-y border-[var(--border)] bg-[var(--border-subtle)] max-md:grid-cols-2">
        {Array.from({ length: 4 }).map((_, index) => (
          <div
            key={index}
            className="flex flex-col gap-2 bg-[var(--bg-primary)] px-5 py-4"
          >
            <Bar className="h-3 w-16" />
            <Bar className="h-5 w-28" />
            <Bar className="h-3 w-20" />
          </div>
        ))}
      </section>

      {/* Provenance */}
      <section className="mb-8 flex max-w-[1280px] flex-col gap-2.5">
        <Bar className="h-3 w-24" />
        <div className="flex flex-wrap gap-1.5">
          {Array.from({ length: 5 }).map((_, index) => (
            <Bar key={index} className="h-6 w-28" />
          ))}
        </div>
        <Bar className="h-9 w-full" />
        <Bar className="h-4 w-2/3" />
      </section>

      {/* Content */}
      <div className="flex flex-col gap-5">
        <div className="grid max-w-[1280px] grid-cols-2 items-start gap-x-5 gap-y-5 max-lg:grid-cols-1">
          <div className="flex min-w-0 flex-col gap-5">
            <TextPaneSkeleton rows={3} />
            <TextPaneSkeleton rows={2} />
          </div>
          <div className="flex min-w-0 flex-col gap-5">
            <TextPaneSkeleton rows={2} />
            <TextPaneSkeleton rows={5} />
          </div>
        </div>

        <div className="flex flex-col gap-2.5">
          <Bar className="h-3 w-40" />
          <div className="grid grid-cols-2 items-start gap-4 max-lg:grid-cols-1">
            <CodePaneSkeleton />
            <CodePaneSkeleton />
          </div>
        </div>
      </div>
    </div>
  )
}
