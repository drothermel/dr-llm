'use client'

import ProviderCard from '@/components/ProviderCard'
import type { ProviderStatus } from '@/lib/types'

const PAGE_CLASS = 'max-w-[1100px]'
const PAGE_HEADER_CLASS = 'mb-8'
const PAGE_TITLE_CLASS =
  'mb-1 font-display text-[26px] font-bold leading-tight tracking-[-0.02em] text-[var(--text-primary)]'
const PAGE_DESCRIPTION_CLASS = 'text-sm text-[var(--text-secondary)]'
const ERROR_STATE_CLASS =
  'flex items-start gap-4 rounded-xl border border-[var(--red-border)] bg-[var(--red-bg)] p-6'
const ERROR_ICON_CLASS =
  'flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[var(--red)] text-sm font-bold text-white'
const SECTION_TITLE_CLASS =
  'mb-3 flex items-center gap-2 font-display text-[12px] font-semibold tracking-[0.08em] text-[var(--text-secondary)] uppercase'

type ProvidersPageProps = {
  initialProviders: ProviderStatus[]
  initialError: string | null
}

export default function ProvidersPage({
  initialProviders,
  initialError,
}: ProvidersPageProps) {
  const available = initialProviders.filter(provider => provider.available)
  const unavailable = initialProviders.filter(provider => !provider.available)

  if (initialError) {
    return (
      <div className={PAGE_CLASS}>
        <div className={PAGE_HEADER_CLASS}>
          <h2 className={PAGE_TITLE_CLASS}>Providers & Models</h2>
        </div>
        <div className={ERROR_STATE_CLASS}>
          <span className={ERROR_ICON_CLASS}>!</span>
          <div>
            <p className="mb-1 font-semibold text-[var(--red)]">
              Failed to load providers
            </p>
            <p className="mb-2 text-[13px] text-[var(--text-secondary)]">
              {initialError}
            </p>
            <p className="text-[13px] text-[var(--text-muted)]">
              Make sure the API server is running:{' '}
              <code className="rounded bg-[var(--bg-tertiary)] px-1.5 py-0.5 font-mono text-xs text-[var(--text-secondary)]">
                uvicorn ui.api.main:app --port 8000
              </code>
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={PAGE_CLASS}>
      <div className={PAGE_HEADER_CLASS}>
        <h2 className={PAGE_TITLE_CLASS}>Providers & Models</h2>
        <p className={PAGE_DESCRIPTION_CLASS}>
          {available.length} of {initialProviders.length} providers available
        </p>
      </div>

      {available.length > 0 && (
        <section className="mb-8">
          <h3 className={SECTION_TITLE_CLASS}>
            <span className="h-2 w-2 rounded-full bg-[var(--green)] shadow-[0_0_6px_rgba(22,163,74,0.3)]" />
            Available
          </h3>
          <div className="flex flex-col gap-2">
            {available.map(provider => (
              <ProviderCard
                key={provider.provider}
                provider={provider}
              />
            ))}
          </div>
        </section>
      )}

      {unavailable.length > 0 && (
        <section className="mb-8">
          <h3 className={SECTION_TITLE_CLASS}>
            <span className="h-2 w-2 rounded-full bg-[var(--text-muted)]" />
            Unavailable
          </h3>
          <div className="flex flex-col gap-2">
            {unavailable.map(provider => (
              <ProviderCard
                key={provider.provider}
                provider={provider}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
