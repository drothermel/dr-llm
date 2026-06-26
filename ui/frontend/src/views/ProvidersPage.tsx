'use client'

import ProviderCard from '@/components/ProviderCard'
import type { ProviderStatus } from '@/lib/types'

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
      <div className="page">
        <div className="page-header">
          <h2>Providers & Models</h2>
        </div>
        <div className="error-state">
          <span className="error-icon">!</span>
          <div>
            <p className="error-title">Failed to load providers</p>
            <p className="error-detail">{initialError}</p>
            <p className="error-hint">
              Make sure the API server is running:{' '}
              <code>uvicorn ui.api.main:app --port 8000</code>
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="page">
      <div className="page-header">
        <h2>Providers & Models</h2>
        <p className="page-description">
          {available.length} of {initialProviders.length} providers available
        </p>
      </div>

      {available.length > 0 && (
        <section className="provider-section">
          <h3 className="section-title">
            <span className="status-dot available" />
            Available
          </h3>
          <div className="provider-grid">
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
        <section className="provider-section">
          <h3 className="section-title">
            <span className="status-dot unavailable" />
            Unavailable
          </h3>
          <div className="provider-grid">
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
