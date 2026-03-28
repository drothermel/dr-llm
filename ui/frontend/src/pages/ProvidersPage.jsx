import { useState, useEffect } from 'react'
import ProviderCard from '../components/ProviderCard.jsx'
import './ProvidersPage.css'

export default function ProvidersPage() {
  const [providers, setProviders] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/api/providers')
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then(data => {
        setProviders(data)
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [])

  const available = providers.filter(p => p.available)
  const unavailable = providers.filter(p => !p.available)

  if (loading) {
    return (
      <div className="page">
        <div className="page-header">
          <h2>Providers & Models</h2>
        </div>
        <div className="loading-state">
          <div className="spinner" />
          <span>Loading providers...</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="page">
        <div className="page-header">
          <h2>Providers & Models</h2>
        </div>
        <div className="error-state">
          <span className="error-icon">!</span>
          <div>
            <p className="error-title">Failed to load providers</p>
            <p className="error-detail">{error}</p>
            <p className="error-hint">
              Make sure the API server is running: <code>uvicorn ui.api.main:app --port 8000</code>
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
          {available.length} of {providers.length} providers available
        </p>
      </div>

      {available.length > 0 && (
        <section className="provider-section">
          <h3 className="section-title">
            <span className="status-dot available" />
            Available
          </h3>
          <div className="provider-grid">
            {available.map(p => (
              <ProviderCard key={p.provider} provider={p} />
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
            {unavailable.map(p => (
              <ProviderCard key={p.provider} provider={p} />
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
