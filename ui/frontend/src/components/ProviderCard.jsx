import { useState } from 'react'
import ModelTable from './ModelTable.jsx'
import './ProviderCard.css'

export default function ProviderCard({ provider }) {
  const [expanded, setExpanded] = useState(false)
  const [models, setModels] = useState(null)
  const [modelsLoading, setModelsLoading] = useState(false)
  const [modelsError, setModelsError] = useState(null)
  const [modelSource, setModelSource] = useState(null)
  const [syncing, setSyncing] = useState(false)

  const toggleExpand = () => {
    const next = !expanded
    setExpanded(next)
    if (next && models === null && !modelsLoading) {
      fetchModels()
    }
  }

  const fetchModels = () => {
    setModelsLoading(true)
    setModelsError(null)
    fetch(`/api/providers/${provider.provider}/models`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then(data => {
        setModels(data.models)
        setModelSource(data.source)
        setModelsLoading(false)
      })
      .catch(err => {
        setModelsError(err.message)
        setModelsLoading(false)
      })
  }

  const handleSync = (e) => {
    e.stopPropagation()
    setSyncing(true)
    setModelsError(null)
    fetch(`/api/providers/${provider.provider}/sync`, { method: 'POST' })
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then(data => {
        setModels(data.models)
        setModelSource(data.source)
        if (!data.success && data.error) {
          setModelsError(`Sync failed: ${data.error}`)
        }
        setSyncing(false)
        if (!expanded) setExpanded(true)
      })
      .catch(err => {
        setModelsError(`Sync error: ${err.message}`)
        setSyncing(false)
      })
  }

  const missingItems = [
    ...provider.missing_env_vars.map(v => ({ type: 'env', name: v })),
    ...provider.missing_executables.map(v => ({ type: 'exe', name: v })),
  ]

  return (
    <div className={`provider-card ${expanded ? 'expanded' : ''} ${!provider.available ? 'unavailable' : ''}`}>
      <div className="provider-card-header" onClick={toggleExpand}>
        <div className="provider-card-left">
          <svg
            className={`chevron ${expanded ? 'open' : ''}`}
            width="14" height="14" viewBox="0 0 14 14"
            fill="none" stroke="currentColor" strokeWidth="2"
            strokeLinecap="round" strokeLinejoin="round"
          >
            <polyline points="4,2 9,7 4,12" />
          </svg>
          <span className="provider-name">{provider.provider}</span>
          <div className="provider-badges">
            {provider.supports_structured_output && (
              <span className="badge badge-blue">structured output</span>
            )}
          </div>
        </div>
        <div className="provider-card-right">
          {models !== null && (
            <span className="model-count">
              {models.length} model{models.length !== 1 ? 's' : ''}
            </span>
          )}
          {provider.available && (
            <button
              className={`sync-btn ${syncing ? 'syncing' : ''}`}
              onClick={handleSync}
              disabled={syncing}
              title="Sync models from provider API"
            >
              <svg
                className={`sync-icon ${syncing ? 'spinning' : ''}`}
                width="14" height="14" viewBox="0 0 14 14"
                fill="none" stroke="currentColor" strokeWidth="1.5"
                strokeLinecap="round" strokeLinejoin="round"
              >
                <path d="M1 7a6 6 0 0 1 10.2-4.2M13 7a6 6 0 0 1-10.2 4.2" />
                <polyline points="1,2.5 1,5 3.5,5" />
                <polyline points="13,11.5 13,9 10.5,9" />
              </svg>
              {syncing ? 'Syncing...' : 'Sync'}
            </button>
          )}
        </div>
      </div>

      {expanded && (
        <div className="provider-card-body">
          {missingItems.length > 0 && (
            <div className="missing-reqs">
              {missingItems.map((item, i) => (
                <span key={i} className="missing-tag">
                  <span className="missing-type">{item.type}</span>
                  <code>{item.name}</code>
                </span>
              ))}
            </div>
          )}

          {modelsLoading && (
            <div className="models-loading">
              <div className="spinner-small" />
              <span>Loading models...</span>
            </div>
          )}

          {modelsError && (
            <div className="models-error">
              {modelsError}
            </div>
          )}

          {models !== null && models.length > 0 && (
            <>
              {modelSource === 'static' && (
                <div className="source-badge">
                  Static model list (no live API connection)
                </div>
              )}
              <ModelTable models={models} />
            </>
          )}

          {models !== null && models.length === 0 && !modelsLoading && (
            <div className="no-models">No models found for this provider.</div>
          )}
        </div>
      )}
    </div>
  )
}
