import { useState } from 'react'
import ModelTable from './ModelTable.jsx'
import useProviderModels from '../hooks/useProviderModels.js'
import './ProviderCard.css'

export default function ProviderCard({ provider }) {
  const [expanded, setExpanded] = useState(false)
  const {
    models,
    modelsLoading,
    modelsError,
    modelSource,
    syncing,
    loadModels,
    syncModels,
  } = useProviderModels(provider.provider)
  const bodyId = `provider-${provider.provider.replace(/[^a-zA-Z0-9_-]/g, '-')}-models`
  const toggleId = `${bodyId}-toggle`

  const toggleExpand = () => {
    const next = !expanded
    setExpanded(next)
    if (next) {
      void loadModels()
    }
  }

  const handleSync = async () => {
    setExpanded(true)
    await syncModels()
  }

  const missingItems = [
    ...provider.missing_env_vars.map(v => ({ type: 'env', name: v })),
    ...provider.missing_executables.map(v => ({ type: 'exe', name: v })),
  ]

  return (
    <div className={`provider-card ${expanded ? 'expanded' : ''} ${!provider.available ? 'unavailable' : ''}`}>
      <div className="provider-card-header">
        <h3 className="provider-card-heading">
          <button
            id={toggleId}
            type="button"
            className="provider-card-toggle"
            onClick={toggleExpand}
            aria-expanded={expanded}
            aria-controls={bodyId}
          >
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
            </div>
          </button>
        </h3>
        {provider.available && (
          <div className="provider-card-actions">
            <button
              type="button"
              className={`sync-btn ${syncing ? 'syncing' : ''}`}
              onClick={() => void handleSync()}
              disabled={syncing}
              title="Sync models from provider API"
              aria-label={`Sync models for ${provider.provider}`}
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
          </div>
        )}
      </div>

      {expanded && (
        <div className="provider-card-body" id={bodyId} role="region" aria-labelledby={toggleId}>
          {missingItems.length > 0 && (
            <div className="missing-reqs">
              {missingItems.map(item => (
                <span key={`${item.type}:${item.name}`} className="missing-tag">
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
