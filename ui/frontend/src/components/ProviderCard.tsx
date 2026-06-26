'use client'

import { useState } from 'react'
import ModelTable from './ModelTable'
import { Tag } from './primitives'
import useProviderModels from '@/hooks/useProviderModels'
import type { ProviderStatus } from '@/lib/types'

type ProviderCardProps = {
  provider: ProviderStatus
}

export default function ProviderCard({ provider }: ProviderCardProps) {
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
    ...provider.missing_env_vars.map(value => ({
      type: 'env',
      name: value,
    })),
    ...provider.missing_executables.map(value => ({
      type: 'exe',
      name: value,
    })),
  ]
  const cardClass = [
    'overflow-hidden rounded-xl border bg-[var(--bg-primary)] transition-colors',
    expanded
      ? 'border-[color-mix(in_oklch,var(--accent)_45%,var(--border))]'
      : 'border-[var(--border)] hover:border-[var(--border-strong)]',
    provider.available ? '' : 'opacity-70',
  ].join(' ')

  return (
    <div className={cardClass}>
      <div className="flex items-center gap-3">
        <h3 className="m-0 min-w-0 flex-1">
          <button
            id={toggleId}
            type="button"
            className="flex w-full min-w-0 cursor-pointer items-center justify-between gap-3 border-0 bg-transparent px-[18px] py-3.5 text-left select-none hover:bg-[var(--bg-hover)]"
            onClick={toggleExpand}
            aria-expanded={expanded}
            aria-controls={bodyId}
          >
            <div className="flex min-w-0 items-center gap-2.5">
              <svg
                className={`shrink-0 text-[var(--text-muted)] transition-transform ${expanded ? 'rotate-90' : ''}`}
                width="14"
                height="14"
                viewBox="0 0 14 14"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <polyline points="4,2 9,7 4,12" />
              </svg>
              <span className="font-mono text-[15px] font-semibold text-[var(--text-primary)]">
                {provider.provider}
              </span>
              <div className="flex gap-1">
                {provider.supports_structured_output && (
                  <Tag tone="blue" className="tracking-[0.04em] uppercase">
                    structured output
                  </Tag>
                )}
              </div>
            </div>
            <div className="flex shrink-0 items-center gap-3">
              {models !== null && (
                <span className="font-mono text-xs text-[var(--text-muted)]">
                  {models.length} model{models.length !== 1 ? 's' : ''}
                </span>
              )}
            </div>
          </button>
        </h3>
        {provider.available && (
          <div className="flex items-center pr-[18px]">
            <button
              type="button"
              className="flex cursor-pointer items-center gap-1 rounded-md border border-[var(--border)] bg-transparent px-3 py-1.5 text-xs font-medium text-[var(--text-secondary)] transition-all hover:border-[var(--accent)] hover:bg-[var(--accent-bg)] hover:text-[var(--accent)] disabled:cursor-not-allowed disabled:opacity-50"
              onClick={() => void handleSync()}
              disabled={syncing}
              title="Sync models from provider API"
              aria-label={`Sync models for ${provider.provider}`}
            >
              <svg
                className={syncing ? 'animate-spin' : ''}
                width="14"
                height="14"
                viewBox="0 0 14 14"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
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
        <div
          className="border-t border-[var(--border-subtle)] px-[18px] pt-0 pb-4"
          id={bodyId}
          role="region"
          aria-labelledby={toggleId}
        >
          {missingItems.length > 0 && (
            <div className="flex flex-wrap gap-1.5 py-3">
              {missingItems.map(item => (
                <span
                  key={`${item.type}:${item.name}`}
                  className="inline-flex items-center gap-1 rounded border border-[rgba(220,38,38,0.1)] bg-[var(--red-bg)] px-2 py-1 text-xs text-[var(--text-secondary)]"
                >
                  <span className="text-[10px] font-bold tracking-[0.5px] text-[var(--red)] uppercase">
                    {item.type}
                  </span>
                  <code className="font-mono text-[11px] text-[var(--text-primary)]">
                    {item.name}
                  </code>
                </span>
              ))}
            </div>
          )}

          {modelsLoading && (
            <div className="flex items-center gap-2.5 py-4 text-[13px] text-[var(--text-secondary)]">
              <div className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-[var(--border)] border-t-[var(--accent)]" />
              <span>Loading models...</span>
            </div>
          )}

          {modelsError && (
            <div className="mt-2.5 rounded-md border border-[rgba(220,38,38,0.1)] bg-[var(--red-bg)] px-3.5 py-2.5 text-[13px] text-[var(--red)]">
              {modelsError}
            </div>
          )}

          {models !== null && models.length > 0 && (
            <>
              {modelSource === 'static' && (
                <div className="my-3 mb-2 inline-block rounded bg-[var(--yellow-bg)] px-2.5 py-1 text-[11px] text-[var(--yellow)]">
                  Static model list (no live API connection)
                </div>
              )}
              <ModelTable models={models} />
            </>
          )}

          {models !== null && models.length === 0 && !modelsLoading && (
            <div className="py-4 text-[13px] text-[var(--text-muted)]">
              No models found for this provider.
            </div>
          )}
        </div>
      )}
    </div>
  )
}
