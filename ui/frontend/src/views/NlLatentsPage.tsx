'use client'

import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { useEffect, useMemo, useState, useTransition } from 'react'
import {
  ALL,
  DEFAULT_PAGE,
  buildSamplesPath,
  listStateToSearchParams,
  parseListState,
  type NlLatentsListState,
} from '@/lib/nlLatents'
import { fetchJson } from '@/lib/http'
import type {
  NlLatentsFilters,
  NlLatentsSamplesResponse,
} from '@/lib/types'

const RESULT_OPTIONS = ['passed', 'failed', 'pending']
const FILTERS_PATH = '/api/nl-latents/filters'

type CachedJson = NlLatentsFilters | NlLatentsSamplesResponse

const jsonCache = new Map<string, CachedJson>()
const jsonInFlight = new Map<string, Promise<CachedJson>>()

async function fetchCachedJson<T extends CachedJson>(
  path: string,
): Promise<T> {
  if (jsonCache.has(path)) {
    return jsonCache.get(path) as T
  }
  if (jsonInFlight.has(path)) {
    return jsonInFlight.get(path) as Promise<T>
  }

  const request = fetchJson<T>(path)
    .then(data => {
      jsonCache.set(path, data)
      jsonInFlight.delete(path)
      return data
    })
    .catch((error: unknown) => {
      jsonInFlight.delete(path)
      throw error
    })

  jsonInFlight.set(path, request)
  return request
}

function prefetchSamplesPage(state: NlLatentsListState, page: number) {
  if (page < DEFAULT_PAGE) return
  fetchCachedJson<NlLatentsSamplesResponse>(
    buildSamplesPath({ ...state, page }),
  ).catch(() => {})
}

type SelectFilterProps = {
  label: string
  value: string
  values: string[]
  onChange: (value: string) => void
}

function SelectFilter({
  label,
  value,
  values,
  onChange,
}: SelectFilterProps) {
  return (
    <label className="nl-filter">
      <span>{label}</span>
      <select
        value={value}
        onChange={event => onChange(event.target.value)}
      >
        <option value={ALL}>All</option>
        {values.map(option => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  )
}

type ResultBadgeProps = {
  state: string
  failureCategory: string | null
}

function ResultBadge({ state, failureCategory }: ResultBadgeProps) {
  const className = `nl-result nl-result-${state}`
  return (
    <span className={className}>
      {state}
      {failureCategory ? ` · ${failureCategory}` : ''}
    </span>
  )
}

type NlLatentsPageProps = {
  initialError: string | null
  initialFilters: NlLatentsFilters | null
  initialListState: NlLatentsListState
  initialSamples: NlLatentsSamplesResponse | null
  initialSamplesPath: string
}

export default function NlLatentsPage({
  initialError,
  initialFilters,
  initialListState,
  initialSamples,
  initialSamplesPath,
}: NlLatentsPageProps) {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [isPending, startTransition] = useTransition()
  const [filters, setFilters] = useState<NlLatentsFilters | null>(
    initialFilters,
  )
  const [filtersError, setFiltersError] = useState<string | null>(null)
  const [samplesResult, setSamplesResult] = useState<{
    path: string | null
    data: NlLatentsSamplesResponse | null
    error: string | null
  }>({
    path: initialSamplesPath,
    data: initialSamples,
    error: null,
  })

  useEffect(() => {
    if (initialFilters) {
      jsonCache.set(FILTERS_PATH, initialFilters)
    }
    if (initialSamples) {
      jsonCache.set(initialSamplesPath, initialSamples)
    }
  }, [initialFilters, initialSamples, initialSamplesPath])

  const listState = useMemo(() => {
    if (!searchParams) {
      return initialListState
    }
    const parsedState = parseListState(searchParams)
    return searchParams.size === 0 ? initialListState : parsedState
  }, [initialListState, searchParams])

  const {
    page,
    family,
    difficulty,
    split,
    encModel,
    budget,
    dataVersion,
    result,
    hidePending,
    hideSmoke,
  } = listState

  useEffect(() => {
    if (filters) {
      return
    }

    let active = true
    fetchCachedJson<NlLatentsFilters>(FILTERS_PATH)
      .then(data => {
        if (active) setFilters(data)
      })
      .catch((error: unknown) => {
        if (!active) return
        setFiltersError(error instanceof Error ? error.message : String(error))
      })
    return () => {
      active = false
    }
  }, [filters])

  const queryPath = useMemo(() => buildSamplesPath(listState), [listState])

  useEffect(() => {
    if (samplesResult.path === queryPath) {
      return
    }

    let active = true
    fetchCachedJson<NlLatentsSamplesResponse>(queryPath)
      .then(data => {
        if (!active) return
        setSamplesResult({ path: queryPath, data, error: null })
      })
      .catch((error: unknown) => {
        if (!active) return
        const message = error instanceof Error ? error.message : String(error)
        setSamplesResult({ path: queryPath, data: null, error: message })
      })
    return () => {
      active = false
    }
  }, [queryPath, samplesResult.path])

  useEffect(() => {
    const samples =
      samplesResult.path === queryPath ? samplesResult.data : null
    if (!samples) return

    prefetchSamplesPage(listState, page - 1)
    if (page < samples.total_pages) {
      prefetchSamplesPage(listState, page + 1)
    }
  }, [listState, page, queryPath, samplesResult])

  const updateListState = (updates: Partial<NlLatentsListState>) => {
    const params = listStateToSearchParams({ ...listState, ...updates })
    const suffix = params.toString()
    startTransition(() => {
      router.push(suffix ? `/nl-latents?${suffix}` : '/nl-latents')
    })
  }

  const updateFilter =
    (key: keyof NlLatentsListState) => (value: string) => {
      updateListState({ [key]: value, page: DEFAULT_PAGE })
    }

  const sampleSearch = searchParams?.toString() ?? ''
  const sampleSearchSuffix = sampleSearch ? `?${sampleSearch}` : ''
  const samples =
    samplesResult.path === queryPath ? samplesResult.data : null
  const samplesError =
    samplesResult.path === queryPath ? samplesResult.error : null
  const loading = isPending || samplesResult.path !== queryPath
  const error = initialError ?? samplesError ?? filtersError
  const pageStatus = samples
    ? `Page ${samples.page} of ${samples.total_pages}`
    : null

  return (
    <div className="page nl-page">
      <div className="page-header nl-page-header">
        <div>
          <h2>nl-latents</h2>
          <p className="page-description">
            Published sample summaries from published_nl_latents_samples
          </p>
        </div>
        <span className="nl-table-pill">published_nl_latents_samples</span>
      </div>

      <div className="nl-filters">
        <SelectFilter
          label="Family"
          value={family}
          values={filters?.families ?? []}
          onChange={updateFilter('family')}
        />
        <SelectFilter
          label="Difficulty"
          value={difficulty}
          values={filters?.difficulties ?? []}
          onChange={updateFilter('difficulty')}
        />
        <SelectFilter
          label="Split"
          value={split}
          values={filters?.splits ?? []}
          onChange={updateFilter('split')}
        />
        <SelectFilter
          label="Encoder"
          value={encModel}
          values={filters?.enc_models ?? []}
          onChange={updateFilter('encModel')}
        />
        <SelectFilter
          label="Budget"
          value={budget}
          values={filters?.budgets ?? []}
          onChange={updateFilter('budget')}
        />
        <SelectFilter
          label="Data"
          value={dataVersion}
          values={filters?.data_versions ?? []}
          onChange={updateFilter('dataVersion')}
        />
        <SelectFilter
          label="Result"
          value={result}
          values={RESULT_OPTIONS}
          onChange={updateFilter('result')}
        />
        <label className="nl-toggle">
          <input
            type="checkbox"
            checked={hidePending}
            onChange={event => {
              updateListState({
                hidePending: event.target.checked,
                page: DEFAULT_PAGE,
              })
            }}
          />
          Hide pending
        </label>
        <label className="nl-toggle">
          <input
            type="checkbox"
            checked={hideSmoke}
            onChange={event => {
              updateListState({
                hideSmoke: event.target.checked,
                page: DEFAULT_PAGE,
              })
            }}
          />
          Hide smoke
        </label>
      </div>

      {error && (
        <div className="error-state nl-error">
          <span className="error-icon">!</span>
          <div>
            <p className="error-title">Failed to load nl-latents data</p>
            <p className="error-detail">{error}</p>
          </div>
        </div>
      )}

      <div className="nl-table-wrap">
        <div className="nl-table-meta">
          <span>
            {loading
              ? 'Loading...'
              : `${(samples?.total ?? 0).toLocaleString()} samples`}
          </span>
          {pageStatus && <span>{pageStatus}</span>}
        </div>
        <table className="nl-table">
          <thead>
            <tr>
              <th>Sample</th>
              <th>Family</th>
              <th>Diff</th>
              <th>Split</th>
              <th>Model</th>
              <th>Budget</th>
              <th>Prompt config</th>
              <th>Result</th>
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {!loading && samples?.samples.length === 0 && (
              <tr>
                <td colSpan={9} className="nl-empty">
                  No samples match.
                </td>
              </tr>
            )}
            {(samples?.samples ?? []).map(sample => (
              <tr key={sample.sample_id}>
                <td>
                  <Link
                    className="nl-sample-link"
                    href={`/nl-latents/samples/${sample.sample_id}${sampleSearchSuffix}`}
                  >
                    {sample.sample_id.slice(0, 12)}
                  </Link>
                </td>
                <td>{sample.family}</td>
                <td>{sample.difficulty}</td>
                <td>{sample.split}</td>
                <td className="nl-mono">{sample.enc_model_label}</td>
                <td className="nl-mono">{sample.budget}</td>
                <td className="nl-config-cell">
                  {sample.prompt_config_label || '-'}
                </td>
                <td>
                  <ResultBadge
                    state={sample.result_state}
                    failureCategory={sample.failure_category_normalized}
                  />
                </td>
                <td>{new Date(sample.created_at).toLocaleDateString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {samples && samples.total_pages > 1 && (
        <div className="nl-pagination">
          <button
            type="button"
            disabled={page <= 1}
            onClick={() => updateListState({ page: page - 1 })}
          >
            Previous
          </button>
          <span>
            Page {samples.page} of {samples.total_pages}
          </span>
          <button
            type="button"
            disabled={page >= samples.total_pages}
            onClick={() => updateListState({ page: page + 1 })}
          >
            Next
          </button>
        </div>
      )}
    </div>
  )
}
