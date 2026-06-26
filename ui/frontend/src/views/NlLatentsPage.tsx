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
const ERROR_STATE_CLASS =
  'mb-5 flex items-start gap-4 rounded-[10px] border border-[rgba(220,38,38,0.15)] bg-[var(--red-bg)] p-6'
const ERROR_ICON_CLASS =
  'flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[var(--red)] text-sm font-bold text-white'
const RESULT_CLASS_BY_STATE: Record<string, string> = {
  passed: 'bg-[var(--green-bg)] text-[var(--green)]',
  failed: 'bg-[var(--red-bg)] text-[var(--red)]',
  pending: 'bg-[var(--yellow-bg)] text-[var(--yellow)]',
}

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
    <label className="flex flex-col gap-1 text-xs font-semibold text-[var(--text-secondary)]">
      <span>{label}</span>
      <select
        value={value}
        onChange={event => onChange(event.target.value)}
        className="min-h-[34px] w-full rounded-md border border-[var(--border)] bg-[var(--bg-primary)] px-2 py-1.5 font-medium text-[var(--text-primary)] [font:inherit]"
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
  return (
    <span
      className={`inline-flex rounded-md px-[7px] py-[3px] text-xs leading-tight font-bold ${RESULT_CLASS_BY_STATE[state] ?? 'bg-[var(--bg-tertiary)] text-[var(--text-secondary)]'}`}
    >
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
    <div className="w-full max-w-none">
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
        <label className="flex min-h-14 items-center gap-2 text-xs font-semibold text-[var(--text-secondary)]">
          <input
            type="checkbox"
            checked={hidePending}
            className="h-4 w-4"
            onChange={event => {
              updateListState({
                hidePending: event.target.checked,
                page: DEFAULT_PAGE,
              })
            }}
          />
          Hide pending
        </label>
        <label className="flex min-h-14 items-center gap-2 text-xs font-semibold text-[var(--text-secondary)]">
          <input
            type="checkbox"
            checked={hideSmoke}
            className="h-4 w-4"
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
        <div className={ERROR_STATE_CLASS}>
          <span className={ERROR_ICON_CLASS}>!</span>
          <div>
            <p className="mb-1 font-semibold text-[var(--red)]">
              Failed to load nl-latents data
            </p>
            <p className="text-[13px] text-[var(--text-secondary)]">{error}</p>
          </div>
        </div>
      )}

      <div className="overflow-x-auto rounded-lg border border-[var(--border)]">
        <div className="flex items-center justify-between gap-3 border-b border-[var(--border-subtle)] px-3 py-2.5 text-[13px] text-[var(--text-secondary)]">
          <span>
            {loading
              ? 'Loading...'
              : `${(samples?.total ?? 0).toLocaleString()} samples`}
          </span>
          {pageStatus && <span>{pageStatus}</span>}
        </div>
        <table className="w-full min-w-[1120px] border-collapse">
          <thead>
            <tr>
              {[
                'Sample',
                'Family',
                'Diff',
                'Split',
                'Model',
                'Budget',
                'Prompt config',
                'Result',
                'Created',
              ].map(header => (
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
            {!loading && samples?.samples.length === 0 && (
              <tr>
                <td
                  colSpan={9}
                  className="border-b border-[var(--border-subtle)] px-3 py-2.5 text-center align-top text-[13px] text-[var(--text-muted)]"
                >
                  No samples match.
                </td>
              </tr>
            )}
            {(samples?.samples ?? []).map(sample => (
              <tr key={sample.sample_id} className="last:[&>td]:border-b-0">
                <td className="border-b border-[var(--border-subtle)] px-3 py-2.5 align-top text-[13px]">
                  <Link
                    className="font-mono"
                    href={`/nl-latents/samples/${sample.sample_id}${sampleSearchSuffix}`}
                  >
                    {sample.sample_id.slice(0, 12)}
                  </Link>
                </td>
                <td className="border-b border-[var(--border-subtle)] px-3 py-2.5 align-top text-[13px]">
                  {sample.family}
                </td>
                <td className="border-b border-[var(--border-subtle)] px-3 py-2.5 align-top text-[13px]">
                  {sample.difficulty}
                </td>
                <td className="border-b border-[var(--border-subtle)] px-3 py-2.5 align-top text-[13px]">
                  {sample.split}
                </td>
                <td className="border-b border-[var(--border-subtle)] px-3 py-2.5 align-top font-mono text-[13px]">
                  {sample.enc_model_label}
                </td>
                <td className="border-b border-[var(--border-subtle)] px-3 py-2.5 align-top font-mono text-[13px]">
                  {sample.budget}
                </td>
                <td className="max-w-[220px] border-b border-[var(--border-subtle)] px-3 py-2.5 align-top text-[13px]">
                  {sample.prompt_config_label || '-'}
                </td>
                <td className="border-b border-[var(--border-subtle)] px-3 py-2.5 align-top text-[13px]">
                  <ResultBadge
                    state={sample.result_state}
                    failureCategory={sample.failure_category_normalized}
                  />
                </td>
                <td className="border-b border-[var(--border-subtle)] px-3 py-2.5 align-top text-[13px]">
                  {new Date(sample.created_at).toLocaleDateString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {samples && samples.total_pages > 1 && (
        <div className="mt-4 flex items-center justify-end gap-3 max-md:justify-start">
          <button
            type="button"
            disabled={page <= 1}
            onClick={() => updateListState({ page: page - 1 })}
            className="cursor-pointer rounded-md border border-[var(--border)] bg-[var(--bg-primary)] px-2.5 py-1.5 text-[13px] text-[var(--text-primary)] disabled:cursor-not-allowed disabled:text-[var(--text-muted)]"
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
            className="cursor-pointer rounded-md border border-[var(--border)] bg-[var(--bg-primary)] px-2.5 py-1.5 text-[13px] text-[var(--text-primary)] disabled:cursor-not-allowed disabled:text-[var(--text-muted)]"
          >
            Next
          </button>
        </div>
      )}
    </div>
  )
}
