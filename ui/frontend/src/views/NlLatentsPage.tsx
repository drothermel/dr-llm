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
import { ResultBadge, SECTION_LABEL } from '@/components/primitives'
import type {
  NlLatentsFilters,
  NlLatentsSamplesResponse,
} from '@/lib/types'

const RESULT_OPTIONS = ['passed', 'failed', 'pending']
const FILTERS_PATH = '/api/nl-latents/filters'
const ERROR_STATE_CLASS =
  'mb-5 flex items-start gap-4 rounded-xl border border-[var(--red-border)] bg-[var(--red-bg)] p-6'
const ERROR_ICON_CLASS =
  'flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[var(--red)] text-sm font-bold text-white'

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
    <label className="flex flex-col gap-1.5">
      <span className={SECTION_LABEL}>{label}</span>
      <select
        value={value}
        onChange={event => onChange(event.target.value)}
        className="min-h-[34px] w-full cursor-pointer rounded-md border border-[var(--border)] bg-[var(--bg-primary)] px-2.5 py-1.5 text-[13px] font-medium text-[var(--text-primary)] transition-colors hover:border-[var(--border-strong)] focus:border-[var(--accent)]"
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

type CheckboxFilterProps = {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
}

function CheckboxFilter({ label, checked, onChange }: CheckboxFilterProps) {
  return (
    <label className="flex cursor-pointer items-center gap-2 self-end py-1.5 text-[13px] font-medium text-[var(--text-secondary)]">
      <input
        type="checkbox"
        checked={checked}
        className="h-4 w-4 accent-[var(--accent)]"
        onChange={event => onChange(event.target.checked)}
      />
      {label}
    </label>
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

  const columns = [
    'Sample',
    'Family',
    'Diff',
    'Split',
    'Model',
    'Budget',
    'Prompt config',
    'Result',
    'Created',
  ]

  return (
    <div className="w-full">
      <div className="mb-7 flex items-start justify-between gap-6 max-md:block">
        <div>
          <h1 className="font-display text-[26px] leading-tight font-bold tracking-[-0.02em] text-[var(--text-primary)]">
            nl-latents
          </h1>
          <p className="mt-1 text-sm text-[var(--text-secondary)]">
            Published sample summaries from{' '}
            <span className="font-mono text-[var(--text-primary)]">
              published_nl_latents_samples
            </span>
          </p>
        </div>
      </div>

      <div className="mb-6 grid grid-cols-[repeat(auto-fit,minmax(150px,1fr))] items-end gap-3">
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
        <CheckboxFilter
          label="Hide pending"
          checked={hidePending}
          onChange={checked =>
            updateListState({ hidePending: checked, page: DEFAULT_PAGE })
          }
        />
        <CheckboxFilter
          label="Hide smoke"
          checked={hideSmoke}
          onChange={checked =>
            updateListState({ hideSmoke: checked, page: DEFAULT_PAGE })
          }
        />
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

      <div className="overflow-x-auto rounded-xl border border-[var(--border)]">
        <div className="flex items-center justify-between gap-3 border-b border-[var(--border-subtle)] bg-[var(--bg-secondary)] px-4 py-2.5 text-[13px]">
          <span className="font-medium text-[var(--text-secondary)]">
            {loading ? (
              <span className="inline-flex items-center gap-2">
                <span className="h-3 w-3 animate-spin rounded-full border-2 border-[var(--border)] border-t-[var(--accent)]" />
                Loading…
              </span>
            ) : (
              <>
                <span className="font-mono text-[var(--text-primary)]">
                  {(samples?.total ?? 0).toLocaleString()}
                </span>{' '}
                samples
              </>
            )}
          </span>
          {pageStatus && (
            <span className="font-mono text-[12px] text-[var(--text-muted)]">
              {pageStatus}
            </span>
          )}
        </div>
        <table className="w-full min-w-[1120px] border-collapse">
          <thead>
            <tr>
              {columns.map(header => (
                <th
                  key={header}
                  className="sticky top-0 z-[1] border-b border-[var(--border)] bg-[var(--bg-secondary)] px-4 py-2.5 text-left align-middle font-display text-[11px] font-semibold tracking-[0.06em] text-[var(--text-muted)] uppercase"
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
                  colSpan={columns.length}
                  className="px-4 py-12 text-center align-middle"
                >
                  <p className="text-sm font-medium text-[var(--text-secondary)]">
                    No samples match these filters
                  </p>
                  <p className="mt-1 text-[13px] text-[var(--text-muted)]">
                    Try widening Result or clearing the Hide pending / Hide smoke
                    toggles.
                  </p>
                </td>
              </tr>
            )}
            {(samples?.samples ?? []).map(sample => (
              <tr
                key={sample.sample_id}
                className="transition-colors last:[&>td]:border-b-0 hover:bg-[var(--bg-hover)]"
              >
                <td className="border-b border-[var(--border-subtle)] px-4 py-2.5 align-middle text-[13px]">
                  <Link
                    className="font-mono text-[var(--accent)] hover:text-[var(--accent-hover)]"
                    href={`/nl-latents/samples/${sample.sample_id}${sampleSearchSuffix}`}
                  >
                    {sample.sample_id.slice(0, 12)}
                  </Link>
                </td>
                <td className="border-b border-[var(--border-subtle)] px-4 py-2.5 align-middle text-[13px] text-[var(--text-secondary)]">
                  {sample.family}
                </td>
                <td className="border-b border-[var(--border-subtle)] px-4 py-2.5 align-middle font-mono text-[13px] text-[var(--text-secondary)]">
                  {sample.difficulty}
                </td>
                <td className="border-b border-[var(--border-subtle)] px-4 py-2.5 align-middle text-[13px] text-[var(--text-secondary)]">
                  {sample.split}
                </td>
                <td className="border-b border-[var(--border-subtle)] px-4 py-2.5 align-middle font-mono text-[13px] text-[var(--text-primary)]">
                  {sample.enc_model_label}
                </td>
                <td className="border-b border-[var(--border-subtle)] px-4 py-2.5 align-middle font-mono text-[13px] text-[var(--text-secondary)]">
                  {sample.budget}
                </td>
                <td className="max-w-[240px] truncate border-b border-[var(--border-subtle)] px-4 py-2.5 align-middle text-[13px] text-[var(--text-secondary)]">
                  {sample.prompt_config_label || '—'}
                </td>
                <td className="border-b border-[var(--border-subtle)] px-4 py-2.5 align-middle text-[13px]">
                  <ResultBadge
                    state={sample.result_state}
                    failure={sample.failure_category_normalized}
                    size="sm"
                  />
                </td>
                <td className="border-b border-[var(--border-subtle)] px-4 py-2.5 align-middle font-mono text-[12px] whitespace-nowrap text-[var(--text-muted)]">
                  {new Date(sample.created_at).toLocaleDateString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {samples && samples.total_pages > 1 && (
        <div className="mt-4 flex items-center justify-end gap-3 text-[13px] max-md:justify-between">
          <button
            type="button"
            disabled={page <= 1}
            onClick={() => updateListState({ page: page - 1 })}
            className="cursor-pointer rounded-md border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-1.5 font-medium text-[var(--text-primary)] transition-colors hover:border-[var(--border-strong)] hover:bg-[var(--bg-hover)] disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:border-[var(--border)] disabled:hover:bg-[var(--bg-primary)]"
          >
            Previous
          </button>
          <span className="font-mono text-[12px] text-[var(--text-muted)]">
            {samples.page} / {samples.total_pages}
          </span>
          <button
            type="button"
            disabled={page >= samples.total_pages}
            onClick={() => updateListState({ page: page + 1 })}
            className="cursor-pointer rounded-md border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-1.5 font-medium text-[var(--text-primary)] transition-colors hover:border-[var(--border-strong)] hover:bg-[var(--bg-hover)] disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:border-[var(--border)] disabled:hover:bg-[var(--bg-primary)]"
          >
            Next
          </button>
        </div>
      )}
    </div>
  )
}
