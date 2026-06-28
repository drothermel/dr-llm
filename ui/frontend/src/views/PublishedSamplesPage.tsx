'use client'

import type { QueryClient } from '@tanstack/react-query'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { useEffect, useMemo, useTransition } from 'react'
import {
  ALL,
  DEFAULT_LIST_STATE,
  DEFAULT_PAGE,
  buildSamplesPath,
  listStateToSearchParams,
  parseListState,
  type PublishedListState,
} from '@/lib/published'
import {
  filtersQueryOptions,
  samplesQueryOptions,
} from '@/lib/publishedQueries'
import { ResultBadge, SECTION_LABEL, Tag } from '@/components/primitives'

const COLUMNS = [
  'Sample',
  'Project',
  'Pool',
  'Role',
  'Task',
  'Model',
  'Result',
  'Created',
] as const

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

function prefetchSamplesPage(
  queryClient: QueryClient,
  state: PublishedListState,
  page: number,
): void {
  if (page < DEFAULT_PAGE) return
  void queryClient.prefetchQuery(
    samplesQueryOptions(buildSamplesPath({ ...state, page })),
  )
}

function shortDate(value: string | null): string {
  if (!value) return 'unknown'
  return new Date(value).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function queryErrorMessage(error: Error | null): string | null {
  return error ? error.message : null
}

function roleLabel(value: string): string {
  return value.replaceAll('_', ' ')
}

export default function PublishedSamplesPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const queryClient = useQueryClient()
  const [isPending, startTransition] = useTransition()

  const listState = useMemo(
    () => (searchParams ? parseListState(searchParams) : DEFAULT_LIST_STATE),
    [searchParams],
  )
  const queryPath = useMemo(() => buildSamplesPath(listState), [listState])
  const filtersQuery = useQuery(filtersQueryOptions())
  const samplesQuery = useQuery(samplesQueryOptions(queryPath))
  const filters = filtersQuery.data ?? null
  const samples = samplesQuery.data ?? null
  const page = listState.page

  useEffect(() => {
    if (!samples) return
    prefetchSamplesPage(queryClient, listState, page - 1)
    if (page < samples.total_pages) {
      prefetchSamplesPage(queryClient, listState, page + 1)
    }
  }, [listState, page, queryClient, samples])

  const updateListState = (updates: Partial<PublishedListState>) => {
    const params = listStateToSearchParams({ ...listState, ...updates })
    const suffix = params.toString()
    startTransition(() => {
      router.push(suffix ? `/samples?${suffix}` : '/samples')
    })
  }

  const updateFilter =
    (key: keyof PublishedListState) => (value: string) => {
      updateListState({ [key]: value, page: DEFAULT_PAGE })
    }

  const sampleSearch = searchParams?.toString() ?? ''
  const sampleSearchSuffix = sampleSearch ? `?${sampleSearch}` : ''
  const error =
    queryErrorMessage(samplesQuery.error) ?? queryErrorMessage(filtersQuery.error)
  const loading = isPending || samplesQuery.isLoading
  const pageStatus = samples
    ? `Page ${samples.page} of ${samples.total_pages}`
    : null

  return (
    <div className="w-full">
      <div className="mb-7 flex items-start justify-between gap-6 max-md:block">
        <div>
          <h1 className="font-display text-[26px] leading-tight font-bold text-[var(--text-primary)]">
            Published samples
          </h1>
          <p className="mt-1 text-sm text-[var(--text-secondary)]">
            Unified summaries from{' '}
            <span className="font-mono text-[var(--text-primary)]">
              published_pool_samples
            </span>
          </p>
        </div>
      </div>

      <div className="mb-6 grid grid-cols-[repeat(auto-fit,minmax(150px,1fr))] items-end gap-3">
        <SelectFilter
          label="Project"
          value={listState.project}
          values={filters?.projects ?? []}
          onChange={updateFilter('project')}
        />
        <SelectFilter
          label="Pool"
          value={listState.sourcePool}
          values={filters?.source_pools ?? []}
          onChange={updateFilter('sourcePool')}
        />
        <SelectFilter
          label="Role"
          value={listState.sampleRole}
          values={filters?.sample_roles ?? []}
          onChange={updateFilter('sampleRole')}
        />
        <SelectFilter
          label="Family"
          value={listState.taskFamily}
          values={filters?.task_families ?? []}
          onChange={updateFilter('taskFamily')}
        />
        <SelectFilter
          label="Model"
          value={listState.model}
          values={filters?.models ?? []}
          onChange={updateFilter('model')}
        />
        <SelectFilter
          label="Result"
          value={listState.result}
          values={filters?.result_states ?? []}
          onChange={updateFilter('result')}
        />
        <SelectFilter
          label="Dataset"
          value={listState.dataset}
          values={filters?.datasets ?? []}
          onChange={updateFilter('dataset')}
        />
        <CheckboxFilter
          label="Hide pending"
          checked={listState.hidePending}
          onChange={checked =>
            updateListState({ hidePending: checked, page: DEFAULT_PAGE })
          }
        />
        <CheckboxFilter
          label="Hide smoke"
          checked={listState.hideSmoke}
          onChange={checked =>
            updateListState({ hideSmoke: checked, page: DEFAULT_PAGE })
          }
        />
      </div>

      {error && (
        <div className="mb-5 flex items-start gap-4 rounded-xl border border-[var(--red-border)] bg-[var(--red-bg)] p-6">
          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[var(--red)] text-sm font-bold text-white">
            !
          </span>
          <div>
            <p className="mb-1 font-semibold text-[var(--red)]">
              Failed to load published samples
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
                Loading
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
              {COLUMNS.map(header => (
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
                  colSpan={COLUMNS.length}
                  className="px-4 py-12 text-center align-middle"
                >
                  <p className="text-sm font-medium text-[var(--text-secondary)]">
                    No samples match these filters
                  </p>
                </td>
              </tr>
            )}
            {(samples?.samples ?? []).map(sample => (
              <tr
                key={`${sample.source_project}:${sample.source_pool}:${sample.source_sample_id}`}
                className="transition-colors last:[&>td]:border-b-0 hover:bg-[var(--bg-hover)]"
              >
                <td className="border-b border-[var(--border-subtle)] px-4 py-2.5 align-middle text-[13px]">
                  <Link
                    className="font-mono text-[var(--accent)] hover:text-[var(--accent-hover)]"
                    href={`/samples/${sample.source_project}/${sample.source_pool}/${sample.source_sample_id}${sampleSearchSuffix}`}
                  >
                    {sample.source_sample_id.slice(0, 12)}
                  </Link>
                  <div className="mt-1 font-mono text-[11px] text-[var(--text-muted)]">
                    #{sample.sample_idx}
                  </div>
                </td>
                <td className="border-b border-[var(--border-subtle)] px-4 py-2.5 align-middle text-[13px]">
                  <Tag mono>{sample.source_project}</Tag>
                </td>
                <td className="max-w-[220px] border-b border-[var(--border-subtle)] px-4 py-2.5 align-middle text-[13px]">
                  <span className="font-mono text-[var(--text-primary)]">
                    {sample.source_pool}
                  </span>
                </td>
                <td className="border-b border-[var(--border-subtle)] px-4 py-2.5 align-middle text-[13px] text-[var(--text-secondary)]">
                  {roleLabel(sample.sample_role)}
                </td>
                <td className="border-b border-[var(--border-subtle)] px-4 py-2.5 align-middle text-[13px]">
                  <div className="font-medium text-[var(--text-primary)]">
                    {sample.task_family ?? 'unknown'}
                  </div>
                  <div className="mt-1 font-mono text-[11px] text-[var(--text-muted)]">
                    {sample.dataset_id ?? sample.task_id ?? 'no dataset'}
                  </div>
                </td>
                <td className="max-w-[220px] border-b border-[var(--border-subtle)] px-4 py-2.5 align-middle text-[13px]">
                  <span className="font-mono text-[var(--text-secondary)]">
                    {sample.model ?? sample.provider ?? 'unknown'}
                  </span>
                </td>
                <td className="border-b border-[var(--border-subtle)] px-4 py-2.5 align-middle">
                  <ResultBadge
                    state={sample.result_state}
                    failure={sample.failure_category}
                    size="sm"
                  />
                </td>
                <td className="border-b border-[var(--border-subtle)] px-4 py-2.5 align-middle font-mono text-[12px] text-[var(--text-muted)]">
                  {shortDate(sample.created_at)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {samples && samples.total_pages > 1 && (
        <div className="mt-4 flex items-center justify-end gap-2">
          <button
            type="button"
            disabled={page <= DEFAULT_PAGE}
            className="rounded-md border border-[var(--border)] px-3 py-1.5 text-[13px] font-medium text-[var(--text-secondary)] disabled:cursor-not-allowed disabled:opacity-45"
            onClick={() => updateListState({ page: page - 1 })}
          >
            Previous
          </button>
          <button
            type="button"
            disabled={page >= samples.total_pages}
            className="rounded-md border border-[var(--border)] px-3 py-1.5 text-[13px] font-medium text-[var(--text-secondary)] disabled:cursor-not-allowed disabled:opacity-45"
            onClick={() => updateListState({ page: page + 1 })}
          >
            Next
          </button>
        </div>
      )}
    </div>
  )
}
