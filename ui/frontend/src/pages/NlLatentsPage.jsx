import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import './NlLatentsPage.css'

const ALL = '__all__'
const RESULT_OPTIONS = ['passed', 'failed', 'pending']
const DEFAULT_PAGE = 1
const PAGE_SIZE = 20
const FILTERS_PATH = '/api/nl-latents/filters'
const FALSE_PARAM = 'false'
const TRUE_PARAM = 'true'
const QUERY_CONFIG = [
  ['family', 'family'],
  ['difficulty', 'difficulty'],
  ['split', 'split'],
  ['encModel', 'enc_model'],
  ['budget', 'budget'],
  ['dataVersion', 'data_version'],
  ['result', 'result'],
]

const jsonCache = new Map()
const jsonInFlight = new Map()

async function fetchJson(path) {
  const response = await fetch(path)
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }
  return response.json()
}

async function fetchCachedJson(path) {
  if (jsonCache.has(path)) {
    return jsonCache.get(path)
  }
  if (jsonInFlight.has(path)) {
    return jsonInFlight.get(path)
  }

  const request = fetchJson(path)
    .then(data => {
      jsonCache.set(path, data)
      jsonInFlight.delete(path)
      return data
    })
    .catch(error => {
      jsonInFlight.delete(path)
      throw error
    })

  jsonInFlight.set(path, request)
  return request
}

function parsePage(searchParams) {
  const page = Number(searchParams.get('page'))
  return Number.isInteger(page) && page > 0 ? page : DEFAULT_PAGE
}

function parseListState(searchParams) {
  return {
    page: parsePage(searchParams),
    family: searchParams.get('family') || ALL,
    difficulty: searchParams.get('difficulty') || ALL,
    split: searchParams.get('split') || ALL,
    encModel: searchParams.get('enc_model') || ALL,
    budget: searchParams.get('budget') || ALL,
    dataVersion: searchParams.get('data_version') || ALL,
    result: searchParams.get('result') || ALL,
    hidePending: searchParams.get('hide_pending') !== FALSE_PARAM,
    hideSmoke: searchParams.get('hide_smoke') !== FALSE_PARAM,
  }
}

function listStateToSearchParams(state) {
  const params = new URLSearchParams()
  if (state.page !== DEFAULT_PAGE) params.set('page', String(state.page))

  for (const [stateKey, paramKey] of QUERY_CONFIG) {
    if (state[stateKey] !== ALL) {
      params.set(paramKey, state[stateKey])
    }
  }

  if (!state.hidePending) params.set('hide_pending', FALSE_PARAM)
  if (!state.hideSmoke) params.set('hide_smoke', FALSE_PARAM)
  return params
}

function buildSamplesPath(state) {
  const params = new URLSearchParams()
  params.set('page', String(state.page))
  params.set('limit', String(PAGE_SIZE))

  for (const [stateKey, paramKey] of QUERY_CONFIG) {
    if (state[stateKey] !== ALL) {
      params.set(paramKey, state[stateKey])
    }
  }

  if (state.hidePending) params.set('hide_pending', TRUE_PARAM)
  if (state.hideSmoke) params.set('hide_smoke', TRUE_PARAM)
  return `/api/nl-latents/samples?${params}`
}

function prefetchSamplesPage(state, page) {
  if (page < DEFAULT_PAGE) return
  fetchCachedJson(buildSamplesPath({ ...state, page })).catch(() => {})
}

function SelectFilter({ label, value, values, onChange }) {
  return (
    <label className="nl-filter">
      <span>{label}</span>
      <select value={value} onChange={event => onChange(event.target.value)}>
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

function ResultBadge({ state, failureCategory }) {
  const className = `nl-result nl-result-${state}`
  return (
    <span className={className}>
      {state}
      {failureCategory ? ` · ${failureCategory}` : ''}
    </span>
  )
}

export default function NlLatentsPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [filters, setFilters] = useState(null)
  const [filtersError, setFiltersError] = useState(null)
  const [samplesResult, setSamplesResult] = useState({
    path: null,
    data: null,
    error: null,
  })

  const listState = useMemo(() => parseListState(searchParams), [searchParams])
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
    let active = true
    fetchCachedJson(FILTERS_PATH)
      .then(data => {
        if (active) setFilters(data)
      })
      .catch(err => {
        if (active) setFiltersError(err.message)
      })
    return () => {
      active = false
    }
  }, [])

  const queryPath = useMemo(() => {
    return buildSamplesPath(listState)
  }, [listState])

  useEffect(() => {
    let active = true
    fetchCachedJson(queryPath)
      .then(data => {
        if (!active) return
        setSamplesResult({ path: queryPath, data, error: null })
      })
      .catch(err => {
        if (!active) return
        setSamplesResult({ path: queryPath, data: null, error: err.message })
      })
    return () => {
      active = false
    }
  }, [queryPath])

  useEffect(() => {
    const samples = samplesResult.path === queryPath ? samplesResult.data : null
    if (!samples) return

    prefetchSamplesPage(listState, page - 1)
    if (page < samples.total_pages) {
      prefetchSamplesPage(listState, page + 1)
    }
  }, [listState, page, queryPath, samplesResult])

  const updateListState = updates => {
    setSearchParams(
      listStateToSearchParams({ ...listState, ...updates }),
    )
  }

  const updateFilter = key => value => {
    updateListState({ [key]: value, page: DEFAULT_PAGE })
  }

  const sampleSearch = searchParams.toString()
  const sampleSearchSuffix = sampleSearch ? `?${sampleSearch}` : ''
  const samples = samplesResult.path === queryPath ? samplesResult.data : null
  const samplesError = samplesResult.path === queryPath ? samplesResult.error : null
  const loading = samplesResult.path !== queryPath
  const error = samplesError ?? filtersError
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
            {loading ? 'Loading...' : `${(samples?.total ?? 0).toLocaleString()} samples`}
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
            {!loading && samples?.samples?.length === 0 && (
              <tr>
                <td colSpan="9" className="nl-empty">No samples match.</td>
              </tr>
            )}
            {(samples?.samples ?? []).map(sample => (
              <tr key={sample.sample_id}>
                <td>
                  <Link
                    className="nl-sample-link"
                    to={`/nl-latents/samples/${sample.sample_id}${sampleSearchSuffix}`}
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
