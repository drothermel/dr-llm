import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import './NlLatentsPage.css'

const ALL = '__all__'
const RESULT_OPTIONS = ['passed', 'failed', 'pending']

async function fetchJson(path) {
  const response = await fetch(path)
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }
  return response.json()
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
  const [filters, setFilters] = useState(null)
  const [samples, setSamples] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(1)
  const [family, setFamily] = useState(ALL)
  const [difficulty, setDifficulty] = useState(ALL)
  const [split, setSplit] = useState(ALL)
  const [encModel, setEncModel] = useState(ALL)
  const [budget, setBudget] = useState(ALL)
  const [dataVersion, setDataVersion] = useState(ALL)
  const [result, setResult] = useState(ALL)
  const [hidePending, setHidePending] = useState(true)
  const [hideSmoke, setHideSmoke] = useState(true)

  useEffect(() => {
    let active = true
    fetchJson('/api/nl-latents/filters')
      .then(data => {
        if (active) setFilters(data)
      })
      .catch(err => {
        if (active) setError(err.message)
      })
    return () => {
      active = false
    }
  }, [])

  const queryPath = useMemo(() => {
    const params = new URLSearchParams()
    params.set('page', String(page))
    params.set('limit', '20')
    if (family !== ALL) params.set('family', family)
    if (difficulty !== ALL) params.set('difficulty', difficulty)
    if (split !== ALL) params.set('split', split)
    if (encModel !== ALL) params.set('enc_model', encModel)
    if (budget !== ALL) params.set('budget', budget)
    if (dataVersion !== ALL) params.set('data_version', dataVersion)
    if (result !== ALL) params.set('result', result)
    if (hidePending) params.set('hide_pending', 'true')
    if (hideSmoke) params.set('hide_smoke', 'true')
    return `/api/nl-latents/samples?${params}`
  }, [
    budget,
    dataVersion,
    difficulty,
    encModel,
    family,
    hidePending,
    hideSmoke,
    page,
    result,
    split,
  ])

  useEffect(() => {
    let active = true
    fetchJson(queryPath)
      .then(data => {
        if (!active) return
        setSamples(data)
        setError(null)
        setLoading(false)
      })
      .catch(err => {
        if (!active) return
        setError(err.message)
        setLoading(false)
      })
    return () => {
      active = false
    }
  }, [queryPath])

  const resetPage = setter => value => {
    setter(value)
    setPage(1)
  }

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
          onChange={resetPage(setFamily)}
        />
        <SelectFilter
          label="Difficulty"
          value={difficulty}
          values={filters?.difficulties ?? []}
          onChange={resetPage(setDifficulty)}
        />
        <SelectFilter
          label="Split"
          value={split}
          values={filters?.splits ?? []}
          onChange={resetPage(setSplit)}
        />
        <SelectFilter
          label="Encoder"
          value={encModel}
          values={filters?.enc_models ?? []}
          onChange={resetPage(setEncModel)}
        />
        <SelectFilter
          label="Budget"
          value={budget}
          values={filters?.budgets ?? []}
          onChange={resetPage(setBudget)}
        />
        <SelectFilter
          label="Data"
          value={dataVersion}
          values={filters?.data_versions ?? []}
          onChange={resetPage(setDataVersion)}
        />
        <SelectFilter
          label="Result"
          value={result}
          values={RESULT_OPTIONS}
          onChange={resetPage(setResult)}
        />
        <label className="nl-toggle">
          <input
            type="checkbox"
            checked={hidePending}
            onChange={event => {
              setHidePending(event.target.checked)
              setPage(1)
            }}
          />
          Hide pending
        </label>
        <label className="nl-toggle">
          <input
            type="checkbox"
            checked={hideSmoke}
            onChange={event => {
              setHideSmoke(event.target.checked)
              setPage(1)
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
          {loading ? 'Loading...' : `${(samples?.total ?? 0).toLocaleString()} samples`}
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
                    to={`/nl-latents/samples/${sample.sample_id}`}
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
            onClick={() => setPage(current => current - 1)}
          >
            Previous
          </button>
          <span>
            Page {samples.page} of {samples.total_pages}
          </span>
          <button
            type="button"
            disabled={page >= samples.total_pages}
            onClick={() => setPage(current => current + 1)}
          >
            Next
          </button>
        </div>
      )}
    </div>
  )
}
