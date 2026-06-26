import { useEffect, useMemo, useState } from 'react'
import { Link, useLocation, useParams } from 'react-router-dom'
import hljs from 'highlight.js/lib/core'
import python from 'highlight.js/lib/languages/python'
import '../styles/code-theme.css'
import './NlLatentsPage.css'

hljs.registerLanguage('python', python)

async function fetchJson(path) {
  const response = await fetch(path)
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }
  return response.json()
}

function formatSeconds(value) {
  if (value === null || value === undefined) return null
  return `${Number(value).toFixed(2)}s`
}

function formatPercent(value) {
  if (value === null || value === undefined) return '—'
  return `${(Number(value) * 100).toFixed(1)}%`
}

function formatDateTime(value) {
  if (!value) return null
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function shortDate(value) {
  if (!value) return null
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

function truncate(value, max) {
  const str = String(value)
  return str.length > max ? `${str.slice(0, max)}…` : str
}

const EFFORT_HIDDEN = new Set(['', 'default', 'none', 'auto', 'na', 'n/a'])
function effortLabel(value) {
  if (!value || EFFORT_HIDDEN.has(String(value).toLowerCase())) return null
  return String(value)
}

function useCopy() {
  const [copied, setCopied] = useState(null)
  const copy = value => {
    if (!value || !navigator.clipboard) return
    navigator.clipboard.writeText(String(value)).then(() => {
      setCopied(value)
      window.setTimeout(() => setCopied(null), 1400)
    })
  }
  return [copied, copy]
}

/* Compact, copyable chip for values that don't need to stay open in the
   foot: long versions, timestamps, and non-interpretable hash ids. */
function MetaChip({ label, value, display, copied, onCopy }) {
  if (value === null || value === undefined || value === '') return null
  const isCopied = copied === value
  return (
    <button
      type="button"
      className="nl-id-chip"
      title={`${label}: ${value} (click to copy)`}
      onClick={() => onCopy(value)}
    >
      <span className="nl-id-chip-key">{label}</span>
      <span className="nl-id-chip-val">
        {isCopied ? 'copied' : (display ?? truncate(value, 10))}
      </span>
    </button>
  )
}

function ResultBadge({ state, failure }) {
  return (
    <span className={`nl-badge nl-badge-${state}`}>
      <span className="nl-badge-dot" aria-hidden="true" />
      <span className="nl-badge-state">{state}</span>
      {failure && state === 'failed' && (
        <span className="nl-badge-reason">{failure}</span>
      )}
    </span>
  )
}

function StatCell({ label, value, sub, mono }) {
  return (
    <div className="nl-stat">
      <span className="nl-stat-label">{label}</span>
      <span className={`nl-stat-value${mono ? ' nl-mono' : ''}`}>
        {value ?? '—'}
      </span>
      {sub && <span className="nl-stat-sub">{sub}</span>}
    </div>
  )
}

function BudgetMeter({ budget, actual, budgetOk }) {
  const limit = Number(budget)
  const used = actual === null || actual === undefined ? null : Number(actual)
  const hasMeter = Number.isFinite(limit) && limit > 0 && used !== null
  const ratio = hasMeter ? used / limit : 0
  const over = budgetOk === false || (hasMeter && used > limit)
  const stateClass = over ? 'over' : 'ok'

  return (
    <div className="nl-stat nl-budget">
      <span className="nl-stat-label">Budget</span>
      <span className="nl-budget-figures">
        <strong className="nl-budget-used">{used ?? '—'}</strong>
        <span className="nl-budget-limit">
          / {Number.isFinite(limit) ? limit : budget ?? '—'} chars
        </span>
      </span>
      {hasMeter && (
        <span className="nl-budget-track" aria-hidden="true">
          <span
            className={`nl-budget-fill nl-budget-fill-${stateClass}`}
            style={{ width: `${Math.min(100, Math.max(2, ratio * 100))}%` }}
          />
        </span>
      )}
      <span className={`nl-budget-status nl-budget-status-${stateClass}`}>
        {over
          ? 'over budget'
          : hasMeter
            ? `${Math.round(ratio * 100)}% of budget`
            : 'within budget'}
      </span>
    </div>
  )
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  return `${(bytes / 1024).toFixed(1)} KB`
}

function codeStats(value) {
  return {
    lines: value.split('\n').length,
    chars: [...value].length,
    bytes: new TextEncoder().encode(value).length,
  }
}

function CodePane({ label, value, badge, area, language = 'python' }) {
  const html = useMemo(() => {
    if (!value) return ''
    try {
      return hljs.highlight(value, { language }).value
    } catch {
      return null
    }
  }, [value, language])
  const stats = useMemo(() => (value ? codeStats(value) : null), [value])
  if (!value) return null
  return (
    <section className="nl-pane nl-pane-code" style={{ gridArea: area }}>
      <header className="nl-pane-head">
        <h3>{label}</h3>
        <div className="nl-pane-meta">
          <span className="nl-metric">
            {stats.lines.toLocaleString()} lines
          </span>
          <span className="nl-metric">
            {stats.chars.toLocaleString()} chars
          </span>
          <span className="nl-metric">{formatBytes(stats.bytes)}</span>
          {badge && (
            <span className="nl-metric nl-metric-lang">{badge}</span>
          )}
        </div>
      </header>
      <pre className="nl-code">
        {html === null ? (
          <code>{value}</code>
        ) : (
          <code className="hljs" dangerouslySetInnerHTML={{ __html: html }} />
        )}
      </pre>
    </section>
  )
}

function TextPane({ label, value, badge, variant, area }) {
  if (!value) return null
  return (
    <section
      className={`nl-pane nl-pane-${variant ?? 'prose'}`}
      style={{ gridArea: area }}
    >
      <header className="nl-pane-head">
        <h3>{label}</h3>
        {badge && <span className="nl-pane-badge">{badge}</span>}
      </header>
      <pre className="nl-prose">{value}</pre>
    </section>
  )
}

export default function NlLatentsDetailPage() {
  const { sampleId } = useParams()
  const location = useLocation()
  const [resolved, setResolved] = useState({
    id: null,
    sample: null,
    error: null,
  })
  const [copied, copy] = useCopy()

  useEffect(() => {
    let active = true
    fetchJson(`/api/nl-latents/samples/${sampleId}`)
      .then(data => {
        if (active) setResolved({ id: sampleId, sample: data, error: null })
      })
      .catch(err => {
        if (active)
          setResolved({ id: sampleId, sample: null, error: err.message })
      })
    return () => {
      active = false
    }
  }, [sampleId])

  const loading = resolved.id !== sampleId
  const sample = loading ? null : resolved.sample
  const error = loading ? null : resolved.error

  return (
    <div className="page nl-page nl-detail">
      <div className="nl-back-row">
        <Link to={`/nl-latents${location.search}`}>← Back to samples</Link>
      </div>

      {loading && (
        <div className="loading-state">
          <span className="spinner" /> Loading sample…
        </div>
      )}

      {error && (
        <div className="error-state nl-error">
          <span className="error-icon">!</span>
          <div>
            <p className="error-title">Failed to load sample</p>
            <p className="error-detail">{error}</p>
          </div>
        </div>
      )}

      {sample && !loading && (
        <>
          <header className="nl-head">
            <div className="nl-head-main">
              <h1 className="nl-title">{sample.task_id}</h1>
              <div className="nl-subline">
                <span className="nl-tag-family">{sample.family}</span>
                <span className="nl-dot" />
                <span>difficulty {sample.difficulty}</span>
                <span className="nl-dot" />
                <span>{sample.split}</span>
                {sample.language && (
                  <>
                    <span className="nl-dot" />
                    <span>{sample.language}</span>
                  </>
                )}
              </div>
            </div>
            <ResultBadge
              state={sample.result_state}
              failure={
                sample.failure_category_normalized ?? sample.failure_category
              }
            />
          </header>

          <section className="nl-meta">
            <div className="nl-meta-grid">
              <BudgetMeter
                budget={sample.budget}
                actual={sample.actual_chars}
                budgetOk={sample.budget_ok}
              />
              <StatCell
                label="Encoder"
                value={sample.enc_model_label}
                sub={[
                  effortLabel(sample.enc_reasoning_effort),
                  formatSeconds(sample.enc_time_s),
                ]
                  .filter(Boolean)
                  .join(' · ')}
                mono
              />
              <StatCell
                label="Decoder"
                value={sample.dec_model_label}
                sub={[
                  effortLabel(sample.dec_reasoning_effort),
                  formatSeconds(sample.dec_time_s),
                ]
                  .filter(Boolean)
                  .join(' · ')}
                mono
              />
              <StatCell
                label="Validation"
                value={formatPercent(sample.validation_pass_rate)}
                sub={
                  sample.validation_compiles === null ||
                  sample.validation_compiles === undefined
                    ? 'pass rate · compile unknown'
                    : `pass rate · ${sample.validation_compiles ? 'compiles' : 'no compile'}`
                }
              />
            </div>

            <div className="nl-meta-config">
              <span className="nl-stat-label">Prompt config</span>
              {sample.prompt_block_names &&
              sample.prompt_block_names.length > 0 ? (
                <div className="nl-chips">
                  {sample.prompt_block_names.map((name, i) => (
                    <span className="nl-chip" key={`${name}-${i}`}>
                      {name}
                    </span>
                  ))}
                </div>
              ) : (
                <span className="nl-stat-value">
                  {sample.prompt_config_label ?? '—'}
                </span>
              )}
            </div>

            <div className="nl-meta-foot">
              <span className="nl-foot-item nl-foot-run">
                <span className="nl-foot-key">run</span>
                <span className="nl-foot-run-val">{sample.run_id ?? '—'}</span>
              </span>
              {sample.finish_reason && (
                <span className="nl-foot-item">
                  <span className="nl-foot-key">finish</span>
                  {sample.finish_reason}
                </span>
              )}
              <span className="nl-foot-ids">
                <MetaChip
                  label="data"
                  value={sample.task_data_version}
                  display={truncate(sample.task_data_version ?? '', 18)}
                  copied={copied}
                  onCopy={copy}
                />
                <MetaChip
                  label="created"
                  value={formatDateTime(sample.created_at)}
                  display={shortDate(sample.created_at)}
                  copied={copied}
                  onCopy={copy}
                />
                <MetaChip
                  label="sample"
                  value={sample.sample_id}
                  copied={copied}
                  onCopy={copy}
                />
                <MetaChip
                  label="call"
                  value={sample.call_id}
                  copied={copied}
                  onCopy={copy}
                />
              </span>
            </div>
          </section>

          {/* Grid areas keep the two code blocks on a shared row track so they
              align vertically regardless of prompt length, and keep the most
              relevant outputs (encoded latent, error) pinned to the top. */}
          <div className="nl-panes">
            <TextPane
              label="Encoded output"
              value={sample.description}
              variant="latent"
              area="outv"
              badge={
                sample.actual_chars !== null &&
                sample.actual_chars !== undefined
                  ? `${sample.actual_chars} chars`
                  : 'NL latent'
              }
            />
            <TextPane
              label="Error detail"
              value={sample.error_detail}
              variant="error"
              area="errv"
            />

            <TextPane
              label="Encoder prompt"
              value={sample.enc_prompt_instructions ?? sample.enc_prompt}
              variant="muted"
              area="encp"
            />
            <div className="nl-pane-group" style={{ gridArea: 'decp' }}>
              <TextPane
                label="Decoder system prompt"
                value={sample.dec_system}
                variant="muted"
              />
              <TextPane label="Decoder task" value={sample.dec_task} />
            </div>

            <CodePane
              label="Input code"
              value={sample.input_code}
              badge={sample.language}
              area="srcc"
            />
            <CodePane
              label="Decoded code"
              value={sample.decoded_code}
              badge={sample.language}
              area="decc"
            />
          </div>
        </>
      )}
    </div>
  )
}
