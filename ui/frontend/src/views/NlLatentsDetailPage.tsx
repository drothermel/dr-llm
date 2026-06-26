'use client'

import Link from 'next/link'
import { useMemo, useState } from 'react'
import hljs from 'highlight.js/lib/core'
import python from 'highlight.js/lib/languages/python'
import type { NlLatentsSampleDetail } from '@/lib/types'

hljs.registerLanguage('python', python)

function formatSeconds(value: number | null | undefined): string | null {
  if (value === null || value === undefined) return null
  return `${Number(value).toFixed(2)}s`
}

function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—'
  return `${(Number(value) * 100).toFixed(1)}%`
}

function shortDate(value: string | null | undefined): string | null {
  if (!value) return null
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

function truncate(value: string, max: number): string {
  return value.length > max ? `${value.slice(0, max)}...` : value
}

const EFFORT_HIDDEN = new Set(['', 'default', 'none', 'auto', 'na', 'n/a'])

function effortLabel(value: string | null | undefined): string | null {
  if (!value || EFFORT_HIDDEN.has(String(value).toLowerCase())) return null
  return String(value)
}

function useCopy(): [string | null, (value: string | null | undefined) => void] {
  const [copied, setCopied] = useState<string | null>(null)
  const copy = (value: string | null | undefined) => {
    if (!value || !navigator.clipboard) return
    navigator.clipboard.writeText(String(value)).then(() => {
      setCopied(value)
      window.setTimeout(() => setCopied(null), 1400)
    })
  }
  return [copied, copy]
}

type MetaChipProps = {
  label: string
  value: string | number | null | undefined
  display?: string | null
  copied: string | null
  onCopy: (value: string) => void
}

function MetaChip({
  label,
  value,
  display,
  copied,
  onCopy,
}: MetaChipProps) {
  if (value === null || value === undefined || value === '') return null
  const stringValue = String(value)
  const isCopied = copied === stringValue
  return (
    <button
      type="button"
      className="nl-id-chip"
      title={`${label}: ${stringValue} (click to copy)`}
      onClick={() => onCopy(stringValue)}
    >
      <span className="nl-id-chip-key">{label}</span>
      <span className="nl-id-chip-val">
        {isCopied ? 'copied' : (display ?? truncate(stringValue, 10))}
      </span>
    </button>
  )
}

type DetailResultBadgeProps = {
  state: string
  failure: string | null
}

function ResultBadge({ state, failure }: DetailResultBadgeProps) {
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

type StatCellProps = {
  label: string
  value: string | null
  sub?: string | null
  mono?: boolean
}

function StatCell({ label, value, sub, mono }: StatCellProps) {
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

type BudgetMeterProps = {
  budget: string | null
  actual: number | null
  budgetOk: boolean | null
}

function BudgetMeter({ budget, actual, budgetOk }: BudgetMeterProps) {
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
          / {Number.isFinite(limit) ? limit : (budget ?? '—')} chars
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

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  return `${(bytes / 1024).toFixed(1)} KB`
}

function codeStats(value: string): { lines: number; chars: number; bytes: number } {
  return {
    lines: value.split('\n').length,
    chars: [...value].length,
    bytes: new TextEncoder().encode(value).length,
  }
}

type PaneProps = {
  label: string
  value: string | null | undefined
  badge?: string | null
  area?: string
}

type CodePaneProps = PaneProps & {
  language?: string
}

function CodePane({
  label,
  value,
  badge,
  area,
  language = 'python',
}: CodePaneProps) {
  const html = useMemo(() => {
    if (!value) return ''
    try {
      return hljs.highlight(value, { language }).value
    } catch {
      return null
    }
  }, [value, language])
  const stats = useMemo(() => (value ? codeStats(value) : null), [value])
  if (!value || !stats) return null
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
          {badge && <span className="nl-metric nl-metric-lang">{badge}</span>}
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

type TextPaneProps = PaneProps & {
  variant?: string
}

function TextPane({ label, value, badge, variant, area }: TextPaneProps) {
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

type NlLatentsDetailPageProps = {
  backSearch: string
  initialError: string | null
  sample: NlLatentsSampleDetail | null
}

export default function NlLatentsDetailPage({
  backSearch,
  initialError,
  sample,
}: NlLatentsDetailPageProps) {
  const [copied, copy] = useCopy()
  const backHref = backSearch ? `/nl-latents?${backSearch}` : '/nl-latents'

  return (
    <div className="page nl-page nl-detail">
      <div className="nl-back-row">
        <Link href={backHref}>← Back to samples</Link>
      </div>

      {initialError && (
        <div className="error-state nl-error">
          <span className="error-icon">!</span>
          <div>
            <p className="error-title">Failed to load sample</p>
            <p className="error-detail">{initialError}</p>
          </div>
        </div>
      )}

      {sample && (
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
              {sample.prompt_block_names.length > 0 ? (
                <div className="nl-chips">
                  {sample.prompt_block_names.map((name, index) => (
                    <span className="nl-chip" key={`${name}-${index}`}>
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
              <MetaChip
                label="sample"
                value={sample.sample_id}
                copied={copied}
                onCopy={copy}
              />
              <MetaChip
                label="config"
                value={sample.config_id}
                copied={copied}
                onCopy={copy}
              />
              <MetaChip
                label="call"
                value={sample.call_id}
                copied={copied}
                onCopy={copy}
              />
              <MetaChip
                label="run"
                value={sample.run_id}
                copied={copied}
                onCopy={copy}
              />
              <MetaChip
                label="created"
                value={sample.created_at}
                display={shortDate(sample.created_at)}
                copied={copied}
                onCopy={copy}
              />
              <MetaChip
                label="finish"
                value={sample.finish_reason}
                display={sample.finish_reason}
                copied={copied}
                onCopy={copy}
              />
              <MetaChip
                label="version"
                value={sample.task_data_version}
                display={sample.task_data_version}
                copied={copied}
                onCopy={copy}
              />
            </div>
          </section>

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
