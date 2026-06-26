'use client'

import Link from 'next/link'
import { useMemo, useState } from 'react'
import hljs from 'highlight.js/lib/core'
import python from 'highlight.js/lib/languages/python'
import type { NlLatentsSampleDetail } from '@/lib/types'

hljs.registerLanguage('python', python)

const ERROR_STATE_CLASS =
  'mb-5 flex items-start gap-4 rounded-[10px] border border-[rgba(220,38,38,0.15)] bg-[var(--red-bg)] p-6'
const ERROR_ICON_CLASS =
  'flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[var(--red)] text-sm font-bold text-white'
const BADGE_CLASS_BY_STATE: Record<string, string> = {
  passed: 'bg-[var(--green-bg)] text-[var(--green)]',
  failed: 'bg-[var(--red-bg)] text-[var(--red)]',
  pending: 'bg-[var(--yellow-bg)] text-[var(--yellow)]',
}

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
      className="inline-flex cursor-pointer items-center gap-1.5 rounded border border-[var(--border-subtle)] bg-[var(--bg-tertiary)] px-2 py-1 font-mono text-[11px] text-[var(--text-secondary)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]"
      title={`${label}: ${stringValue} (click to copy)`}
      onClick={() => onCopy(stringValue)}
    >
      <span className="font-sans text-[10px] font-semibold tracking-[0.4px] text-[var(--text-muted)] uppercase">
        {label}
      </span>
      <span>
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
    <span
      className={`inline-flex shrink-0 items-center gap-[7px] rounded-md px-2.5 py-1.5 text-xs leading-tight font-semibold ${BADGE_CLASS_BY_STATE[state] ?? 'bg-[var(--bg-tertiary)] text-[var(--text-secondary)]'}`}
    >
      <span
        className="h-[7px] w-[7px] shrink-0 rounded-full bg-current"
        aria-hidden="true"
      />
      <span className="capitalize">{state}</span>
      {failure && state === 'failed' && (
        <span className="border-l border-current pl-2 font-mono text-[11px] font-medium opacity-85">
          {failure}
        </span>
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
    <div className="flex min-w-0 flex-col gap-1 bg-[var(--bg-primary)] px-[18px] py-4">
      <span className="text-[11px] font-bold tracking-[0.6px] text-[var(--text-muted)] uppercase">
        {label}
      </span>
      <span
        className={`[overflow-wrap:anywhere] text-sm font-semibold text-[var(--text-primary)] ${mono ? 'font-mono' : ''}`}
      >
        {value ?? '—'}
      </span>
      {sub && (
        <span className="[overflow-wrap:anywhere] text-xs text-[var(--text-secondary)]">
          {sub}
        </span>
      )}
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

  return (
    <div className="flex min-w-0 flex-col gap-1 bg-[var(--accent-bg)] px-[18px] py-4">
      <span className="text-[11px] font-bold tracking-[0.6px] text-[var(--text-muted)] uppercase">
        Budget
      </span>
      <span className="flex items-baseline gap-1.5">
        <strong className="font-mono text-xl font-bold text-[var(--text-primary)]">
          {used ?? '—'}
        </strong>
        <span className="font-mono text-[13px] text-[var(--text-secondary)]">
          / {Number.isFinite(limit) ? limit : (budget ?? '—')} chars
        </span>
      </span>
      {hasMeter && (
        <span
          className="mt-0.5 block h-1.5 overflow-hidden rounded-full bg-[oklch(0.9_0.012_272)]"
          aria-hidden="true"
        >
          <span
            className={`block h-full rounded-full transition-[width] ${over ? 'bg-[var(--red)]' : 'bg-[var(--accent)]'}`}
            style={{ width: `${Math.min(100, Math.max(2, ratio * 100))}%` }}
          />
        </span>
      )}
      <span
        className={`text-xs font-medium ${over ? 'text-[var(--red)]' : 'text-[var(--accent)]'}`}
      >
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
}

type CodePaneProps = PaneProps & {
  language?: string
}

function CodePane({
  label,
  value,
  badge,
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
    <section
      className="flex min-w-0 flex-col self-stretch overflow-hidden rounded-[10px] border border-[var(--border)] bg-[var(--bg-primary)]"
    >
      <header className="flex items-center gap-2.5 border-b border-[var(--border-subtle)] bg-[var(--bg-secondary)] px-3.5 py-2.5">
        <h3 className="text-[11px] font-bold tracking-[0.6px] text-[var(--text-secondary)] uppercase">
          {label}
        </h3>
        <div className="ml-auto flex flex-wrap items-center justify-end gap-1.5">
          <span className="rounded bg-[var(--bg-tertiary)] px-[7px] py-0.5 font-mono text-[11px] whitespace-nowrap text-[var(--text-muted)]">
            {stats.lines.toLocaleString()} lines
          </span>
          <span className="rounded bg-[var(--bg-tertiary)] px-[7px] py-0.5 font-mono text-[11px] whitespace-nowrap text-[var(--text-muted)]">
            {stats.chars.toLocaleString()} chars
          </span>
          <span className="rounded bg-[var(--bg-tertiary)] px-[7px] py-0.5 font-mono text-[11px] whitespace-nowrap text-[var(--text-muted)]">
            {formatBytes(stats.bytes)}
          </span>
          {badge && (
            <span className="rounded bg-[var(--accent-bg)] px-[7px] py-0.5 font-mono text-[11px] whitespace-nowrap text-[var(--accent)]">
              {badge}
            </span>
          )}
        </div>
      </header>
      <pre className="m-0 flex-1 overflow-auto bg-[oklch(0.99_0.002_270)] p-4 font-mono text-[12.5px] leading-relaxed whitespace-pre text-[var(--text-primary)]">
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

function TextPane({ label, value, badge, variant }: TextPaneProps) {
  if (!value) return null
  const isLatent = variant === 'latent'
  const isError = variant === 'error'
  const isMuted = variant === 'muted'
  return (
    <section
      className={`min-w-0 overflow-hidden rounded-[10px] border bg-[var(--bg-primary)] ${
        isLatent
          ? 'border-[oklch(0.86_0.05_272)]'
          : isError
            ? 'border-[var(--red-border)]'
            : 'border-[var(--border)]'
      }`}
    >
      <header
        className={`flex items-center gap-2.5 border-b border-[var(--border-subtle)] px-3.5 py-2.5 ${
          isLatent
            ? 'bg-[var(--accent-bg)]'
            : isError
              ? 'bg-[var(--red-bg)]'
              : 'bg-[var(--bg-secondary)]'
        }`}
      >
        <h3
          className={`text-[11px] font-bold tracking-[0.6px] uppercase ${
            isLatent
              ? 'text-[var(--accent-strong)]'
              : isError
                ? 'text-[var(--red)]'
                : isMuted
                  ? 'text-[var(--text-muted)]'
                  : 'text-[var(--text-secondary)]'
          }`}
        >
          {label}
        </h3>
        {badge && (
          <span className="ml-auto rounded bg-[var(--bg-tertiary)] px-[7px] py-0.5 font-mono text-[11px] text-[var(--text-muted)]">
            {badge}
          </span>
        )}
      </header>
      <pre
        className={`m-0 max-h-[420px] overflow-auto p-4 font-mono text-[12.5px] leading-relaxed whitespace-pre-wrap text-[var(--text-primary)] break-words ${
          isLatent
            ? 'bg-[var(--accent-bg)] font-sans text-sm leading-[1.65]'
            : isError
              ? 'bg-[var(--red-bg)] text-[var(--red)]'
              : isMuted
                ? 'max-h-[260px] text-[var(--text-secondary)]'
                : ''
        }`}
      >
        {value}
      </pre>
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
    <div className="w-full max-w-none">
      <div className="mb-[18px] text-[13px]">
        <Link href={backHref} className="font-medium">
          ← Back to samples
        </Link>
      </div>

      {initialError && (
        <div className={ERROR_STATE_CLASS}>
          <span className={ERROR_ICON_CLASS}>!</span>
          <div>
            <p className="mb-1 font-semibold text-[var(--red)]">
              Failed to load sample
            </p>
            <p className="text-[13px] text-[var(--text-secondary)]">
              {initialError}
            </p>
          </div>
        </div>
      )}

      {sample && (
        <>
          <header className="mb-6 flex items-start justify-between gap-5 max-md:flex-col">
            <div>
              <h1 className="break-words font-mono text-[22px] leading-tight font-semibold text-[var(--text-primary)]">
                {sample.task_id}
              </h1>
              <div className="mt-2 flex flex-wrap items-center gap-2 text-[13px] text-[var(--text-secondary)]">
                <span className="font-semibold text-[var(--text-primary)]">
                  {sample.family}
                </span>
                <span className="h-[3px] w-[3px] rounded-full bg-[var(--border-strong)]" />
                <span>difficulty {sample.difficulty}</span>
                <span className="h-[3px] w-[3px] rounded-full bg-[var(--border-strong)]" />
                <span>{sample.split}</span>
                {sample.language && (
                  <>
                    <span className="h-[3px] w-[3px] rounded-full bg-[var(--border-strong)]" />
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

          <section className="mb-7 overflow-hidden rounded-[10px] border border-[var(--border)] bg-[var(--bg-primary)]">
            <div className="grid grid-cols-[repeat(auto-fit,minmax(190px,1fr))] gap-px border-b border-[var(--border-subtle)] bg-[var(--border-subtle)]">
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

            <div className="flex flex-col gap-[9px] border-b border-[var(--border-subtle)] px-[18px] py-4">
              <span className="text-[11px] font-bold tracking-[0.6px] text-[var(--text-muted)] uppercase">
                Prompt config
              </span>
              {sample.prompt_block_names.length > 0 ? (
                <div className="flex flex-wrap gap-1.5">
                  {sample.prompt_block_names.map((name, index) => (
                    <span
                      className="rounded border border-[var(--border-subtle)] bg-[var(--bg-tertiary)] px-[9px] py-1 text-xs font-medium text-[var(--text-secondary)]"
                      key={`${name}-${index}`}
                    >
                      {name}
                    </span>
                  ))}
                </div>
              ) : (
                <span className="text-sm font-semibold text-[var(--text-primary)]">
                  {sample.prompt_config_label ?? '—'}
                </span>
              )}
            </div>

            <div className="flex flex-wrap items-center gap-y-2 gap-x-[18px] bg-[var(--bg-secondary)] px-[18px] py-3 text-xs text-[var(--text-secondary)]">
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

          <div className="grid grid-cols-2 gap-y-4 gap-x-5 max-lg:grid-cols-1">
            <TextPane
              label="Encoded output"
              value={sample.description}
              variant="latent"
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
            />

            <TextPane
              label="Encoder prompt"
              value={sample.enc_prompt_instructions ?? sample.enc_prompt}
              variant="muted"
            />
            <div className="flex min-w-0 flex-col gap-4">
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
              />
            <CodePane
              label="Decoded code"
              value={sample.decoded_code}
              badge={sample.language}
            />
          </div>
        </>
      )}
    </div>
  )
}
