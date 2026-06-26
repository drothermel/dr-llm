'use client'

import Link from 'next/link'
import { useMemo, useState } from 'react'
import hljs from 'highlight.js/lib/core'
import python from 'highlight.js/lib/languages/python'
import type { NlLatentsSampleDetail } from '@/lib/types'
import { Dot, ResultBadge, SECTION_LABEL } from '@/components/primitives'

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
  return value.length > max ? `${value.slice(0, max)}…` : value
}

const EFFORT_HIDDEN = new Set(['', 'default', 'none', 'auto', 'na', 'n/a'])

function effortLabel(value: string | null | undefined): string | null {
  if (!value || EFFORT_HIDDEN.has(String(value).toLowerCase())) return null
  return String(value)
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

type StatCellProps = {
  label: string
  value: string | null
  sub?: string | null
  mono?: boolean
}

function StatCell({ label, value, sub, mono }: StatCellProps) {
  return (
    <div className="flex min-w-0 flex-col gap-1.5 bg-[var(--bg-primary)] px-5 py-4">
      <span className={SECTION_LABEL}>{label}</span>
      <span
        className={`[overflow-wrap:anywhere] text-sm leading-snug font-semibold text-[var(--text-primary)] ${
          mono ? 'font-mono' : ''
        }`}
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

type BudgetStatProps = {
  budget: string | null
  actual: number | null
  budgetOk: boolean | null
}

function BudgetStat({ budget, actual, budgetOk }: BudgetStatProps) {
  const limit = Number(budget)
  const used = actual === null || actual === undefined ? null : Number(actual)
  const hasMeter = Number.isFinite(limit) && limit > 0 && used !== null
  const ratio = hasMeter ? used / limit : 0
  const over = budgetOk === false || (hasMeter && used > limit)

  return (
    <div className="flex min-w-0 flex-col gap-1.5 bg-[var(--accent-bg)] px-5 py-4">
      <span className={SECTION_LABEL}>Budget</span>
      <span className="flex items-baseline gap-1.5">
        <strong className="font-mono text-xl leading-none font-semibold text-[var(--text-primary)]">
          {used ?? '—'}
        </strong>
        <span className="font-mono text-[13px] text-[var(--text-secondary)]">
          / {Number.isFinite(limit) ? limit : (budget ?? '—')} chars
        </span>
      </span>
      {hasMeter && (
        <span
          className="mt-0.5 block h-1.5 overflow-hidden rounded-full bg-[color-mix(in_oklch,var(--accent)_18%,transparent)]"
          aria-hidden="true"
        >
          <span
            className={`block h-full rounded-full transition-[width] duration-200 ${
              over ? 'bg-[var(--red)]' : 'bg-[var(--accent)]'
            }`}
            style={{ width: `${Math.min(100, Math.max(2, ratio * 100))}%` }}
          />
        </span>
      )}
      <span
        className={`text-xs font-medium ${
          over ? 'text-[var(--red)]' : 'text-[var(--accent-strong)]'
        }`}
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

type IdChipProps = {
  label: string
  value: string | number | null | undefined
  display?: string | null
  copied: string | null
  onCopy: (value: string) => void
}

function IdChip({ label, value, display, copied, onCopy }: IdChipProps) {
  if (value === null || value === undefined || value === '') return null
  const stringValue = String(value)
  const isCopied = copied === stringValue
  return (
    <button
      type="button"
      className="group inline-flex cursor-pointer items-baseline gap-1.5 rounded px-1.5 py-0.5 text-left transition-colors hover:bg-[var(--bg-hover)]"
      title={`${label}: ${stringValue} (click to copy)`}
      onClick={() => onCopy(stringValue)}
    >
      <span className="text-[10px] font-semibold tracking-[0.06em] text-[var(--text-muted)] uppercase">
        {label}
      </span>
      <span className="font-mono text-[12px] text-[var(--text-secondary)] group-hover:text-[var(--text-primary)]">
        {isCopied ? 'copied ✓' : (display ?? truncate(stringValue, 14))}
      </span>
    </button>
  )
}

type CodePaneProps = {
  label: string
  value: string | null | undefined
  badge?: string | null
  accent?: boolean
}

function CodePane({ label, value, badge, accent }: CodePaneProps) {
  const html = useMemo(() => {
    if (!value) return ''
    try {
      return hljs.highlight(value, { language: 'python' }).value
    } catch {
      return null
    }
  }, [value])
  const stats = useMemo(() => (value ? codeStats(value) : null), [value])
  if (!value || !stats) return null
  return (
    <section className="flex min-w-0 flex-col self-stretch overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--bg-primary)]">
      <header className="flex items-center gap-2.5 border-b border-[var(--border-subtle)] bg-[var(--bg-secondary)] px-4 py-2.5">
        <span
          className={`h-1.5 w-1.5 shrink-0 rounded-full ${
            accent ? 'bg-[var(--accent)]' : 'bg-[var(--border-strong)]'
          }`}
          aria-hidden="true"
        />
        <h3 className="font-display text-[12px] font-semibold tracking-[0.04em] text-[var(--text-primary)] uppercase">
          {label}
        </h3>
        <div className="ml-auto flex flex-wrap items-center justify-end gap-1.5">
          {[
            `${stats.lines.toLocaleString()} lines`,
            `${stats.chars.toLocaleString()} chars`,
            formatBytes(stats.bytes),
          ].map(text => (
            <span
              key={text}
              className="rounded bg-[var(--bg-tertiary)] px-[7px] py-0.5 font-mono text-[11px] whitespace-nowrap text-[var(--text-muted)]"
            >
              {text}
            </span>
          ))}
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

type TextSectionProps = {
  label: string
  value: string | null | undefined
}

/** Prose-forward section for the NL latent — accent framed, with chars + bytes. */
function LatentSection({ label, value }: TextSectionProps) {
  if (!value) return null
  const chars = [...value].length
  const bytes = new TextEncoder().encode(value).length
  return (
    <section className="overflow-hidden rounded-xl border border-[color-mix(in_oklch,var(--accent)_28%,var(--border))] bg-[var(--accent-bg)]">
      <header className="flex items-center gap-2.5 border-b border-[color-mix(in_oklch,var(--accent)_18%,var(--border-subtle))] px-4 py-2.5">
        <h3 className="font-display text-[12px] font-semibold tracking-[0.04em] text-[var(--accent-strong)] uppercase">
          {label}
        </h3>
        <div className="ml-auto flex items-center gap-1.5">
          {[`${chars.toLocaleString()} chars`, formatBytes(bytes)].map(text => (
            <span
              key={text}
              className="rounded bg-[color-mix(in_oklch,var(--accent)_12%,transparent)] px-[7px] py-0.5 font-mono text-[11px] whitespace-nowrap text-[var(--accent-strong)]"
            >
              {text}
            </span>
          ))}
        </div>
      </header>
      <p className="px-5 py-4 text-[15px] leading-[1.65] text-[var(--text-primary)]">
        {value}
      </p>
    </section>
  )
}

/** Red error section, full width on failure. */
function ErrorSection({ label, value }: TextSectionProps) {
  if (!value) return null
  return (
    <section className="overflow-hidden rounded-xl border border-[var(--red-border)] bg-[var(--red-bg)]">
      <header className="border-b border-[color-mix(in_oklch,var(--red)_18%,var(--border-subtle))] px-4 py-2.5">
        <h3 className="font-display text-[12px] font-semibold tracking-[0.04em] text-[var(--red)] uppercase">
          {label}
        </h3>
      </header>
      <pre className="m-0 max-h-[320px] overflow-auto px-4 py-3 font-mono text-[12.5px] leading-relaxed whitespace-pre-wrap break-words text-[var(--red)]">
        {value}
      </pre>
    </section>
  )
}

/** Expanded reference text panel (prompts) — always visible, no char count. */
function TextPanel({ label, value }: TextSectionProps) {
  if (!value) return null
  return (
    <section className="overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--bg-primary)]">
      <header className="border-b border-[var(--border-subtle)] bg-[var(--bg-secondary)] px-4 py-2.5">
        <h3 className="font-display text-[12px] font-semibold tracking-[0.04em] text-[var(--text-secondary)] uppercase">
          {label}
        </h3>
      </header>
      <pre className="m-0 overflow-auto px-4 py-3 font-mono text-[12.5px] leading-relaxed whitespace-pre-wrap break-words text-[var(--text-secondary)]">
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
    <div className="w-full">
      <div className="mb-6 max-w-[1280px]">
        <Link
          href={backHref}
          className="inline-flex items-center gap-1.5 text-[13px] font-medium text-[var(--text-secondary)] transition-colors hover:text-[var(--text-primary)]"
        >
          <span aria-hidden="true">←</span> Back to samples
        </Link>
      </div>

      {initialError && (
        <div className="mb-5 flex items-start gap-4 rounded-xl border border-[var(--red-border)] bg-[var(--red-bg)] p-6">
          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[var(--red)] text-sm font-bold text-white">
            !
          </span>
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
          {/* Masthead */}
          <header className="mb-8 flex max-w-[1280px] items-start justify-between gap-6 max-md:flex-col">
            <div className="min-w-0">
              <div className="mb-2 flex flex-wrap items-center gap-2 text-[13px] text-[var(--text-secondary)]">
                <span className="font-semibold text-[var(--text-primary)]">
                  {sample.family}
                </span>
                <Dot />
                <span>difficulty {sample.difficulty}</span>
                <Dot />
                <span>{sample.split}</span>
                {sample.language && (
                  <>
                    <Dot />
                    <span>{sample.language}</span>
                  </>
                )}
              </div>
              <h1 className="font-mono text-[26px] leading-tight font-semibold tracking-[-0.01em] break-all text-[var(--text-primary)]">
                {sample.task_id}
              </h1>
            </div>
            <ResultBadge
              state={sample.result_state}
              failure={
                sample.failure_category_normalized ?? sample.failure_category
              }
            />
          </header>

          {/* Stat bar — flat, hairline-bounded, no nested cards */}
          <section className="mb-7 grid max-w-[1280px] grid-cols-4 gap-px border-y border-[var(--border)] bg-[var(--border-subtle)] max-md:grid-cols-2">
            <BudgetStat
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
          </section>

          {/* Provenance — recessive metadata row */}
          <section className="mb-8 flex max-w-[1280px] flex-col gap-2.5">
            <span className={SECTION_LABEL}>Provenance</span>
            {sample.prompt_block_names.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {sample.prompt_block_names.map((name, index) => (
                  <span
                    key={`${name}-${index}`}
                    className="rounded border border-[var(--border-subtle)] bg-[var(--bg-tertiary)] px-2 py-0.5 text-xs font-medium text-[var(--text-secondary)]"
                  >
                    {name}
                  </span>
                ))}
              </div>
            )}
            {sample.run_id && (
              <button
                type="button"
                className="group flex w-full items-baseline gap-2 rounded-md border border-[var(--border-subtle)] bg-[var(--bg-tertiary)] px-3 py-2 text-left transition-colors hover:border-[var(--border-strong)]"
                title={`run: ${sample.run_id} (click to copy)`}
                onClick={() => copy(sample.run_id)}
              >
                <span className="shrink-0 text-[10px] font-semibold tracking-[0.06em] text-[var(--text-muted)] uppercase">
                  Run
                </span>
                <span className="[overflow-wrap:anywhere] font-mono text-[12.5px] text-[var(--text-primary)]">
                  {copied === sample.run_id ? 'copied ✓' : sample.run_id}
                </span>
              </button>
            )}
            <div className="-ml-1.5 flex flex-wrap items-center gap-x-1 gap-y-1">
              <IdChip label="sample" value={sample.sample_id} copied={copied} onCopy={copy} />
              <IdChip label="config" value={sample.config_id} copied={copied} onCopy={copy} />
              <IdChip label="call" value={sample.call_id} copied={copied} onCopy={copy} />
              <IdChip
                label="created"
                value={sample.created_at}
                display={shortDate(sample.created_at)}
                copied={copied}
                onCopy={copy}
              />
              <IdChip
                label="finish"
                value={sample.finish_reason}
                display={sample.finish_reason}
                copied={copied}
                onCopy={copy}
              />
              <IdChip
                label="version"
                value={sample.task_data_version}
                display={sample.task_data_version}
                copied={copied}
                onCopy={copy}
              />
            </div>
          </section>

          {/* Content */}
          <div className="flex flex-col gap-5">
            {/* Reading-width content stays capped */}
            <div className="flex max-w-[1280px] flex-col gap-5">
              <ErrorSection label="Error detail" value={sample.error_detail} />

              {/* Encoder (left) · Decoder (right) */}
              <div className="grid grid-cols-2 items-start gap-x-5 gap-y-5 max-lg:grid-cols-1">
              <div className="flex min-w-0 flex-col gap-5">
                <TextPanel
                  label="Encoder prompt"
                  value={sample.enc_prompt_instructions ?? sample.enc_prompt}
                />
                <LatentSection
                  label="Encoded output · NL latent"
                  value={sample.description}
                />
              </div>
              <div className="flex min-w-0 flex-col gap-5">
                <TextPanel
                  label="Decoder system prompt"
                  value={sample.dec_system}
                />
                <TextPanel label="Decoder task" value={sample.dec_task} />
              </div>
              </div>
            </div>

            {(sample.input_code || sample.decoded_code) && (
              <div className="flex flex-col gap-2.5">
                <span className={SECTION_LABEL}>Code · input → decoded</span>
                <div className="grid grid-cols-2 items-start gap-4 max-lg:grid-cols-1">
                  <CodePane
                    label="Input code"
                    value={sample.input_code}
                    badge={sample.language}
                  />
                  <CodePane
                    label="Decoded code"
                    value={sample.decoded_code}
                    badge={sample.language}
                    accent
                  />
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
