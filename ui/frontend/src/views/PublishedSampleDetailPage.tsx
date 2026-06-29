'use client'

import Link from 'next/link'
import type { PublishedSampleDetail } from '@/lib/types'
import { CodePane } from '@/components/code/CodePane'
import { TextPanel } from '@/components/panels/TextPanel'
import { Dot, ResultBadge, SECTION_LABEL, Tag } from '@/components/primitives'

type PublishedSampleDetailPageProps = {
  backSearch: string
  initialError: string | null
  sample: PublishedSampleDetail | null
}

type DetailItem = {
  label: string
  value: string | number | boolean | null | undefined
}

function prettyJson(value: unknown): string | null {
  return value === null || value === undefined
    ? null
    : JSON.stringify(value, null, 2)
}

function displayValue(value: DetailItem['value']): string {
  if (value === null || value === undefined || value === '') return 'none'
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'number') return value.toLocaleString()
  return value
}

function hasAnyValue(items: DetailItem[]): boolean {
  return items.some(
    item =>
      item.value !== null && item.value !== undefined && item.value !== '',
  )
}

function shortDate(value: string | null): string {
  if (!value) return 'unknown'
  return new Date(value).toLocaleString()
}

function roleLabel(value: string): string {
  return value.replaceAll('_', ' ')
}

function DetailGrid({
  label,
  items,
}: {
  label: string
  items: DetailItem[]
}) {
  if (!hasAnyValue(items)) return null
  return (
    <section className="flex flex-col gap-2.5">
      <span className={SECTION_LABEL}>{label}</span>
      <div className="grid grid-cols-3 gap-2 max-lg:grid-cols-2 max-sm:grid-cols-1">
        {items.map(item => (
          <div
            key={item.label}
            className="rounded-md border border-[var(--border-subtle)] bg-[var(--bg-tertiary)] px-3 py-2"
          >
            <div className="text-[10px] font-semibold tracking-[0.06em] text-[var(--text-muted)] uppercase">
              {item.label}
            </div>
            <div className="mt-1 font-mono text-[12px] break-all text-[var(--text-primary)]">
              {displayValue(item.value)}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

export default function PublishedSampleDetailPage({
  backSearch,
  initialError,
  sample,
}: PublishedSampleDetailPageProps) {
  const backHref = backSearch ? `/samples?${backSearch}` : '/samples'

  return (
    <div className="w-full">
      <div className="mb-6 max-w-[1280px]">
        <Link
          href={backHref}
          className="inline-flex items-center gap-1.5 text-[13px] font-medium text-[var(--text-secondary)] transition-colors hover:text-[var(--text-primary)]"
        >
          <span aria-hidden="true">&lt;-</span> Back to samples
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
          <header className="mb-7 flex max-w-[1280px] items-start justify-between gap-6 max-md:flex-col">
            <div className="min-w-0">
              <div className="mb-2 flex flex-wrap items-center gap-2 text-[13px] text-[var(--text-secondary)]">
                <Tag mono>{sample.source_project}</Tag>
                <Dot />
                <span className="font-mono">{sample.source_pool}</span>
                <Dot />
                <span>{roleLabel(sample.sample_role)}</span>
                {sample.task_family && (
                  <>
                    <Dot />
                    <span>{sample.task_family}</span>
                  </>
                )}
              </div>
              <h1 className="font-mono text-[26px] leading-tight font-semibold break-all text-[var(--text-primary)]">
                {sample.source_sample_id}
              </h1>
            </div>
            <ResultBadge
              state={sample.result_state}
              failure={sample.failure_category}
            />
          </header>

          <section className="mb-8 grid max-w-[1280px] grid-cols-4 gap-px border-y border-[var(--border)] bg-[var(--border-subtle)] max-lg:grid-cols-2">
            {[
              ['Model', sample.model ?? sample.llm_config_id ?? 'unknown'],
              [
                'Dataset',
                sample.dataset_id ?? sample.task_id ?? 'unknown',
              ],
              ['Budget', sample.budget_label ?? String(sample.budget_chars ?? '')],
              ['Created', shortDate(sample.created_at)],
            ].map(([label, value]) => (
              <div key={label} className="bg-[var(--bg-primary)] px-4 py-3">
                <div className={SECTION_LABEL}>{label}</div>
                <div className="mt-1 font-mono text-[13px] break-all text-[var(--text-primary)]">
                  {value || 'unknown'}
                </div>
              </div>
            ))}
          </section>

          <section className="mb-8 flex max-w-[1280px] flex-col gap-2.5">
            <span className={SECTION_LABEL}>Provenance</span>
            <div className="grid grid-cols-2 gap-2 max-lg:grid-cols-1">
              {[
                ['source table', sample.source_table],
                ['run', sample.run_id],
                ['prompt template', sample.prompt_template_id],
                ['llm config', sample.llm_config_id],
                ['encoder sample', sample.enc_sample_id],
                ['upstream', sample.upstream_sample_id],
              ].map(([label, value]) => (
                <div
                  key={label}
                  className="rounded-md border border-[var(--border-subtle)] bg-[var(--bg-tertiary)] px-3 py-2"
                >
                  <div className="text-[10px] font-semibold tracking-[0.06em] text-[var(--text-muted)] uppercase">
                    {label}
                  </div>
                  <div className="mt-1 font-mono text-[12px] break-all text-[var(--text-primary)]">
                    {value ?? 'none'}
                  </div>
                </div>
              ))}
            </div>
          </section>

          <div className="flex max-w-[1280px] flex-col gap-5">
            <div className="grid grid-cols-2 items-start gap-5 max-lg:grid-cols-1">
              <TextPanel
                label={sample.input_text_source ?? 'Input'}
                value={sample.input_text}
              />
              <TextPanel
                label={`${sample.output_kind} output`}
                value={sample.output_text}
              />
            </div>

            {(sample.input_text || sample.output_text) && (
              <div className="grid grid-cols-2 items-start gap-4 max-lg:grid-cols-1">
                <CodePane
                  label="Input text"
                  value={sample.input_text}
                  language={sample.language}
                  badge={sample.language}
                />
                <CodePane
                  label="Output text"
                  value={sample.output_text}
                  language={sample.language}
                  badge={sample.language}
                  accent
                />
              </div>
            )}

            <DetailGrid
              label="LLM result"
              items={[
                { label: 'provider', value: sample.provider },
                { label: 'model', value: sample.model },
                { label: 'mode', value: sample.mode },
                { label: 'finish reason', value: sample.finish_reason },
                { label: 'warnings', value: sample.warning_count },
                { label: 'error', value: sample.error_text },
              ]}
            />

            <DetailGrid
              label="Usage"
              items={[
                { label: 'prompt tokens', value: sample.prompt_tokens },
                {
                  label: 'completion tokens',
                  value: sample.completion_tokens,
                },
                { label: 'reasoning tokens', value: sample.reasoning_tokens },
                { label: 'total tokens', value: sample.total_tokens },
                {
                  label: 'computed total',
                  value: sample.computed_total_tokens,
                },
              ]}
            />

            <DetailGrid
              label="Cost"
              items={[
                { label: 'total usd', value: sample.total_cost_usd },
                { label: 'prompt usd', value: sample.prompt_cost_usd },
                {
                  label: 'completion usd',
                  value: sample.completion_cost_usd,
                },
                { label: 'reasoning usd', value: sample.reasoning_cost_usd },
                { label: 'currency', value: sample.cost_currency },
              ]}
            />

            <DetailGrid
              label="Validation"
              items={[
                { label: 'passed', value: sample.passed },
                { label: 'pass rate', value: sample.validation_pass_rate },
                {
                  label: 'validation seconds',
                  value: sample.validation_time_seconds,
                },
                { label: 'compiles', value: sample.compiles },
                { label: 'compile error', value: sample.compile_error },
                { label: 'has code fences', value: sample.has_code_fences },
                {
                  label: 'has expected function',
                  value: sample.has_expected_function,
                },
              ]}
            />

            {sample.test_failures.length > 0 && (
              <section className="flex flex-col gap-3">
                <span className={SECTION_LABEL}>Failed tests</span>
                {sample.test_failures.map((failure, index) => (
                  <div
                    key={`${failure.case_key ?? 'case'}-${failure.case_idx ?? index}`}
                    className="grid grid-cols-2 items-start gap-4 border-t border-[var(--border-subtle)] pt-4 max-lg:grid-cols-1"
                  >
                    <CodePane
                      label={`Failure ${failure.case_key ?? failure.case_idx ?? index}`}
                      value={prettyJson(failure.failure_json)}
                      language="json"
                      badge="json"
                    />
                    <CodePane
                      label="Input"
                      value={prettyJson(failure.input_json)}
                      language="json"
                      badge="json"
                    />
                    <CodePane
                      label="Expected"
                      value={prettyJson(failure.expected_json)}
                      language="json"
                      badge="json"
                    />
                    <CodePane
                      label="Actual"
                      value={prettyJson(failure.actual_json)}
                      language="json"
                      badge="json"
                      accent
                    />
                    <TextPanel
                      label="Error"
                      value={failure.error_text}
                      className="col-span-2 max-lg:col-span-1"
                    />
                  </div>
                ))}
              </section>
            )}
          </div>
        </>
      )}
    </div>
  )
}
