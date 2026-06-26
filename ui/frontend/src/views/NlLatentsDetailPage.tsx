'use client'

import Link from 'next/link'
import type { NlLatentsSampleDetail } from '@/lib/types'
import { effortLabel, formatPercent, formatSeconds, shortDate } from '@/lib/format'
import { useCopy } from '@/hooks/useCopy'
import { Dot, ResultBadge, SECTION_LABEL } from '@/components/primitives'
import { CodePane } from '@/components/code/CodePane'
import { StatCell } from '@/components/stats/StatCell'
import { BudgetStat } from '@/components/stats/BudgetStat'
import { LatentSection } from '@/components/panels/LatentSection'
import { ErrorSection } from '@/components/panels/ErrorSection'
import { TextPanel } from '@/components/panels/TextPanel'
import { IdChip } from '@/components/chips/IdChip'

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
                    language={sample.language}
                    badge={sample.language}
                  />
                  <CodePane
                    label="Decoded code"
                    value={sample.decoded_code}
                    language={sample.language}
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
