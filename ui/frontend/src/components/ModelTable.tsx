import { cn } from '@/lib/cn'
import { Tag } from '@/components/primitives'
import type { ModelEntry } from '@/lib/types'

const HEADER_CLASS =
  'border-b border-[var(--border)] px-3 py-2 font-display text-[10px] font-semibold tracking-[0.06em] whitespace-nowrap text-[var(--text-muted)] uppercase'
const CELL_CLASS = 'border-b border-[var(--border-subtle)] px-3 py-2 align-middle'

type CapIconProps = {
  value: boolean | null | undefined
  label: string
}

function CapIcon({ value, label }: CapIconProps) {
  if (value === null || value === undefined) {
    return (
      <span
        className="text-[11px] font-semibold text-[var(--text-muted)]"
        title={`${label}: unknown`}
      >
        ?
      </span>
    )
  }
  return value ? (
    <span
      className="text-[13px] font-semibold text-[var(--green)]"
      title={`${label}: yes`}
    >
      &#10003;
    </span>
  ) : (
    <span className="text-[13px] text-[var(--text-muted)]" title={`${label}: no`}>
      &mdash;
    </span>
  )
}

type ControlModeProps = {
  value: string | null
}

function ControlMode({ value }: ControlModeProps) {
  if (!value) {
    return <span className="text-[var(--text-muted)]">&mdash;</span>
  }
  return (
    <code className="font-mono text-xs text-[var(--text-secondary)]">
      {value}
    </code>
  )
}

type ModelTableProps = {
  models: ModelEntry[]
}

export default function ModelTable({ models }: ModelTableProps) {
  return (
    <div className="mt-2 overflow-x-auto">
      <table className="w-full border-collapse text-[13px]">
        <thead>
          <tr>
            <th className={cn(HEADER_CLASS, 'text-left')}>Model</th>
            <th className={cn(HEADER_CLASS, 'text-left')}>Display name</th>
            <th className={cn(HEADER_CLASS, 'w-24 text-center')}>Control mode</th>
            <th className={cn(HEADER_CLASS, 'w-16 text-center')}>Vision</th>
            <th className={cn(HEADER_CLASS, 'w-16 text-center')}>Context</th>
            <th className={cn(HEADER_CLASS, 'w-20 text-center')}>Source</th>
          </tr>
        </thead>
        <tbody>
          {models.map(model => (
            <tr
              key={`${model.provider}:${model.model}`}
              className="transition-colors last:[&>td]:border-b-0 hover:bg-[var(--bg-hover)]"
            >
              <td className={CELL_CLASS}>
                <span className="font-mono text-xs text-[var(--text-primary)]">
                  {model.model}
                </span>
              </td>
              <td className={cn(CELL_CLASS, 'text-[var(--text-secondary)]')}>
                {model.display_name && model.display_name !== model.model ? (
                  model.display_name
                ) : (
                  <span className="text-[var(--text-muted)]">&mdash;</span>
                )}
              </td>
              <td className={cn(CELL_CLASS, 'w-24 text-center')}>
                <ControlMode value={model.control_mode} />
              </td>
              <td className={cn(CELL_CLASS, 'w-16 text-center')}>
                <CapIcon value={model.supports_vision} label="Vision" />
              </td>
              <td
                className={cn(
                  CELL_CLASS,
                  'w-16 text-center font-mono text-xs text-[var(--text-secondary)]',
                )}
              >
                {model.context_window ? (
                  `${(model.context_window / 1000).toFixed(0)}k`
                ) : (
                  <span className="text-[var(--text-muted)]">&mdash;</span>
                )}
              </td>
              <td className={cn(CELL_CLASS, 'w-20 text-center')}>
                <Tag
                  tone={model.source_quality === 'live' ? 'green' : 'yellow'}
                  className="tracking-[0.04em] uppercase"
                >
                  {model.source_quality}
                </Tag>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
