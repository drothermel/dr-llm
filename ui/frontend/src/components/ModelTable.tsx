import type { ModelEntry } from '@/lib/types'

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
    <span
      className="text-[13px] font-semibold text-[var(--text-muted)]"
      title={`${label}: no`}
    >
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
  return <code>{value}</code>
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
            <th className="border-b border-[var(--border)] px-3 py-2 text-left text-[11px] font-semibold whitespace-nowrap text-[var(--text-muted)] uppercase">
              Model
            </th>
            <th className="border-b border-[var(--border)] px-3 py-2 text-left text-[11px] font-semibold whitespace-nowrap text-[var(--text-muted)] uppercase">
              Display Name
            </th>
            <th className="w-20 border-b border-[var(--border)] px-3 py-2 text-center text-[11px] font-semibold whitespace-nowrap text-[var(--text-muted)] uppercase">
              Control Mode
            </th>
            <th className="w-20 border-b border-[var(--border)] px-3 py-2 text-center text-[11px] font-semibold whitespace-nowrap text-[var(--text-muted)] uppercase">
              Vision
            </th>
            <th className="w-20 border-b border-[var(--border)] px-3 py-2 text-center text-[11px] font-semibold whitespace-nowrap text-[var(--text-muted)] uppercase">
              Context
            </th>
            <th className="w-20 border-b border-[var(--border)] px-3 py-2 text-center text-[11px] font-semibold whitespace-nowrap text-[var(--text-muted)] uppercase">
              Source
            </th>
          </tr>
        </thead>
        <tbody>
          {models.map(model => (
            <tr
              key={`${model.provider}:${model.model}`}
              className="last:[&>td]:border-b-0 hover:bg-[var(--bg-hover)]"
            >
              <td className="border-b border-[var(--border-subtle)] px-3 py-[7px] align-middle">
                <code className="rounded bg-[var(--bg-tertiary)] px-1.5 py-0.5 font-mono text-xs text-[var(--text-primary)]">
                  {model.model}
                </code>
              </td>
              <td className="border-b border-[var(--border-subtle)] px-3 py-[7px] align-middle text-[var(--text-secondary)]">
                {model.display_name && model.display_name !== model.model ? (
                  model.display_name
                ) : (
                  <span className="text-[var(--text-muted)]">&mdash;</span>
                )}
              </td>
              <td className="w-20 border-b border-[var(--border-subtle)] px-3 py-[7px] text-center align-middle">
                <ControlMode value={model.control_mode} />
              </td>
              <td className="w-20 border-b border-[var(--border-subtle)] px-3 py-[7px] text-center align-middle">
                <CapIcon value={model.supports_vision} label="Vision" />
              </td>
              <td className="w-20 border-b border-[var(--border-subtle)] px-3 py-[7px] text-center align-middle font-mono text-xs text-[var(--text-secondary)]">
                {model.context_window ? (
                  `${(model.context_window / 1000).toFixed(0)}k`
                ) : (
                  <span className="text-[var(--text-muted)]">&mdash;</span>
                )}
              </td>
              <td className="w-20 border-b border-[var(--border-subtle)] px-3 py-[7px] text-center align-middle">
                <span
                  className={`rounded px-1.5 py-0.5 text-[10px] font-semibold tracking-[0.3px] uppercase ${
                    model.source_quality === 'live'
                      ? 'bg-[var(--green-bg)] text-[var(--green)]'
                      : 'bg-[var(--yellow-bg)] text-[var(--yellow)]'
                  }`}
                >
                  {model.source_quality}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
