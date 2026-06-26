import type { ModelEntry } from '@/lib/types'

type CapIconProps = {
  value: boolean | null | undefined
  label: string
}

function CapIcon({ value, label }: CapIconProps) {
  if (value === null || value === undefined) {
    return (
      <span className="cap-icon cap-unknown" title={`${label}: unknown`}>
        ?
      </span>
    )
  }
  return value ? (
    <span className="cap-icon cap-yes" title={`${label}: yes`}>
      &#10003;
    </span>
  ) : (
    <span className="cap-icon cap-no" title={`${label}: no`}>
      &mdash;
    </span>
  )
}

type ControlModeProps = {
  value: string | null
}

function ControlMode({ value }: ControlModeProps) {
  if (!value) {
    return <span className="text-muted">&mdash;</span>
  }
  return <code>{value}</code>
}

type ModelTableProps = {
  models: ModelEntry[]
}

export default function ModelTable({ models }: ModelTableProps) {
  return (
    <div className="model-table-wrapper">
      <table className="model-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Display Name</th>
            <th className="cap-col">Control Mode</th>
            <th className="cap-col">Vision</th>
            <th className="cap-col">Context</th>
            <th className="cap-col">Source</th>
          </tr>
        </thead>
        <tbody>
          {models.map(model => (
            <tr key={`${model.provider}:${model.model}`}>
              <td className="model-id">
                <code>{model.model}</code>
              </td>
              <td className="model-display">
                {model.display_name && model.display_name !== model.model ? (
                  model.display_name
                ) : (
                  <span className="text-muted">&mdash;</span>
                )}
              </td>
              <td className="cap-col">
                <ControlMode value={model.control_mode} />
              </td>
              <td className="cap-col">
                <CapIcon value={model.supports_vision} label="Vision" />
              </td>
              <td className="cap-col context-val">
                {model.context_window ? (
                  `${(model.context_window / 1000).toFixed(0)}k`
                ) : (
                  <span className="text-muted">&mdash;</span>
                )}
              </td>
              <td className="cap-col">
                <span className={`source-tag source-${model.source_quality}`}>
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
