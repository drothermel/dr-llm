import './ModelTable.css'

function CapIcon({ value, label }) {
  if (value === null || value === undefined) {
    return <span className="cap-icon cap-unknown" title={`${label}: unknown`}>?</span>
  }
  return value
    ? <span className="cap-icon cap-yes" title={`${label}: yes`}>&#10003;</span>
    : <span className="cap-icon cap-no" title={`${label}: no`}>&mdash;</span>
}

export default function ModelTable({ models }) {
  return (
    <div className="model-table-wrapper">
      <table className="model-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Display Name</th>
            <th className="cap-col">Reasoning</th>
            <th className="cap-col">Vision</th>
            <th className="cap-col">Context</th>
            <th className="cap-col">Source</th>
          </tr>
        </thead>
        <tbody>
          {models.map(m => (
            <tr key={`${m.provider}:${m.model}`}>
              <td className="model-id">
                <code>{m.model}</code>
              </td>
              <td className="model-display">
                {m.display_name && m.display_name !== m.model
                  ? m.display_name
                  : <span className="text-muted">&mdash;</span>
                }
              </td>
              <td className="cap-col">
                <CapIcon value={m.supports_reasoning} label="Reasoning" />
              </td>
              <td className="cap-col">
                <CapIcon value={m.supports_vision} label="Vision" />
              </td>
              <td className="cap-col context-val">
                {m.context_window
                  ? `${(m.context_window / 1000).toFixed(0)}k`
                  : <span className="text-muted">&mdash;</span>
                }
              </td>
              <td className="cap-col">
                <span className={`source-tag source-${m.source_quality}`}>
                  {m.source_quality}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
