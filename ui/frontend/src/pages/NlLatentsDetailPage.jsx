import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import './NlLatentsPage.css'

async function fetchJson(path) {
  const response = await fetch(path)
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }
  return response.json()
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined) return '-'
  return Number(value).toFixed(digits)
}

function DetailField({ label, value }) {
  return (
    <div className="nl-detail-field">
      <span>{label}</span>
      <strong>{value ?? '-'}</strong>
    </div>
  )
}

function CodeBlock({ title, value }) {
  if (!value) return null
  return (
    <section className="nl-detail-section">
      <h3>{title}</h3>
      <pre>{value}</pre>
    </section>
  )
}

export default function NlLatentsDetailPage() {
  const { sampleId } = useParams()
  const [sample, setSample] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true
    fetchJson(`/api/nl-latents/samples/${sampleId}`)
      .then(data => {
        if (!active) return
        setSample(data)
        setError(null)
        setLoading(false)
      })
      .catch(err => {
        if (!active) return
        setError(err.message)
        setLoading(false)
      })
    return () => {
      active = false
    }
  }, [sampleId])

  return (
    <div className="page nl-page">
      <div className="nl-back-row">
        <Link to="/nl-latents">Back to samples</Link>
      </div>

      {loading && <div className="loading-state">Loading sample...</div>}

      {error && (
        <div className="error-state nl-error">
          <span className="error-icon">!</span>
          <div>
            <p className="error-title">Failed to load sample</p>
            <p className="error-detail">{error}</p>
          </div>
        </div>
      )}

      {sample && (
        <>
          <div className="page-header nl-page-header">
            <div>
              <h2>{sample.sample_id}</h2>
              <p className="page-description">
                {sample.family} / difficulty {sample.difficulty} / {sample.split}
              </p>
            </div>
            <span className={`nl-result nl-result-${sample.result_state}`}>
              {sample.result_state}
            </span>
          </div>

          <section className="nl-detail-grid">
            <DetailField label="Encoder" value={sample.enc_model_label} />
            <DetailField label="Decoder" value={sample.dec_model_label} />
            <DetailField label="Budget" value={sample.budget} />
            <DetailField label="Actual chars" value={sample.actual_chars} />
            <DetailField label="Prompt config" value={sample.prompt_config_label} />
            <DetailField label="Task" value={sample.task_id} />
            <DetailField label="Data version" value={sample.task_data_version} />
            <DetailField
              label="Compile"
              value={
                sample.validation_compiles === null
                  ? null
                  : String(sample.validation_compiles)
              }
            />
            <DetailField
              label="Pass rate"
              value={formatNumber(sample.validation_pass_rate)}
            />
            <DetailField label="Encoder seconds" value={formatNumber(sample.enc_time_s)} />
            <DetailField label="Decoder seconds" value={formatNumber(sample.dec_time_s)} />
            <DetailField
              label="Failure"
              value={sample.failure_category_normalized ?? sample.failure_category}
            />
          </section>

          <CodeBlock title="Input code" value={sample.input_code} />
          <CodeBlock title="Decoded code" value={sample.decoded_code} />
          <CodeBlock title="Decoder task" value={sample.dec_task} />
          <CodeBlock title="Error detail" value={sample.error_detail} />
        </>
      )}
    </div>
  )
}
