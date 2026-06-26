const EFFORT_HIDDEN = new Set(['', 'default', 'none', 'auto', 'na', 'n/a'])

export function formatSeconds(value: number | null | undefined): string | null {
  if (value === null || value === undefined) return null
  return `${Number(value).toFixed(2)}s`
}

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—'
  return `${(Number(value) * 100).toFixed(1)}%`
}

export function shortDate(value: string | null | undefined): string | null {
  if (!value) return null
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

export function truncate(value: string, max: number): string {
  return value.length > max ? `${value.slice(0, max)}…` : value
}

export function effortLabel(value: string | null | undefined): string | null {
  if (!value || EFFORT_HIDDEN.has(String(value).toLowerCase())) return null
  return String(value)
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  return `${(bytes / 1024).toFixed(1)} KB`
}

export function codeStats(value: string): {
  lines: number
  chars: number
  bytes: number
} {
  return {
    lines: value.split('\n').length,
    chars: [...value].length,
    bytes: new TextEncoder().encode(value).length,
  }
}
