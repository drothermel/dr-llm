'use client'

import { useState } from 'react'

/**
 * Clipboard copy with a transient "copied" marker. Returns the currently-copied
 * value (cleared after a short delay) and a copy function.
 */
export function useCopy(): [
  string | null,
  (value: string | null | undefined) => void,
] {
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
