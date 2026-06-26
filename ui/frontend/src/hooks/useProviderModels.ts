'use client'

import { useEffect, useRef, useState } from 'react'
import type { ModelEntry, SyncResultResponse } from '@/lib/types'

type ProviderModelsPayload = {
  models: ModelEntry[]
  source: string
  success?: boolean
  error?: string | null
}

type RequestKind = 'load' | 'sync'

type RunRequestInput = {
  path: string
  method: 'GET' | 'POST'
  kind: RequestKind
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }
  return response.json() as Promise<T>
}

export default function useProviderModels(providerName: string) {
  const [models, setModels] = useState<ModelEntry[] | null>(null)
  const [modelsLoading, setModelsLoading] = useState(false)
  const [modelsError, setModelsError] = useState<string | null>(null)
  const [modelSource, setModelSource] = useState<string | null>(null)
  const [syncing, setSyncing] = useState(false)
  const requestSeqRef = useRef(0)
  const activeControllerRef = useRef<AbortController | null>(null)

  useEffect(() => {
    return () => {
      requestSeqRef.current += 1
      activeControllerRef.current?.abort()
    }
  }, [])

  const runRequest = async ({ path, method, kind }: RunRequestInput) => {
    activeControllerRef.current?.abort()

    const controller = new AbortController()
    const requestSeq = requestSeqRef.current + 1

    requestSeqRef.current = requestSeq
    activeControllerRef.current = controller

    if (kind === 'load') {
      setModelsLoading(true)
    } else {
      setModelsLoading(models === null)
      setSyncing(true)
    }
    setModelsError(null)

    try {
      const data = await fetch(path, {
        method,
        signal: controller.signal,
      }).then(response =>
        parseJsonResponse<ProviderModelsPayload | SyncResultResponse>(
          response,
        ),
      )

      if (requestSeq !== requestSeqRef.current) {
        return
      }

      setModels(data.models)
      setModelSource(data.source)
      if ('success' in data && !data.success && data.error) {
        setModelsError(`Sync failed: ${data.error}`)
      }
    } catch (error) {
      if (
        error instanceof DOMException &&
        error.name === 'AbortError'
      ) {
        return
      }
      if (requestSeq !== requestSeqRef.current) {
        return
      }

      const message =
        error instanceof Error ? error.message : String(error)
      setModelsError(kind === 'sync' ? `Sync error: ${message}` : message)
    } finally {
      if (requestSeq === requestSeqRef.current) {
        activeControllerRef.current = null
        setModelsLoading(false)
        setSyncing(false)
      }
    }
  }

  const loadModels = async () => {
    if (models !== null || modelsLoading || syncing) {
      return
    }

    await runRequest({
      path: `/api/providers/${providerName}/models`,
      method: 'GET',
      kind: 'load',
    })
  }

  const syncModels = async () => {
    if (syncing) {
      return
    }

    await runRequest({
      path: `/api/providers/${providerName}/sync`,
      method: 'POST',
      kind: 'sync',
    })
  }

  return {
    models,
    modelsLoading,
    modelsError,
    modelSource,
    syncing,
    loadModels,
    syncModels,
  }
}
