import { useEffect, useRef, useState } from 'react'

async function parseJsonResponse(response) {
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }
  return response.json()
}

export default function useProviderModels(providerName) {
  const [models, setModels] = useState(null)
  const [modelsLoading, setModelsLoading] = useState(false)
  const [modelsError, setModelsError] = useState(null)
  const [modelSource, setModelSource] = useState(null)
  const [syncing, setSyncing] = useState(false)
  const requestSeqRef = useRef(0)
  const activeControllerRef = useRef(null)

  useEffect(() => {
    return () => {
      requestSeqRef.current += 1
      activeControllerRef.current?.abort()
    }
  }, [])

  const runRequest = async ({ path, method, kind }) => {
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
      }).then(parseJsonResponse)

      if (requestSeq !== requestSeqRef.current) {
        return
      }

      setModels(data.models)
      setModelSource(data.source)
      if (!data.success && data.error) {
        setModelsError(`Sync failed: ${data.error}`)
      }
    } catch (error) {
      if (error.name === 'AbortError' || requestSeq !== requestSeqRef.current) {
        return
      }

      const message = kind === 'sync'
        ? `Sync error: ${error.message}`
        : error.message
      setModelsError(message)
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
