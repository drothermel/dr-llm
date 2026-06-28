import type {
  NlLatentsFilters,
  NlLatentsSampleDetail,
  NlLatentsSamplesResponse,
  PublishedFilters,
  PublishedSampleDetail,
  PublishedSamplesResponse,
  ProviderStatus,
} from './types'
import { ApiError, fetchJson } from './http'

const SERVER_API_BASE_URL =
  process.env.DR_LLM_UI_API_BASE_URL ?? 'http://localhost:8000'

export async function fetchServerJson<T>(path: string): Promise<T> {
  const url = new URL(path, SERVER_API_BASE_URL)
  return fetchJson<T>(url.toString())
}

/**
 * Server fetch with Next revalidation instead of `no-store`, so callers (the
 * static nl-latents shell) are not opted into dynamic rendering.
 */
export async function fetchServerJsonCached<T>(
  path: string,
  revalidateSeconds = 30,
): Promise<T> {
  const url = new URL(path, SERVER_API_BASE_URL)
  const response = await fetch(url.toString(), {
    next: { revalidate: revalidateSeconds },
  })
  if (!response.ok) {
    throw new ApiError(`HTTP ${response.status}`, response.status)
  }
  return response.json() as Promise<T>
}

export async function fetchNlLatentsFiltersCached(): Promise<NlLatentsFilters> {
  return fetchServerJsonCached<NlLatentsFilters>('/api/nl-latents/filters')
}

export async function fetchNlLatentsSamplesCached(
  path: string,
): Promise<NlLatentsSamplesResponse> {
  return fetchServerJsonCached<NlLatentsSamplesResponse>(path)
}

export async function fetchProviders(): Promise<ProviderStatus[]> {
  return fetchServerJson<ProviderStatus[]>('/api/providers')
}

export async function fetchNlLatentsFilters(): Promise<NlLatentsFilters> {
  return fetchServerJson<NlLatentsFilters>('/api/nl-latents/filters')
}

export async function fetchNlLatentsSamples(
  path: string,
): Promise<NlLatentsSamplesResponse> {
  return fetchServerJson<NlLatentsSamplesResponse>(path)
}

export async function fetchNlLatentsSample(
  sampleId: string,
): Promise<NlLatentsSampleDetail> {
  return fetchServerJson<NlLatentsSampleDetail>(
    `/api/nl-latents/samples/${encodeURIComponent(sampleId)}`,
  )
}

export async function fetchPublishedFiltersCached(): Promise<PublishedFilters> {
  return fetchServerJsonCached<PublishedFilters>('/api/published/filters')
}

export async function fetchPublishedSamplesCached(
  path: string,
): Promise<PublishedSamplesResponse> {
  return fetchServerJsonCached<PublishedSamplesResponse>(path)
}

export async function fetchPublishedFilters(): Promise<PublishedFilters> {
  return fetchServerJson<PublishedFilters>('/api/published/filters')
}

export async function fetchPublishedSamples(
  path: string,
): Promise<PublishedSamplesResponse> {
  return fetchServerJson<PublishedSamplesResponse>(path)
}

export async function fetchPublishedSample(
  project: string,
  sourcePool: string,
  sampleId: string,
): Promise<PublishedSampleDetail> {
  return fetchServerJson<PublishedSampleDetail>(
    `/api/published/samples/${encodeURIComponent(project)}/${encodeURIComponent(
      sourcePool,
    )}/${encodeURIComponent(sampleId)}`,
  )
}
