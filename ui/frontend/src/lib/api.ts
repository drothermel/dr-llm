import type {
  NlLatentsFilters,
  NlLatentsSampleDetail,
  NlLatentsSamplesResponse,
  ProviderStatus,
} from './types'
import { fetchJson } from './http'

const SERVER_API_BASE_URL =
  process.env.DR_LLM_UI_API_BASE_URL ?? 'http://localhost:8000'

export async function fetchServerJson<T>(path: string): Promise<T> {
  const url = new URL(path, SERVER_API_BASE_URL)
  return fetchJson<T>(url.toString())
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
