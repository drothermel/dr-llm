import { fetchJson } from '@/lib/http'
import type { NlLatentsFilters, NlLatentsSamplesResponse } from '@/lib/types'

export const FILTERS_PATH = '/api/nl-latents/filters'

export const filtersQueryKey = ['nl-latents', 'filters'] as const

export function samplesQueryKey(
  path: string,
): readonly ['nl-latents', 'samples', string] {
  return ['nl-latents', 'samples', path] as const
}

/** Client-side query options (relative fetch) shared by useQuery + prefetch. */
export function filtersQueryOptions() {
  return {
    queryKey: filtersQueryKey,
    queryFn: (): Promise<NlLatentsFilters> =>
      fetchJson<NlLatentsFilters>(FILTERS_PATH),
  }
}

export function samplesQueryOptions(path: string) {
  return {
    queryKey: samplesQueryKey(path),
    queryFn: (): Promise<NlLatentsSamplesResponse> =>
      fetchJson<NlLatentsSamplesResponse>(path),
  }
}
