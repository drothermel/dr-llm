import { queryOptions } from '@tanstack/react-query'
import {
  fetchPublishedFilters,
  fetchPublishedSamples,
} from '@/lib/api'

export function filtersQueryOptions() {
  return queryOptions({
    queryKey: ['published-filters'],
    queryFn: fetchPublishedFilters,
    staleTime: 60_000,
  })
}

export function samplesQueryOptions(path: string) {
  return queryOptions({
    queryKey: ['published-samples', path],
    queryFn: () => fetchPublishedSamples(path),
    staleTime: 30_000,
  })
}
