import { Suspense } from 'react'
import { HydrationBoundary, dehydrate } from '@tanstack/react-query'
import NlLatentsPage from '@/views/NlLatentsPage'
import Loading from './loading'
import { makeQueryClient } from '@/lib/queryClient'
import {
  fetchNlLatentsFiltersCached,
  fetchNlLatentsSamplesCached,
} from '@/lib/api'
import { DEFAULT_LIST_STATE, buildSamplesPath } from '@/lib/nlLatents'
import { filtersQueryKey, samplesQueryKey } from '@/lib/nlLatentsQueries'

/**
 * Static shell: SSR-prefetch the default (unfiltered, page 1) view with a
 * revalidating fetch and hydrate it. The route stays static so back-navigation
 * is served from the Router Cache; the client (NlLatentsPage) reads the URL and
 * owns all filtered fetching via TanStack Query.
 */
export default async function Page() {
  const queryClient = makeQueryClient()
  const defaultPath = buildSamplesPath(DEFAULT_LIST_STATE)

  await Promise.all([
    queryClient.prefetchQuery({
      queryKey: filtersQueryKey,
      queryFn: () => fetchNlLatentsFiltersCached(),
    }),
    queryClient.prefetchQuery({
      queryKey: samplesQueryKey(defaultPath),
      queryFn: () => fetchNlLatentsSamplesCached(defaultPath),
    }),
  ])

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <Suspense fallback={<Loading />}>
        <NlLatentsPage />
      </Suspense>
    </HydrationBoundary>
  )
}
