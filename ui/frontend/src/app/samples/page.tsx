import { Suspense } from 'react'
import { HydrationBoundary, dehydrate } from '@tanstack/react-query'
import PublishedSamplesPage from '@/views/PublishedSamplesPage'
import Loading from './loading'
import { makeQueryClient } from '@/lib/queryClient'
import {
  fetchPublishedFiltersCached,
  fetchPublishedSamplesCached,
} from '@/lib/api'
import { DEFAULT_LIST_STATE, buildSamplesPath } from '@/lib/published'

export default async function Page() {
  const queryClient = makeQueryClient()
  const defaultPath = buildSamplesPath(DEFAULT_LIST_STATE)

  await Promise.all([
    queryClient.prefetchQuery({
      queryKey: ['published-filters'],
      queryFn: () => fetchPublishedFiltersCached(),
    }),
    queryClient.prefetchQuery({
      queryKey: ['published-samples', defaultPath],
      queryFn: () => fetchPublishedSamplesCached(defaultPath),
    }),
  ])

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <Suspense fallback={<Loading />}>
        <PublishedSamplesPage />
      </Suspense>
    </HydrationBoundary>
  )
}
