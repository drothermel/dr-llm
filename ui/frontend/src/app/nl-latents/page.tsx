import NlLatentsPage from '@/views/NlLatentsPage'
import {
  fetchNlLatentsFilters,
  fetchNlLatentsSamples,
} from '@/lib/api'
import { buildSamplesPath, parseListState } from '@/lib/nlLatents'
import type {
  NlLatentsFilters,
  NlLatentsSamplesResponse,
} from '@/lib/types'

export const dynamic = 'force-dynamic'

type PageProps = {
  searchParams: Promise<Record<string, string | string[] | undefined>>
}

export default async function Page({ searchParams }: PageProps) {
  const resolvedSearchParams = await searchParams
  const listState = parseListState(resolvedSearchParams)
  const samplesPath = buildSamplesPath(listState)
  let filters: NlLatentsFilters | null = null
  let samples: NlLatentsSamplesResponse | null = null
  let error: string | null = null

  try {
    const [nextFilters, nextSamples] = await Promise.all([
      fetchNlLatentsFilters(),
      fetchNlLatentsSamples(samplesPath),
    ])
    filters = nextFilters
    samples = nextSamples
  } catch (err) {
    error = err instanceof Error ? err.message : String(err)
  }

  return (
    <NlLatentsPage
      key={samplesPath}
      initialError={error}
      initialFilters={filters}
      initialListState={listState}
      initialSamples={samples}
      initialSamplesPath={samplesPath}
    />
  )
}
