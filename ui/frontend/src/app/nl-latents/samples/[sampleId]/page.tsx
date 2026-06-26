import NlLatentsDetailPage from '@/views/NlLatentsDetailPage'
import { fetchNlLatentsSample } from '@/lib/api'
import { searchParamsToString } from '@/lib/nlLatents'
import type { NlLatentsSampleDetail } from '@/lib/types'

export const dynamic = 'force-dynamic'

type PageProps = {
  params: Promise<{ sampleId: string }>
  searchParams: Promise<Record<string, string | string[] | undefined>>
}

export default async function Page({ params, searchParams }: PageProps) {
  const [{ sampleId }, resolvedSearchParams] = await Promise.all([
    params,
    searchParams,
  ])
  let sample: NlLatentsSampleDetail | null = null
  let error: string | null = null

  try {
    sample = await fetchNlLatentsSample(sampleId)
  } catch (err) {
    error = err instanceof Error ? err.message : String(err)
  }

  const backSearch = searchParamsToString(resolvedSearchParams)

  return (
    <NlLatentsDetailPage
      backSearch={backSearch}
      initialError={error}
      sample={sample}
    />
  )
}
