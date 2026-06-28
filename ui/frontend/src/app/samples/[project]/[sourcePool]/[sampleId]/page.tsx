import PublishedSampleDetailPage from '@/views/PublishedSampleDetailPage'
import { fetchPublishedSample } from '@/lib/api'
import { searchParamsToString } from '@/lib/published'
import type { PublishedSampleDetail } from '@/lib/types'

export const dynamic = 'force-dynamic'

type PageProps = {
  params: Promise<{
    project: string
    sourcePool: string
    sampleId: string
  }>
  searchParams: Promise<Record<string, string | string[] | undefined>>
}

export default async function Page({ params, searchParams }: PageProps) {
  const [{ project, sourcePool, sampleId }, resolvedSearchParams] =
    await Promise.all([params, searchParams])
  let sample: PublishedSampleDetail | null = null
  let error: string | null = null

  try {
    sample = await fetchPublishedSample(project, sourcePool, sampleId)
  } catch (err) {
    error = err instanceof Error ? err.message : String(err)
  }

  const backSearch = searchParamsToString(resolvedSearchParams)

  return (
    <PublishedSampleDetailPage
      backSearch={backSearch}
      initialError={error}
      sample={sample}
    />
  )
}
