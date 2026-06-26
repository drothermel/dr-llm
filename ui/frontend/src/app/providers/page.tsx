import ProvidersPage from '@/views/ProvidersPage'
import { fetchProviders } from '@/lib/api'
import type { ProviderStatus } from '@/lib/types'

export const dynamic = 'force-dynamic'

export default async function Page() {
  let providers: ProviderStatus[] = []
  let error: string | null = null

  try {
    providers = await fetchProviders()
  } catch (err) {
    error = err instanceof Error ? err.message : String(err)
  }

  return <ProvidersPage initialProviders={providers} initialError={error} />
}
