import { QueryClient } from '@tanstack/react-query'

/**
 * Shared QueryClient factory. Used as a browser singleton in the provider and
 * created per-request on the server for SSR prefetch/dehydrate.
 */
export function makeQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 60_000,
        gcTime: 15 * 60_000,
        refetchOnWindowFocus: false,
        retry: false,
      },
    },
  })
}
