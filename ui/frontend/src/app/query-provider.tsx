'use client'

import { QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { useState } from 'react'
import { makeQueryClient } from '@/lib/queryClient'

type QueryProviderProps = {
  children: ReactNode
}

export default function QueryProvider({ children }: QueryProviderProps) {
  const [queryClient] = useState(makeQueryClient)

  return (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}
