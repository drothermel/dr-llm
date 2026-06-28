'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import type { ReactNode } from 'react'
import { cn } from '@/lib/cn'

type AppShellProps = {
  children: ReactNode
}

type NavItem = {
  href: string
  label: string
  exact?: boolean
  icon: ReactNode
}

const NAV_ITEMS: NavItem[] = [
  {
    href: '/providers',
    label: 'Providers & Models',
    icon: (
      <svg
        aria-hidden="true"
        role="presentation"
        focusable="false"
        width="16"
        height="16"
        viewBox="0 0 16 16"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <rect x="2" y="2" width="5" height="5" rx="1" />
        <rect x="9" y="2" width="5" height="5" rx="1" />
        <rect x="2" y="9" width="5" height="5" rx="1" />
        <rect x="9" y="9" width="5" height="5" rx="1" />
      </svg>
    ),
  },
  {
    href: '/nl-latents',
    label: 'nl-latents',
    icon: (
      <svg
        aria-hidden="true"
        role="presentation"
        focusable="false"
        width="16"
        height="16"
        viewBox="0 0 16 16"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M3 12.5V3.5" />
        <path d="M8 12.5V3.5" />
        <path d="M13 12.5V3.5" />
        <path d="M2 5.5H14" />
        <path d="M2 10.5H14" />
      </svg>
    ),
  },
  {
    href: '/samples',
    label: 'Published samples',
    icon: (
      <svg
        aria-hidden="true"
        role="presentation"
        focusable="false"
        width="16"
        height="16"
        viewBox="0 0 16 16"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M2.5 4.5H13.5" />
        <path d="M2.5 8H13.5" />
        <path d="M2.5 11.5H13.5" />
        <path d="M5 2.5V13.5" />
        <path d="M11 2.5V13.5" />
      </svg>
    ),
  },
]

function isActive(pathname: string, item: NavItem): boolean {
  if (item.exact) {
    return pathname === item.href
  }
  return pathname === item.href || pathname.startsWith(`${item.href}/`)
}

export default function AppShell({ children }: AppShellProps) {
  const pathname = usePathname() ?? '/'

  return (
    <div className="min-h-screen md:flex">
      <nav className="z-10 flex flex-col border-b border-[var(--border)] bg-[var(--bg-secondary)] py-5 md:fixed md:inset-y-0 md:left-0 md:w-60 md:border-r md:border-b-0">
        <div className="mb-4 border-b border-[var(--border)] px-5 pb-5">
          <Link href="/" className="inline-block">
            <h1 className="font-display text-xl font-bold tracking-[-0.02em] text-[var(--accent)]">
              dr-llm
            </h1>
            <span className="ml-0.5 font-display text-[11px] font-medium tracking-[0.12em] text-[var(--text-muted)] uppercase">
              ui tools
            </span>
          </Link>
        </div>
        <div className="px-3">
          <span className="mb-1.5 block px-2 font-display text-[11px] font-semibold tracking-[0.08em] text-[var(--text-muted)] uppercase">
            Explore
          </span>
          {NAV_ITEMS.map(item => (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                'flex items-center gap-2.5 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                isActive(pathname, item)
                  ? 'bg-[var(--accent-bg)] text-[var(--accent)]'
                  : 'text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]',
              )}
            >
              {item.icon}
              {item.label}
            </Link>
          ))}
        </div>
      </nav>
      <main className="min-h-screen flex-1 px-5 py-6 md:ml-60 md:px-10 md:py-8">
        {children}
      </main>
    </div>
  )
}
