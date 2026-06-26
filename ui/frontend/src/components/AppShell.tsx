'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import type { ReactNode } from 'react'

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
    href: '/',
    label: 'Providers & Models',
    exact: true,
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
    <div className="app">
      <nav className="sidebar">
        <div className="sidebar-header">
          <h1 className="logo">dr-llm</h1>
          <span className="logo-sub">ui tools</span>
        </div>
        <div className="nav-section">
          <span className="nav-section-label">Explore</span>
          {NAV_ITEMS.map(item => (
            <Link
              key={item.href}
              href={item.href}
              className={`nav-link ${isActive(pathname, item) ? 'active' : ''}`}
            >
              {item.icon}
              {item.label}
            </Link>
          ))}
        </div>
      </nav>
      <main className="content">{children}</main>
    </div>
  )
}
