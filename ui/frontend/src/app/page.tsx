import Link from 'next/link'
import type { ReactNode } from 'react'

type Tool = {
  href: string
  title: string
  description: string
  icon: ReactNode
}

const TOOLS: Tool[] = [
  {
    href: '/providers',
    title: 'Providers & Models',
    description:
      'Inspect configured LLM providers, their availability, and the models each one exposes.',
    icon: (
      <svg
        aria-hidden="true"
        width="20"
        height="20"
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
    title: 'nl-latents',
    description:
      'Browse published NL-latent experiment samples — encoder outputs, decoded code, and validation results.',
    icon: (
      <svg
        aria-hidden="true"
        width="20"
        height="20"
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

export default function Page() {
  return (
    <div className="mx-auto w-full max-w-[760px]">
      <header className="mb-8">
        <h1 className="font-display text-[30px] leading-tight font-bold tracking-[-0.02em] text-[var(--text-primary)]">
          dr-llm
        </h1>
        <p className="mt-1.5 text-[15px] text-[var(--text-secondary)]">
          UI tools for inspecting dr-llm subsystems. Pick a workspace to get
          started.
        </p>
      </header>

      <div className="grid grid-cols-2 gap-4 max-sm:grid-cols-1">
        {TOOLS.map(tool => (
          <Link
            key={tool.href}
            href={tool.href}
            className="group flex flex-col gap-3 rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] p-5 transition-colors hover:border-[color-mix(in_oklch,var(--accent)_45%,var(--border))] hover:bg-[var(--bg-hover)]"
          >
            <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--accent-bg)] text-[var(--accent)]">
              {tool.icon}
            </span>
            <div>
              <h2 className="font-display text-[17px] font-semibold tracking-[-0.01em] text-[var(--text-primary)]">
                {tool.title}
              </h2>
              <p className="mt-1 text-[13px] leading-relaxed text-[var(--text-secondary)]">
                {tool.description}
              </p>
            </div>
            <span className="mt-auto inline-flex items-center gap-1 text-[13px] font-medium text-[var(--accent)]">
              Open
              <span
                aria-hidden="true"
                className="transition-transform duration-200 group-hover:translate-x-0.5"
              >
                →
              </span>
            </span>
          </Link>
        ))}
      </div>
    </div>
  )
}
