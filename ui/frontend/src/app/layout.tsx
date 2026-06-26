import type { Metadata } from 'next'
import type { ReactNode } from 'react'
import { Space_Grotesk, Hanken_Grotesk, Fira_Code } from 'next/font/google'
import AppShell from '@/components/AppShell'
import './globals.css'

const display = Space_Grotesk({
  subsets: ['latin'],
  weight: ['500', '600', '700'],
  variable: '--font-display-src',
  display: 'swap',
})

const sans = Hanken_Grotesk({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-sans-src',
  display: 'swap',
})

const mono = Fira_Code({
  subsets: ['latin'],
  weight: ['400', '500', '600'],
  variable: '--font-mono-src',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'dr-llm UI',
  description: 'UI tools for inspecting dr-llm subsystems.',
}

type RootLayoutProps = {
  children: ReactNode
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html
      lang="en"
      className={`${display.variable} ${sans.variable} ${mono.variable}`}
    >
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  )
}
