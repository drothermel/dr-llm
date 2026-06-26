import type { NextConfig } from 'next'

const API_BASE_URL = process.env.DR_LLM_UI_API_BASE_URL ?? 'http://localhost:8000'

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${API_BASE_URL}/api/:path*`,
      },
    ]
  },
}

export default nextConfig
