export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
  }
}

export async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { cache: 'no-store' })
  if (!response.ok) {
    throw new ApiError(`HTTP ${response.status}`, response.status)
  }
  return response.json() as Promise<T>
}
