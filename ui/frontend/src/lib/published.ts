export const ALL = ''
export const DEFAULT_PAGE = 1
export const DEFAULT_LIMIT = 25

export type PublishedListState = {
  page: number
  project: string
  sourcePool: string
  sampleRole: string
  taskFamily: string
  model: string
  result: string
  dataset: string
  hidePending: boolean
  hideSmoke: boolean
}

export const DEFAULT_LIST_STATE: PublishedListState = {
  page: DEFAULT_PAGE,
  project: ALL,
  sourcePool: ALL,
  sampleRole: ALL,
  taskFamily: ALL,
  model: ALL,
  result: ALL,
  dataset: ALL,
  hidePending: false,
  hideSmoke: true,
}

function parsePage(value: string | null): number {
  if (!value) return DEFAULT_PAGE
  const parsed = Number.parseInt(value, 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_PAGE
}

function parseBoolean(value: string | null, fallback: boolean): boolean {
  if (value === null) return fallback
  return value === '1' || value === 'true'
}

export function parseListState(params: URLSearchParams): PublishedListState {
  return {
    page: parsePage(params.get('page')),
    project: params.get('project') ?? ALL,
    sourcePool: params.get('source_pool') ?? ALL,
    sampleRole: params.get('sample_role') ?? ALL,
    taskFamily: params.get('task_family') ?? ALL,
    model: params.get('model') ?? ALL,
    result: params.get('result') ?? ALL,
    dataset: params.get('dataset') ?? ALL,
    hidePending: parseBoolean(
      params.get('hide_pending'),
      DEFAULT_LIST_STATE.hidePending,
    ),
    hideSmoke: parseBoolean(
      params.get('hide_smoke'),
      DEFAULT_LIST_STATE.hideSmoke,
    ),
  }
}

export function listStateToSearchParams(
  state: PublishedListState,
): URLSearchParams {
  const params = new URLSearchParams()
  if (state.page !== DEFAULT_PAGE) params.set('page', String(state.page))
  if (state.project) params.set('project', state.project)
  if (state.sourcePool) params.set('source_pool', state.sourcePool)
  if (state.sampleRole) params.set('sample_role', state.sampleRole)
  if (state.taskFamily) params.set('task_family', state.taskFamily)
  if (state.model) params.set('model', state.model)
  if (state.result) params.set('result', state.result)
  if (state.dataset) params.set('dataset', state.dataset)
  if (state.hidePending) params.set('hide_pending', '1')
  if (state.hideSmoke !== DEFAULT_LIST_STATE.hideSmoke) {
    params.set('hide_smoke', state.hideSmoke ? '1' : '0')
  }
  return params
}

export function buildSamplesPath(state: PublishedListState): string {
  const params = listStateToSearchParams(state)
  params.set('page', String(state.page))
  params.set('limit', String(DEFAULT_LIMIT))
  if (state.hideSmoke) params.set('hide_smoke', '1')
  const suffix = params.toString()
  return `/api/published/samples${suffix ? `?${suffix}` : ''}`
}

export function searchParamsToString(
  params: Record<string, string | string[] | undefined>,
): string {
  const nextParams = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (Array.isArray(value)) {
      for (const item of value) {
        nextParams.append(key, item)
      }
    } else if (value !== undefined) {
      nextParams.set(key, value)
    }
  }
  return nextParams.toString()
}
