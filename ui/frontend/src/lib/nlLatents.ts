export const ALL = '__all__'
export const DEFAULT_PAGE = 1
export const PAGE_SIZE = 20
export const FALSE_PARAM = 'false'
export const TRUE_PARAM = 'true'

export type NlLatentsListState = {
  page: number
  family: string
  difficulty: string
  split: string
  encModel: string
  budget: string
  dataVersion: string
  result: string
  hidePending: boolean
  hideSmoke: boolean
}

/** Default (unfiltered, page 1) state — matches `parseListState` of empty params. */
export const DEFAULT_LIST_STATE: NlLatentsListState = {
  page: DEFAULT_PAGE,
  family: ALL,
  difficulty: ALL,
  split: ALL,
  encModel: ALL,
  budget: ALL,
  dataVersion: ALL,
  result: ALL,
  hidePending: true,
  hideSmoke: true,
}

type QueryEntry = [keyof NlLatentsListState, string]

const QUERY_CONFIG: QueryEntry[] = [
  ['family', 'family'],
  ['difficulty', 'difficulty'],
  ['split', 'split'],
  ['encModel', 'enc_model'],
  ['budget', 'budget'],
  ['dataVersion', 'data_version'],
  ['result', 'result'],
]

type SearchParamReader = {
  get(key: string): string | null
}

export type SearchParamInput =
  | SearchParamReader
  | Record<string, string | string[] | undefined>

function getSearchParam(params: SearchParamInput, key: string): string | null {
  if ('get' in params && typeof params.get === 'function') {
    return params.get(key)
  }

  const record = params as Record<string, string | string[] | undefined>
  const value = record[key]
  if (Array.isArray(value)) {
    return value[0] ?? null
  }
  return value ?? null
}

export function parsePage(params: SearchParamInput): number {
  const page = Number(getSearchParam(params, 'page'))
  return Number.isInteger(page) && page > 0 ? page : DEFAULT_PAGE
}

export function parseListState(
  searchParams: SearchParamInput,
): NlLatentsListState {
  return {
    page: parsePage(searchParams),
    family: getSearchParam(searchParams, 'family') || ALL,
    difficulty: getSearchParam(searchParams, 'difficulty') || ALL,
    split: getSearchParam(searchParams, 'split') || ALL,
    encModel: getSearchParam(searchParams, 'enc_model') || ALL,
    budget: getSearchParam(searchParams, 'budget') || ALL,
    dataVersion: getSearchParam(searchParams, 'data_version') || ALL,
    result: getSearchParam(searchParams, 'result') || ALL,
    hidePending: getSearchParam(searchParams, 'hide_pending') !== FALSE_PARAM,
    hideSmoke: getSearchParam(searchParams, 'hide_smoke') !== FALSE_PARAM,
  }
}

export function listStateToSearchParams(
  state: NlLatentsListState,
): URLSearchParams {
  const params = new URLSearchParams()
  if (state.page !== DEFAULT_PAGE) params.set('page', String(state.page))

  for (const [stateKey, paramKey] of QUERY_CONFIG) {
    if (state[stateKey] !== ALL) {
      params.set(paramKey, String(state[stateKey]))
    }
  }

  if (!state.hidePending) params.set('hide_pending', FALSE_PARAM)
  if (!state.hideSmoke) params.set('hide_smoke', FALSE_PARAM)
  return params
}

export function buildSamplesPath(state: NlLatentsListState): string {
  const params = new URLSearchParams()
  params.set('page', String(state.page))
  params.set('limit', String(PAGE_SIZE))

  for (const [stateKey, paramKey] of QUERY_CONFIG) {
    if (state[stateKey] !== ALL) {
      params.set(paramKey, String(state[stateKey]))
    }
  }

  if (state.hidePending) params.set('hide_pending', TRUE_PARAM)
  if (state.hideSmoke) params.set('hide_smoke', TRUE_PARAM)
  return `/api/nl-latents/samples?${params}`
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
