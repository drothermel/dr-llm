import { clsx, type ClassValue } from 'clsx'

/**
 * Merge class names. Thin wrapper over clsx so there is a single swap point if
 * we later adopt tailwind-merge. Incoming/override classes should be passed
 * last so they win in source order.
 */
export function cn(...inputs: ClassValue[]): string {
  return clsx(inputs)
}
