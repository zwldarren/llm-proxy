import type { WebSearchCall, WebSearchSource } from "@/types/schemas";

/**
 * Safely parse a JSON string into a plain object.
 * Logs parse errors for debugging so failures don't silently pass `{}`.
 * Returns `{}` for non-object values (primitives, arrays, null).
 */
export function safeJsonParse(val: unknown): Record<string, unknown> {
  if (typeof val !== "string") {
    if (val && typeof val === "object" && !Array.isArray(val)) {
      return val as Record<string, unknown>;
    }
    return {};
  }
  try {
    const parsed = JSON.parse(val);
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      console.warn("safeJsonParse: JSON value is not a plain object", parsed);
      return {};
    }
    return parsed as Record<string, unknown>;
  } catch (err) {
    console.warn("safeJsonParse: Failed to parse JSON", err, val);
    return {};
  }
}

/**
 * Runtime type guard that returns a fallback if the value is not a number.
 * Handles the common pattern of `parsedData.index as number` where the field may be missing.
 */
export function numberOrDefault(val: unknown, fallback: number): number {
  return typeof val === "number" ? val : fallback;
}

/**
 * String value with fallback for null/undefined, preserving empty string.
 */
export function stringOrEmpty(val: unknown): string {
  return typeof val === "string" ? val : "";
}

/**
 * Extract `{url, title}` pairs from a raw wire list of search results,
 * skipping entries that carry neither.
 */
export function toWebSearchSources(raw: unknown): WebSearchSource[] {
  if (!Array.isArray(raw)) return [];
  const sources: WebSearchSource[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const record = item as Record<string, unknown>;
    const url = stringOrEmpty(record.url);
    const title = stringOrEmpty(record.title);
    if (url || title) sources.push({ url, title });
  }
  return sources;
}

/**
 * Web searches whose recorded result can be replayed to the model on a later
 * turn. `canReplay` encodes the protocol's notion of a complete search; a call
 * it rejects has no result to pair with the rebuilt request, so it is dropped
 * and the replayed history stays valid.
 */
export function replayableWebSearches(
  calls: WebSearchCall[] | undefined,
  canReplay: (call: WebSearchCall) => boolean
): WebSearchCall[] {
  return (calls ?? []).filter((call) => !!call?.id && canReplay(call));
}
