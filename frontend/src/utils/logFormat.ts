/**
 * Shared helpers for the Logs I/O views and log parsers.
 *
 * Three groups:
 *  - JSON shape guards / safe stringify used by the parsers
 *  - Display formatting (bytes, char counts, one-line previews)
 *  - Memoized tool-argument parsing
 *
 * Everything here is intentionally cheap: strings are referenced, never
 * copied, and caches are bounded so multi-MB payloads cannot leak memory.
 */

// --- JSON shape guards ---------------------------------------------------

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function asString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

/** JSON.stringify that never throws; strings pass through, nullish → undefined. */
export function safeStringify(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  try {
    return typeof value === "string" ? value : JSON.stringify(value);
  } catch {
    return undefined;
  }
}

/** Parse a JSON string, falling back to the raw string when not valid JSON. */
export function parseResultOutput(output: string): unknown {
  try {
    return JSON.parse(output);
  } catch {
    return output;
  }
}

/** Whether a request/response body payload has any visible content. */
export function hasPayload(body: unknown): boolean {
  if (!body) return false;
  if (typeof body === "object") return Object.keys(body).length > 0;
  if (typeof body === "string") return body.trim().length > 0;
  return true;
}

// --- Display formatting ---------------------------------------------------

/** Human-readable byte size ("12.5 KB"); empty string for non-positive input. */
export function formatBytes(bytes?: number): string {
  if (!bytes || bytes <= 0) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/** Compact char count ("1.2k" / "840") used on collapse headers. */
export function formatCharCount(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}

/** First non-empty line, truncated with an ellipsis for one-line previews. */
export function firstLinePreview(text: string, max = 140): string {
  const firstLine = text.split("\n").find((l) => l.trim().length > 0) ?? "";
  return firstLine.length > max ? firstLine.slice(0, max - 3) + "…" : firstLine;
}

/**
 * Max body/output size (≈256 KB) before the views stop inline-rendering:
 * beyond this, JSON tree viewers build too many DOM nodes and text blocks
 * lay out too slowly, so content is sliced / rendered as text instead.
 */
export const MAX_INLINE_BYTES = 256 * 1024;

/** Whether a raw string payload exceeds the inline-render budget. Narrowing:
 * true ⇒ the payload is a string (over the limit). */
export function isOversizedPayload(value: unknown): value is string {
  return typeof value === "string" && value.length > MAX_INLINE_BYTES;
}

// --- Memoized tool-argument parsing ---------------------------------------

const toolArgsCache = new WeakMap<object, Record<string, unknown>>();

/** Strings larger than this are parsed without caching (would pin memory). */
const STRING_CACHE_ENTRY_LIMIT = 64 * 1024;

/** Hard cap on cached string entries; evicts oldest (insertion order). */
const STRING_CACHE_MAX_ENTRIES = 200;

const stringArgsCache = new Map<string, Record<string, unknown>>();

export function parseToolArgs(args: unknown): Record<string, unknown> {
  if (typeof args === "string") {
    const cacheable = args.length <= STRING_CACHE_ENTRY_LIMIT;
    const cached = cacheable ? stringArgsCache.get(args) : undefined;
    if (cached !== undefined) return cached;
    try {
      const parsed = JSON.parse(args);
      if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
        if (cacheable) {
          if (stringArgsCache.size >= STRING_CACHE_MAX_ENTRIES) {
            const oldest = stringArgsCache.keys().next().value;
            if (oldest !== undefined) stringArgsCache.delete(oldest);
          }
          stringArgsCache.set(args, parsed as Record<string, unknown>);
        }
        return parsed as Record<string, unknown>;
      }
      return {};
    } catch {
      return {};
    }
  }
  if (typeof args === "object" && args !== null) {
    const cached = toolArgsCache.get(args);
    if (cached !== undefined) return cached;
    const result = args as Record<string, unknown>;
    toolArgsCache.set(result, result);
    return result;
  }
  return {};
}
