/**
 * Memoized formatters for log message content and tool call arguments.
 *
 * Uses WeakMap to cache results keyed by object reference, avoiding
 * repeated JSON.stringify/parse on the same data during re-renders.
 * Strings are returned directly (no allocation), arrays are mapped
 * fresh each call (content blocks are typically small), and object
 * results are cached by reference for the lifetime of the source data.
 */

const contentCache = new WeakMap<object, string>();

/**
 * Format message content for display.
 *
 * Handles string (fast path), array of content blocks (Anthropic-style),
 * and object types. Results for object inputs are cached by reference
 * so repeated calls in a v-for loop hit the WeakMap instead of re-stringifying.
 */
export function formatMessageContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((block) => {
        if (typeof block === "string") return block;
        if (typeof block === "object" && block !== null) {
          const obj = block as Record<string, unknown>;
          if ("text" in obj) return String(obj.text ?? "");
          if ("type" in obj && obj.type === "image") return "[Image Block]";
        }
        return JSON.stringify(block);
      })
      .join("\n");
  }
  if (typeof content === "object" && content !== null) {
    const cached = contentCache.get(content);
    if (cached !== undefined) return cached;
    const result = JSON.stringify(content);
    contentCache.set(content, result);
    return result;
  }
  if (content === null || content === undefined) return "";
  return JSON.stringify(content);
}

const toolArgsCache = new WeakMap<object, Record<string, unknown>>();
const stringArgsCache = new Map<string, Record<string, unknown>>();

/**
 * Safely parse tool call arguments into a Record for JSON display.
 *
 * Caches parsed results by reference so that the same arguments object
 * (or the same JSON string) is only parsed once. Returns an empty object
 * for unparseable input.
 */
export function parseToolArgs(args: unknown): Record<string, unknown> {
  if (typeof args === "string") {
    const cached = stringArgsCache.get(args);
    if (cached !== undefined) return cached;
    try {
      const parsed = JSON.parse(args);
      if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
        stringArgsCache.set(args, parsed as Record<string, unknown>);
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
